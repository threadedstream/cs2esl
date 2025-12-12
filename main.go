package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// --------- Domain model (your events) ---------

type Cs2EventType string

const (
	EventKill       Cs2EventType = "KILL"
	EventDeath      Cs2EventType = "DEATH"
	EventRoundStart Cs2EventType = "ROUND_START"
	EventRoundEnd   Cs2EventType = "ROUND_END"
)

type Cs2Event struct {
	Type      Cs2EventType   `json:"type"`
	Player    string         `json:"player"`
	Target    string         `json:"target,omitempty"`
	Weapon    string         `json:"weapon,omitempty"`
	Map       string         `json:"map,omitempty"`
	Timestamp time.Time      `json:"timestamp"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// --------- GSI payload (subset of real schema) ---------

type GsiPayload struct {
	Map struct {
		Name string `json:"name"`
	} `json:"map"`

	Round struct {
		Phase   string `json:"phase"` // e.g. "freezetime", "live", "over"
		WinTeam string `json:"win_team,omitempty"`
	} `json:"round"`

	Player struct {
		Name  string `json:"name"`
		Team  string `json:"team"`
		State struct {
			Health      int  `json:"health"`
			Armor       int  `json:"armor"`
			Helmet      bool `json:"helmet"`
			RoundKills  int  `json:"round_kills"`
			RoundKillHS int  `json:"round_killhs"`
		} `json:"state"`
		MatchStats struct {
			Kills   int `json:"kills"`
			Assists int `json:"assists"`
			Deaths  int `json:"deaths"`
			Mvps    int `json:"mvps"`
			Score   int `json:"score"`
		} `json:"match_stats"`
	} `json:"player"`
}

// --------- Event processor (sliding window) ---------

type EventProcessor struct {
	mu     sync.Mutex
	events []Cs2Event
	maxLen int
}

func NewEventProcessor(maxLen int) *EventProcessor {
	return &EventProcessor{
		events: make([]Cs2Event, 0, maxLen),
		maxLen: maxLen,
	}
}

func (p *EventProcessor) AddEvent(evt Cs2Event) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.events = append(p.events, evt)
	if len(p.events) > p.maxLen {
		p.events = p.events[len(p.events)-p.maxLen:]
	}
}

func (p *EventProcessor) Snapshot() []Cs2Event {
	p.mu.Lock()
	defer p.mu.Unlock()

	out := make([]Cs2Event, len(p.events))
	copy(out, p.events)
	return out
}

// --------- LLM integration (real OpenAI HTTP) ---------

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message openAIChatMessage `json:"message"`
	} `json:"choices"`
}

type LlmResponse struct {
	Commentary string `json:"commentary"`
}

func callLlm(ctx context.Context, events []Cs2Event) (LlmResponse, error) {
	eventsJSON, _ := json.Marshal(events)

	prompt := fmt.Sprintf(`
You are an ESL-style Counter-Strike 2 commentator.
Be energetic, concise, and analytical.
React to the following events in real time.

Events JSON:
%s

If map's name starts with de_, ignore it. For example, if it's de_Mirage, call it Mirage
Give hype commentary.
Max 2 sentences.
`, string(eventsJSON))

	reqBody := map[string]any{
		"model":  "llama3.1:8b",
		"prompt": prompt,
		"stream": false,
	}

	body, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		"http://127.0.0.1:11434/api/generate",
		bytes.NewReader(body),
	)
	if err != nil {
		return LlmResponse{}, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return LlmResponse{}, err
	}
	defer resp.Body.Close()

	var ollamaResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return LlmResponse{}, err
	}

	return LlmResponse{Commentary: ollamaResp.Response}, nil
}

// --------- Global state: previous GSI snapshot & latest commentary ---------

var (
	processor = NewEventProcessor(20)
	prevGsiMu sync.Mutex
	prevGsi   *GsiPayload
)

// --------- GSI handler: real integration with CS2 ---------

func handleGsi(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}

	var payload GsiPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		log.Printf("Failed to parse GSI JSON: %v\nBody: %s\n", err, string(body))
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	now := time.Now()
	mapName := payload.Map.Name
	playerName := payload.Player.Name

	events := diffGsiToEvents(&payload, now, mapName, playerName)
	for _, evt := range events {
		processor.AddEvent(evt)
		log.Printf("Event detected: %+v\n", evt)
	}

	// Respond quickly; CS2 doesn't care about body, just status.
	w.WriteHeader(http.StatusNoContent)
}

// diffGsiToEvents looks at previous payload vs current and emits logical events.
func diffGsiToEvents(curr *GsiPayload, now time.Time, mapName, playerName string) []Cs2Event {
	prevGsiMu.Lock()
	defer prevGsiMu.Unlock()

	var out []Cs2Event

	// If no previous snapshot -> maybe just detect round start if live
	if prevGsi == nil {
		if curr.Round.Phase == "live" {
			out = append(out, Cs2Event{
				Type:      EventRoundStart,
				Player:    playerName,
				Map:       mapName,
				Timestamp: now,
				Metadata: map[string]any{
					"phase": "live",
				},
			})
		}
		prevGsi = curr
		return out
	}

	// Round phase changes
	if curr.Round.Phase != prevGsi.Round.Phase {
		switch curr.Round.Phase {
		case "live":
			out = append(out, Cs2Event{
				Type:      EventRoundStart,
				Player:    playerName,
				Map:       mapName,
				Timestamp: now,
				Metadata: map[string]any{
					"prev_phase": prevGsi.Round.Phase,
				},
			})
		case "over":
			out = append(out, Cs2Event{
				Type:      EventRoundEnd,
				Player:    playerName,
				Map:       mapName,
				Timestamp: now,
				Metadata: map[string]any{
					"winner":     curr.Round.WinTeam,
					"prev_phase": prevGsi.Round.Phase,
				},
			})
		}
	}

	// Match stats kill/death deltas
	dKills := curr.Player.MatchStats.Kills - prevGsi.Player.MatchStats.Kills
	dDeaths := curr.Player.MatchStats.Deaths - prevGsi.Player.MatchStats.Deaths

	for i := 0; i < dKills; i++ {
		out = append(out, Cs2Event{
			Type:      EventKill,
			Player:    playerName,
			Map:       mapName,
			Timestamp: now,
			Metadata: map[string]any{
				"total_kills": curr.Player.MatchStats.Kills,
			},
		})
	}

	for i := 0; i < dDeaths; i++ {
		out = append(out, Cs2Event{
			Type:      EventDeath,
			Player:    playerName,
			Map:       mapName,
			Timestamp: now,
			Metadata: map[string]any{
				"total_deaths": curr.Player.MatchStats.Deaths,
			},
		})
	}

	// Save current as previous for next diff
	prevGsi = curr
	return out
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Background goroutine: periodically send recent events to LLM
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				events := processor.Snapshot()
				if len(events) == 0 {
					continue
				}

				resp, err := callLlm(ctx, events)
				if err != nil {
					log.Printf("LLM error: %v\n", err)
					continue
				}
				log.Printf("LLM commentary: %s\n", resp.Commentary)
			}
		}
	}()

	http.HandleFunc("/cs2-gsi", handleGsi)

	addr := ":8080"
	log.Printf("Listening on %s (GSI endpoint: http://127.0.0.1%s/cs2-gsi)", addr, addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
