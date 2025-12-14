package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

var (
	speechQueue = make(chan string, 10) // buffered queue
)

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

/* =========================
   GSI payload (subset)
========================= */

type GsiPayload struct {
	Map struct {
		Name string `json:"name"`
	} `json:"map"`

	Round struct {
		Phase   string `json:"phase"`
		WinTeam string `json:"win_team,omitempty"`
	} `json:"round"`

	Player struct {
		Name       string `json:"name"`
		MatchStats struct {
			Kills  int `json:"kills"`
			Deaths int `json:"deaths"`
		} `json:"match_stats"`
	} `json:"player"`
}

/* =========================
   Event processor
========================= */

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

func (p *EventProcessor) Add(evt Cs2Event) {
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

func callLLM(ctx context.Context, events []Cs2Event) (string, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY not set")
	}

	eventsJSON, _ := json.Marshal(events)

	systemPrompt := `
You are an ESL Counter-Strike play-by-play commentator.

ABSOLUTE RULES:
- NEVER explain the game.
- NEVER narrate like a recap.
- NEVER start with map names, player names, or round context.
- NEVER sound neutral.

STYLE:
- Speak like the action is unfolding RIGHT NOW.
- Assume the listener already understands CS.
- Compress meaning aggressively.
- Every word must earn its place.

DELIVERY:
- Short bursts.
- Controlled hype.
- Sentence fragments are allowed.
- Silence is better than filler.

FORMAT:
- 1 sentence for live action.
- 2 sentences max for round end.
- 6–12 words per sentence.

GOAL:
Sound like an ESL caster calling a live match, not an analyst.

Use ESL-style phrasing such as:
- "cracks it wide open"
- "no room to breathe"
- "dictating the pace"
- "isolates the fight"
- "this round is done"
But never quote them verbatim every time.
`

	userPrompt := fmt.Sprintf(`
Think in terms of:
- pressure
- timing
- spacing
- isolation
- initiative

Events JSON:
%s

If map name starts with de_, drop the prefix.
Give hype commentary.
`, string(eventsJSON))

	reqBody := openAIChatRequest{
		Model: "gpt-4.1-mini",
		Messages: []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	}

	body, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var out openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}

	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no LLM output")
	}

	return out.Choices[0].Message.Content, nil
}

func startSpeechWorker(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case text := <-speechQueue:
				// Block until speech finishes
				if err := speak(ctx, text); err != nil {
					log.Println("TTS error:", err)
				}
			}
		}
	}()
}

func speak(ctx context.Context, text string) error {
	apiKey := os.Getenv("OPENAI_API_KEY")

	reqBody := map[string]any{
		"model": "gpt-4o-mini-tts",
		"voice": "alloy",
		"input": text,
	}

	body, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		"https://api.openai.com/v1/audio/speech",
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	cmd := exec.Command(
		"ffplay",
		"-autoexit",
		"-nodisp",
		"-af", "atempo=1.38,volume=1.1",
		"-",
	)
	cmd.Stdin = resp.Body
	return cmd.Run()
}

/* =========================
   Global state
========================= */

var (
	processor = NewEventProcessor(15)
	prevMu    sync.Mutex
	prevGsi   *GsiPayload
)

/* =========================
   GSI handler
========================= */

func handleGsi(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	body, _ := io.ReadAll(r.Body)

	var payload GsiPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		w.WriteHeader(400)
		return
	}

	now := time.Now()
	player := payload.Player.Name
	mapName := payload.Map.Name

	prevMu.Lock()
	defer prevMu.Unlock()

	if prevGsi != nil {
		if payload.Player.MatchStats.Kills > prevGsi.Player.MatchStats.Kills {
			processor.Add(Cs2Event{
				Type:      EventKill,
				Player:    player,
				Map:       mapName,
				Timestamp: now,
			})
		}
		if payload.Player.MatchStats.Deaths > prevGsi.Player.MatchStats.Deaths {
			processor.Add(Cs2Event{
				Type:      EventDeath,
				Player:    player,
				Map:       mapName,
				Timestamp: now,
			})
		}
	}

	prevGsi = &payload
	w.WriteHeader(204)
}

func main() {
	ctx := context.Background()

	startSpeechWorker(ctx)

	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			events := processor.Snapshot()
			if len(events) == 0 {
				continue
			}

			text, err := callLLM(ctx, events)
			if err != nil {
				log.Println("LLM error:", err)
				continue
			}

			log.Println("Commentary:", text)

			select {
			case speechQueue <- text:
				// queued successfully
			default:
				// queue full → drop commentary (prevents lag buildup)
				log.Println("Speech queue full, dropping commentary")
			}
		}
	}()

	http.HandleFunc("/cs2-gsi", handleGsi)

	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
