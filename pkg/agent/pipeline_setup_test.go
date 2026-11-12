package agent

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/media"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// compactThenLoadImageProvider answers the summary call of a forced
// compaction, asks for load_image on the turn's first call, and keeps the
// messages of the turn's second call.
type compactThenLoadImageProvider struct {
	path         string
	summaryCalls int
	turnCalls    int
	secondCall   []providers.Message
}

func (p *compactThenLoadImageProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	if len(messages) == 1 && strings.HasPrefix(messages[0].Content, summaryPromptPrefix) {
		p.summaryCalls++
		return &providers.LLMResponse{Content: "Earlier turns were routine status checks."}, nil
	}

	p.turnCalls++
	if p.turnCalls == 1 {
		return &providers.LLMResponse{
			ToolCalls: []providers.ToolCall{{
				ID:        "call_load_image_after_compaction",
				Type:      "function",
				Name:      "load_image",
				Arguments: map[string]any{"path": p.path},
			}},
		}, nil
	}

	p.secondCall = append([]providers.Message(nil), messages...)
	return &providers.LLMResponse{Content: "I can see the screenshot."}, nil
}

func (p *compactThenLoadImageProvider) GetDefaultModel() string {
	return "compact-then-load-image"
}

func writeTestJPEG(t *testing.T, path string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := range 8 {
		for x := range 8 {
			img.Set(x, y, color.RGBA{R: 200, G: 80, B: 40, A: 255})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer f.Close()
	if err := jpeg.Encode(f, img, nil); err != nil {
		t.Fatalf("jpeg.Encode() error = %v", err)
	}
}

// A turn that starts over the context budget runs the forced compaction in
// SetupTurn, which rebuilds the messages from a shorter history. An image a
// tool loads later in that turn must still reach the model.
func TestSetupTurn_ForcedCompactionKeepsCurrentTurnToolImage(t *testing.T) {
	workspace := t.TempDir()
	imagePath := filepath.Join(workspace, "screenshot.jpg")
	writeTestJPEG(t, imagePath)

	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         workspace,
				ModelName:         "test-model",
				MaxTokens:         4096,
				ContextWindow:     40000,
				MaxToolIterations: 3,
			},
		},
		Tools: config.ToolsConfig{
			LoadImage: config.ToolConfig{Enabled: true},
		},
	}

	provider := &compactThenLoadImageProvider{path: imagePath}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	al.SetMediaStore(media.NewFileMediaStore())
	agent := al.registry.GetDefaultAgent()
	if agent == nil {
		t.Fatal("expected default agent")
	}

	sessionKey := "session-compaction-image"
	filler := strings.Repeat("routine status line. ", 200)
	history := make([]providers.Message, 0, 60)
	for i := range 30 {
		history = append(history,
			providers.Message{Role: "user", Content: fmt.Sprintf("Old question %d: %s", i, filler)},
			providers.Message{Role: "assistant", Content: fmt.Sprintf("Old answer %d: %s", i, filler)},
		)
	}
	agent.Sessions.SetHistory(sessionKey, history)

	ctx, cancel := context.WithTimeout(context.Background(), responseTimeout)
	defer cancel()
	resp, err := al.runAgentLoop(ctx, agent, processOptions{
		SessionKey:      sessionKey,
		Channel:         "cli",
		ChatID:          "direct",
		UserMessage:     "Load the screenshot and describe it.",
		DefaultResponse: defaultResponse,
		EnableSummary:   false,
		SendResponse:    false,
	})
	if err != nil {
		t.Fatalf("runAgentLoop() error = %v", err)
	}
	if resp != "I can see the screenshot." {
		t.Fatalf("response = %q, want %q", resp, "I can see the screenshot.")
	}
	if provider.summaryCalls == 0 {
		t.Fatal("expected the turn to start with a forced compaction")
	}
	if provider.turnCalls != 2 {
		t.Fatalf("turn calls = %d, want 2", provider.turnCalls)
	}

	followUp := -1
	for i, msg := range provider.secondCall {
		if msg.Role == "user" && msg.Content == "[Loaded image from tool result above]" {
			followUp = i
		}
	}
	if followUp < 0 {
		t.Fatalf("second call has no tool image follow-up message; the image was dropped. Last message: %#v",
			provider.secondCall[len(provider.secondCall)-1])
	}
	if prev := provider.secondCall[followUp-1]; prev.Role != "tool" ||
		!strings.Contains(prev.Content, "[image:"+imagePath+"]") {
		t.Fatalf("message before the follow-up = %#v, want the load_image tool result", prev)
	}
	msg := provider.secondCall[followUp]
	if len(msg.Media) != 1 || !strings.HasPrefix(msg.Media[0], "data:image/jpeg;base64,") {
		t.Fatalf("follow-up media = %v, want one data:image/jpeg URL", msg.Media)
	}
}
