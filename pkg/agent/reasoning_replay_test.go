package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// reasoningTurnProvider answers the first call of each turn with a tool call
// that carries an encrypted reasoning item, and the second with a final
// answer. It records the messages of every call.
type reasoningTurnProvider struct {
	mu    sync.Mutex
	calls [][]providers.Message
}

func (p *reasoningTurnProvider) Chat(
	_ context.Context,
	messages []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls = append(p.calls, append([]providers.Message(nil), messages...))
	if len(p.calls)%2 == 1 {
		return &providers.LLMResponse{
			Content: "Checking the file.",
			Phase:   "commentary",
			ToolCalls: []providers.ToolCall{{
				ID:        "call_" + string(rune('a'+len(p.calls))),
				Type:      "function",
				Name:      "echo_text",
				Arguments: map[string]any{"text": "hi"},
			}},
			ReasoningItems: []providers.ReasoningItem{{
				ID: "rs_turn", EncryptedContent: "ENCRYPTED-REASONING", Model: "test-model",
			}},
			FinishReason: "tool_calls",
		}, nil
	}
	return &providers.LLMResponse{Content: "Done.", Phase: "final_answer", FinishReason: "stop"}, nil
}

func (p *reasoningTurnProvider) GetDefaultModel() string { return "test-model" }

func assistantReasoningIn(messages []providers.Message) (items int, phase string) {
	for _, m := range messages {
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			items += len(m.ReasoningItems)
			phase = m.Phase
		}
	}
	return items, phase
}

// Within a turn, the reasoning a tool call came with rides along to the next
// call; it is never written to the session, so a later turn starts without it.
func TestReasoningItems_ReplayedWithinTurnNotPersisted(t *testing.T) {
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	workspace := t.TempDir()
	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         workspace,
				ModelName:         "test-model",
				MaxTokens:         4096,
				MaxToolIterations: 5,
			},
		},
	}
	provider := &reasoningTurnProvider{}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	al.RegisterTool(&echoTextTool{})
	agent := al.GetRegistry().GetDefaultAgent()

	run := func(text string) {
		t.Helper()
		if _, err := al.runAgentLoop(context.Background(), agent, processOptions{
			SessionKey:      "reasoning-session",
			UserMessage:     text,
			DefaultResponse: defaultResponse,
		}); err != nil {
			t.Fatalf("runAgentLoop: %v", err)
		}
	}

	run("first")
	if len(provider.calls) != 2 {
		t.Fatalf("calls after turn 1 = %d, want 2", len(provider.calls))
	}
	if items, phase := assistantReasoningIn(provider.calls[1]); items != 1 || phase != "commentary" {
		t.Fatalf("second call of the turn: reasoning items = %d, phase = %q; want 1, commentary", items, phase)
	}

	run("second")
	if len(provider.calls) != 4 {
		t.Fatalf("calls after turn 2 = %d, want 4", len(provider.calls))
	}
	// Turn 2's first call sees turn 1 from the session: phase kept,
	// reasoning items gone.
	items, phase := assistantReasoningIn(provider.calls[2])
	if items != 0 {
		t.Fatalf("turn 2 first call carries %d reasoning items from turn 1, want 0", items)
	}
	if phase != "commentary" {
		t.Fatalf("turn 1 assistant phase after reload = %q, want commentary", phase)
	}

	// The encrypted reasoning never reaches disk.
	files, _ := filepath.Glob(filepath.Join(workspace, "sessions", "*"))
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "ENCRYPTED-REASONING") {
			t.Fatalf("encrypted reasoning persisted in %s", f)
		}
	}
	if len(files) == 0 {
		t.Fatal("no session files written; the persistence check proved nothing")
	}
}

// The after_llm process hook gets the LLM response as JSON on every call. The
// encrypted reasoning stays in the runtime: it is not in that payload.
func TestReasoningItems_NotInAfterLLMHookPayload(t *testing.T) {
	// Typed as the process hook's own call() takes it (params any).
	var params any = &LLMHookResponse{
		Model: "test-model",
		Response: &providers.LLMResponse{
			Content: "Checking the file.",
			Phase:   "commentary",
			ReasoningItems: []providers.ReasoningItem{{
				ID: "rs_hook", EncryptedContent: "ENCRYPTED-REASONING", Model: "test-model",
			}},
		},
	}
	payload, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal hook payload: %v", err)
	}
	if strings.Contains(string(payload), "ENCRYPTED-REASONING") || strings.Contains(string(payload), "rs_hook") {
		t.Fatalf("after_llm hook payload carries encrypted reasoning: %s", payload)
	}
	if !strings.Contains(string(payload), `"phase":"commentary"`) {
		t.Fatalf("after_llm hook payload lost the phase: %s", payload)
	}
}
