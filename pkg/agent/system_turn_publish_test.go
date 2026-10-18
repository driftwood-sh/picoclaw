package agent

import (
	"context"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/tools"
)

// diffLikeTool returns user-facing output the way edit_file (a diff) and
// exec (command output) do.
type diffLikeTool struct{}

func (t *diffLikeTool) Name() string        { return "edit_like" }
func (t *diffLikeTool) Description() string { return "edits a file" }
func (t *diffLikeTool) Parameters() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}

func (t *diffLikeTool) Execute(context.Context, map[string]any) *tools.ToolResult {
	return &tools.ToolResult{
		ForLLM:  "File edited: GOALS.md",
		ForUser: "File edited: GOALS.md\n```diff\n-old\n+new\n```",
	}
}

func newSystemTurnTestLoop(t *testing.T, responses ...*providers.LLMResponse) (*AgentLoop, *bus.MessageBus) {
	t.Helper()
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         t.TempDir(),
				ModelName:         "test-model",
				MaxTokens:         4096,
				MaxToolIterations: 10,
			},
		},
	}
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, &sequenceProvider{responses: responses})
	al.RegisterTool(&diffLikeTool{})
	return al, msgBus
}

func asyncResultMessage() bus.InboundMessage {
	return bus.NormalizeInboundMessage(bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "system",
			ChatID:   "pico_client:C0123",
			ChatType: "direct",
			SenderID: "async:spawn",
		},
		Content: "Task 'scan' completed.\n\nResult:\nfive competitors found",
	})
}

func drainOutbound(msgBus *bus.MessageBus) []bus.OutboundMessage {
	var out []bus.OutboundMessage
	for {
		select {
		case msg := <-msgBus.OutboundChan():
			out = append(out, msg)
		case <-time.After(200 * time.Millisecond):
			return out
		}
	}
}

// An async-result turn publishes only the final reply the model wrote: no
// edit diffs or exec output from its tools.
func TestSystemTurn_PublishesFinalReplyOnly(t *testing.T) {
	al, msgBus := newSystemTurnTestLoop(t,
		&providers.LLMResponse{
			ToolCalls: []providers.ToolCall{{
				ID:        "call_1",
				Type:      "function",
				Name:      "edit_like",
				Arguments: map[string]any{},
			}},
			FinishReason: "tool_calls",
		},
		&providers.LLMResponse{Content: "The scan found five competitors.", FinishReason: "stop"},
	)

	if _, err := al.processSystemMessage(context.Background(), asyncResultMessage()); err != nil {
		t.Fatalf("processSystemMessage: %v", err)
	}

	out := drainOutbound(msgBus)
	if len(out) != 1 {
		contents := make([]string, 0, len(out))
		for _, m := range out {
			contents = append(contents, m.Content)
		}
		t.Fatalf("published %d messages %q, want only the final reply", len(out), contents)
	}
	got := out[0]
	if got.Content != "The scan found five competitors." {
		t.Fatalf("published %q, want the final reply", got.Content)
	}
	if got.Channel != "pico_client" || got.ChatID != "C0123" {
		t.Fatalf("published to %s/%s, want the origin pico_client/C0123", got.Channel, got.ChatID)
	}
	if got.Context.Raw[metadataKeyOutboundKind] != outboundKindFinal {
		t.Fatalf("final reply not marked final: %v", got.Context.Raw)
	}
}

// A system turn the model ends without text publishes nothing: the
// "Background task completed." placeholder is never posted.
func TestSystemTurn_EmptyReplyPublishesNothing(t *testing.T) {
	responses := []*providers.LLMResponse{{
		ToolCalls: []providers.ToolCall{{
			ID:        "call_1",
			Type:      "function",
			Name:      "edit_like",
			Arguments: map[string]any{},
		}},
		FinishReason: "tool_calls",
	}}
	// Every later call (including empty-response recovery retries) is empty.
	for i := 0; i < 10; i++ {
		responses = append(responses, &providers.LLMResponse{Content: "", FinishReason: "stop"})
	}
	al, msgBus := newSystemTurnTestLoop(t, responses...)

	reply, err := al.processSystemMessage(context.Background(), asyncResultMessage())
	if err != nil {
		t.Fatalf("processSystemMessage: %v", err)
	}
	if reply != backgroundTurnDefaultResponse {
		t.Fatalf("turn reply = %q, want the placeholder %q", reply, backgroundTurnDefaultResponse)
	}
	if out := drainOutbound(msgBus); len(out) != 0 {
		t.Fatalf("published %d messages (first %q), want none", len(out), out[0].Content)
	}
}
