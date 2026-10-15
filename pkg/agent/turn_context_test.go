package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// TestBuildMessages_SystemPromptStableAcrossTurnContextChanges pins the cache
// contract behind the turn context message: editing MEMORY.md, writing daily
// notes, or replacing the summary between turns must leave the system message
// byte-identical (content AND structured parts). Before, both lived at the end
// of the system prompt, so OpenAI's prefix cache missed on nearly every turn's
// first call.
func TestBuildMessages_SystemPromptStableAcrossTurnContextChanges(t *testing.T) {
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	tmpDir := setupWorkspace(t, map[string]string{
		"AGENT.md":         "# Agent\nStable persona.",
		"memory/MEMORY.md": "- fact one",
	})
	cb := NewContextBuilder(tmpDir)
	history := []providers.Message{
		{Role: "user", Content: "earlier question"},
		{Role: "assistant", Content: "earlier answer"},
	}

	first := cb.BuildMessages(history, "summary v1", "turn one", nil, "pico", "chat-1", "", "")

	memPath := filepath.Join(tmpDir, "memory", "MEMORY.md")
	if err := os.WriteFile(memPath, []byte("- fact one\n- fact two"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := cb.memory.AppendToday("daily note"); err != nil {
		t.Fatal(err)
	}
	second := cb.BuildMessages(history, "summary v2", "turn two", nil, "pico", "chat-1", "", "")

	if first[0].Role != "system" || second[0].Role != "system" {
		t.Fatalf("expected leading system messages, got %q and %q", first[0].Role, second[0].Role)
	}
	if first[0].Content != second[0].Content {
		t.Fatalf("system content changed across a memory edit + summary rewrite:\nbefore: %q\nafter:  %q",
			first[0].Content, second[0].Content)
	}
	if !reflect.DeepEqual(first[0].SystemParts, second[0].SystemParts) {
		t.Fatalf("system parts changed:\nbefore: %#v\nafter:  %#v", first[0].SystemParts, second[0].SystemParts)
	}
	for _, leak := range []string{"fact one", "summary v1", "CONTEXT_SUMMARY", "# Memory"} {
		if strings.Contains(first[0].Content, leak) {
			t.Fatalf("system message contains per-turn content %q", leak)
		}
	}

	// Layout: system, history (verbatim), turn context, current user turn.
	if len(second) != 5 {
		t.Fatalf("messages len = %d, want 5", len(second))
	}
	if !reflect.DeepEqual(second[1:3], history) {
		t.Fatalf("history not passed through verbatim: %#v", second[1:3])
	}
	turnCtx := second[3]
	if turnCtx.PromptSource != string(PromptSourceTurnContext) || turnCtx.Role != "user" {
		t.Fatalf("messages[3] = %#v, want the turn context message", turnCtx)
	}
	for _, want := range []string{
		"[turn context]", "[/turn context]", "# Memory", "fact two",
		"## Recent Daily Notes", "daily note", "CONTEXT_SUMMARY:", "summary v2",
	} {
		if !strings.Contains(turnCtx.Content, want) {
			t.Errorf("turn context missing %q:\n%s", want, turnCtx.Content)
		}
	}
	if strings.Index(turnCtx.Content, "# Memory") > strings.Index(turnCtx.Content, "CONTEXT_SUMMARY:") {
		t.Error("memory should precede the summary inside the turn context message")
	}
	if !strings.HasSuffix(second[4].Content, "turn two") || second[4].Role != "user" {
		t.Fatalf("last message = %#v, want the current user turn", second[4])
	}
}

// TestBuildMessages_UserMessageSentAsStored pins fix 2 of the context audit:
// the current user message goes out byte-identical to what SetupTurn persists
// (userPromptMessage of the raw text), so the next turn re-sends it unchanged.
// The runtime context rides in the unpersisted turn context message.
func TestBuildMessages_UserMessageSentAsStored(t *testing.T) {
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	cb := NewContextBuilder(t.TempDir())

	msgs := cb.BuildMessages(nil, "", "task", nil, "pico", "chat-1", "pico-remote", "")
	if len(msgs) != 3 {
		t.Fatalf("messages len = %d, want 3 (system + turn context + user)", len(msgs))
	}
	stored := userPromptMessage("task", nil)
	if !reflect.DeepEqual(msgs[2], stored) {
		t.Fatalf("sent user message = %#v, want the persisted form %#v", msgs[2], stored)
	}
	turnCtx := assertUserTurnWithRuntimeContext(t, msgs, "task")
	if strings.Contains(turnCtx.Content, "# Memory") || strings.Contains(turnCtx.Content, "CONTEXT_SUMMARY") {
		t.Fatalf("turn context has memory/summary with none configured: %q", turnCtx.Content)
	}
}

// TestBuildMessages_TurnContextPrecedesActiveTurnOnRetryRebuild covers the
// context-overflow retry, which rebuilds with no CurrentMessage and the active
// turn's persisted messages at the end of History: the turn context message
// must land before those messages (where it sat on the turn's first call), not
// after the in-flight tool results.
func TestBuildMessages_TurnContextPrecedesActiveTurnOnRetryRebuild(t *testing.T) {
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	cb := NewContextBuilder(t.TempDir())

	earlier := []providers.Message{
		{Role: "user", Content: "old question"},
		{Role: "assistant", Content: "old answer"},
	}
	activeTurn := []providers.Message{
		{Role: "user", Content: "current question"},
		{Role: "assistant", ToolCalls: []providers.ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: &providers.FunctionCall{Name: "list_dir", Arguments: "{}"},
		}}},
		{Role: "tool", ToolCallID: "call_1", Content: "FILE: a.txt"},
	}
	history := append(append([]providers.Message(nil), earlier...), activeTurn...)

	msgs := cb.BuildMessagesFromPrompt(PromptBuildRequest{
		History:            history,
		Summary:            "older summary",
		Channel:            "pico",
		ChatID:             "chat-1",
		CurrentTurnHistory: len(activeTurn),
	})

	if len(msgs) != 1+len(earlier)+1+len(activeTurn) {
		t.Fatalf("messages len = %d, want %d", len(msgs), 1+len(earlier)+1+len(activeTurn))
	}
	turnCtxIdx := 1 + len(earlier)
	if msgs[turnCtxIdx].PromptSource != string(PromptSourceTurnContext) {
		t.Fatalf("messages[%d] = %#v, want the turn context message", turnCtxIdx, msgs[turnCtxIdx])
	}
	// With no current user message, the runtime context rides in the turn
	// context message.
	for _, want := range []string{"older summary", "[runtime context]", "Current Time"} {
		if !strings.Contains(msgs[turnCtxIdx].Content, want) {
			t.Errorf("turn context missing %q", want)
		}
	}
	if !reflect.DeepEqual(msgs[turnCtxIdx+1:], activeTurn) {
		t.Fatalf("active turn not preserved after the turn context:\n%#v", msgs[turnCtxIdx+1:])
	}
}

type allCallsProvider struct {
	calls [][]providers.Message
}

func (p *allCallsProvider) Chat(
	_ context.Context,
	messages []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.calls = append(p.calls, append([]providers.Message(nil), messages...))
	return &providers.LLMResponse{Content: fmt.Sprintf("reply %d", len(p.calls))}, nil
}

func (p *allCallsProvider) GetDefaultModel() string { return "test-model" }

// TestAgentLoop_CrossTurnPrefixStableAfterMemoryAndSummaryChange runs two real
// turns through the agent loop with a MEMORY.md edit and a new summary in
// between, and checks what the provider saw: the system message is
// byte-identical, and turn 1's user message is re-sent on turn 2 exactly as it
// was sent on turn 1 (no runtime wrapper on one side only).
func TestAgentLoop_CrossTurnPrefixStableAfterMemoryAndSummaryChange(t *testing.T) {
	cfg := &config.Config{}
	provider := &allCallsProvider{}
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	cfg.Agents.Defaults.Workspace = t.TempDir()
	cfg.Agents.Defaults.ModelName = "test-model"
	cfg.Agents.Defaults.MaxTokens = 4096
	cfg.Agents.Defaults.MaxToolIterations = 5
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	agent := al.GetRegistry().GetDefaultAgent()
	sessionKey := "agent:main:cache-test"
	memPath := filepath.Join(cfg.Agents.Defaults.Workspace, "memory", "MEMORY.md")
	if err := os.WriteFile(memPath, []byte("- fact one"), 0o600); err != nil {
		t.Fatal(err)
	}

	run := func(msg string) {
		t.Helper()
		if _, err := al.runAgentLoop(context.Background(), agent, processOptions{
			SessionKey:      sessionKey,
			UserMessage:     msg,
			DefaultResponse: defaultResponse,
		}); err != nil {
			t.Fatalf("runAgentLoop(%q) error = %v", msg, err)
		}
	}

	run("first question")
	if err := os.WriteFile(memPath, []byte("- fact one\n- fact two"), 0o600); err != nil {
		t.Fatal(err)
	}
	agent.Sessions.SetSummary(sessionKey, "a fresh summary")
	run("second question")

	if len(provider.calls) != 2 {
		t.Fatalf("provider calls = %d, want 2", len(provider.calls))
	}
	turn1, turn2 := provider.calls[0], provider.calls[1]
	if turn1[0].Content != turn2[0].Content || !reflect.DeepEqual(turn1[0].SystemParts, turn2[0].SystemParts) {
		t.Fatal("system message changed across a memory edit + summary change")
	}
	// turn 1: system, turn context, "first question"
	// turn 2: system, "first question", "reply 1", turn context, "second question"
	if len(turn2) != 5 {
		t.Fatalf("turn 2 messages = %d, want 5", len(turn2))
	}
	sent1 := turn1[len(turn1)-1]
	resent := turn2[1]
	if sent1.Role != resent.Role || sent1.Content != resent.Content {
		t.Fatalf("turn 1 user message re-sent differently:\nturn 1: %q\nturn 2: %q", sent1.Content, resent.Content)
	}
	if turn2[2].Role != "assistant" || turn2[2].Content != "reply 1" {
		t.Fatalf("turn 2 history = %#v, want turn 1's reply", turn2[2])
	}
	if !strings.Contains(turn2[3].Content, "fact two") || !strings.Contains(turn2[3].Content, "a fresh summary") {
		t.Fatalf("turn 2 turn context missing the updated memory/summary: %q", turn2[3].Content)
	}
	assertUserTurnWithRuntimeContext(t, turn2, "second question")
}
