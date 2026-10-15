package agent

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

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

// TestBuildMessages_NoTurnContextWithoutMemoryOrSummary keeps workers with no
// memory file and no summary on their old two-message layout.
func TestBuildMessages_NoTurnContextWithoutMemoryOrSummary(t *testing.T) {
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	cb := NewContextBuilder(t.TempDir())

	msgs := cb.BuildMessages(nil, "", "task", nil, "pico", "chat-1", "", "")
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2 (system + user)", len(msgs))
	}
	if _, ok := turnContextMessage(msgs); ok {
		t.Fatal("unexpected turn context message with no memory and no summary")
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
