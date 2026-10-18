package openai_responses_common

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

func assistantWithReasoning(model string) protocoltypes.Message {
	return protocoltypes.Message{
		Role:      "assistant",
		Content:   "Checking.",
		Phase:     "commentary",
		ToolCalls: []protocoltypes.ToolCall{{ID: "call_1", Name: "get_job", Arguments: map[string]any{}}},
		ReasoningItems: []protocoltypes.ReasoningItem{
			{ID: "rs_1", EncryptedContent: "ENC", Model: model},
			{ID: "rs_no_enc", Model: model}, // cannot be replayed with store=false
		},
	}
}

func marshalInput(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

// The plain translation (used by other Responses providers) is unchanged:
// no reasoning items, no phase.
func TestTranslateMessages_PlainReplaysNothing(t *testing.T) {
	input, _ := TranslateMessages([]protocoltypes.Message{assistantWithReasoning("m")})
	got := marshalInput(t, input)
	if strings.Contains(got, `"reasoning"`) || strings.Contains(got, `"phase"`) {
		t.Fatalf("plain translation replayed reasoning or phase: %s", got)
	}
}

func TestTranslateMessagesForModel_ReplaysOnlyUsableItems(t *testing.T) {
	input, _, stats := TranslateMessagesForModel([]protocoltypes.Message{assistantWithReasoning("m")}, "m")
	got := marshalInput(t, input)
	if stats.ReasoningItems != 1 || stats.Phases != 1 {
		t.Fatalf("stats = %+v, want 1 reasoning item and 1 phase", stats)
	}
	if !strings.Contains(got, `"id":"rs_1"`) || !strings.Contains(got, `"encrypted_content":"ENC"`) {
		t.Fatalf("reasoning item not replayed: %s", got)
	}
	if strings.Contains(got, "rs_no_enc") {
		t.Fatalf("item without encrypted content was replayed: %s", got)
	}
	if !strings.Contains(got, `"summary":[]`) {
		t.Fatalf("reasoning item must carry a (possibly empty) summary: %s", got)
	}
	if strings.Index(got, "rs_1") > strings.Index(got, "call_1") {
		t.Fatalf("reasoning must precede the function call it led to: %s", got)
	}

	_, _, other := TranslateMessagesForModel([]protocoltypes.Message{assistantWithReasoning("m")}, "other")
	if other.ReasoningItems != 0 {
		t.Fatalf("replayed %d reasoning items to a different model", other.ReasoningItems)
	}
}

func TestParseResponse_SkipsReasoningWithoutEncryptedContent(t *testing.T) {
	body := `{"id":"r","status":"completed","output":[
		{"type":"reasoning","id":"rs_a","summary":[],"encrypted_content":"E1"},
		{"type":"reasoning","id":"rs_b","summary":[{"type":"summary_text","text":"t"}]},
		{"type":"message","id":"m","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"done"}]}
	]}`
	resp, err := ParseResponseBody(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ParseResponseBody: %v", err)
	}
	if len(resp.ReasoningItems) != 1 || resp.ReasoningItems[0].ID != "rs_a" {
		t.Fatalf("reasoning items = %+v, want only rs_a", resp.ReasoningItems)
	}
	if resp.Phase != "final_answer" {
		t.Fatalf("phase = %q, want final_answer", resp.Phase)
	}
	if resp.ReasoningContent != "t" {
		t.Fatalf("reasoning summary text = %q, want t", resp.ReasoningContent)
	}
}
