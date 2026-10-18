package azure

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

// toolCallWithReasoningBody is a stateless (store=false) Responses API reply:
// an encrypted reasoning item, a commentary message, then a function call.
func toolCallWithReasoningBody() map[string]any {
	return map[string]any{
		"id":     "resp_1",
		"object": "response",
		"status": "completed",
		"output": []map[string]any{
			{
				"type":              "reasoning",
				"id":                "rs_1",
				"summary":           []map[string]any{{"type": "summary_text", "text": "Check the job first."}},
				"encrypted_content": "ENCRYPTED-1",
			},
			{
				"type":    "message",
				"id":      "msg_1",
				"role":    "assistant",
				"phase":   "commentary",
				"content": []map[string]any{{"type": "output_text", "text": "Checking the job."}},
			},
			{
				"type":      "function_call",
				"id":        "fc_1",
				"call_id":   "call_1",
				"name":      "get_job",
				"arguments": `{"job_id":"job_7"}`,
			},
		},
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
	}
}

type capturedRequests struct {
	mu     sync.Mutex
	bodies []map[string]any
}

func (c *capturedRequests) add(r *http.Request) {
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	c.mu.Lock()
	c.bodies = append(c.bodies, body)
	c.mu.Unlock()
}

func inputItems(body map[string]any) []map[string]any {
	raw, _ := body["input"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func TestChat_ParsesReasoningItemsAndPhase(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(toolCallWithReasoningBody())
	}))
	defer server.Close()

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	resp, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "status?"}}, nil, "gpt-6.1-sol",
		map[string]any{"thinking_level": "high"})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(resp.ReasoningItems) != 1 {
		t.Fatalf("reasoning items = %d, want 1", len(resp.ReasoningItems))
	}
	item := resp.ReasoningItems[0]
	if item.ID != "rs_1" || item.EncryptedContent != "ENCRYPTED-1" || item.Model != "gpt-6.1-sol" {
		t.Fatalf("reasoning item = %+v, want rs_1 / ENCRYPTED-1 stamped with the request model", item)
	}
	if len(item.Summary) != 1 || item.Summary[0] != "Check the job first." {
		t.Fatalf("summary = %v", item.Summary)
	}
	if resp.Phase != "commentary" {
		t.Fatalf("phase = %q, want commentary", resp.Phase)
	}
}

// The request asks for encrypted reasoning back (store stays false), and the
// next call of the turn sends that reasoning, and the message phase, back in
// front of the function call it led to.
func TestChat_ReplaysReasoningWithinTurn(t *testing.T) {
	var reqs capturedRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.add(r)
		writeValidResponse(w)
	}))
	defer server.Close()

	history := []Message{
		{Role: "user", Content: "status?"},
		{
			Role:    "assistant",
			Content: "Checking the job.",
			Phase:   "commentary",
			ToolCalls: []protocoltypes.ToolCall{{
				ID:        "call_1",
				Name:      "get_job",
				Arguments: map[string]any{"job_id": "job_7"},
			}},
			ReasoningItems: []protocoltypes.ReasoningItem{
				{
					ID:               "rs_1",
					EncryptedContent: "ENCRYPTED-1",
					Summary:          []string{"Check the job first."},
					Model:            "gpt-6.1-sol",
				},
			},
		},
		{Role: "tool", ToolCallID: "call_1", Content: `{"status":"done"}`},
	}

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	if _, err := p.Chat(
		t.Context(),
		history,
		nil,
		"gpt-6.1-sol",
		map[string]any{"thinking_level": "high"},
	); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	body := reqs.bodies[0]
	if body["store"] != false {
		t.Fatalf("store = %v, want false", body["store"])
	}
	include, _ := body["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %v, want [reasoning.encrypted_content]", body["include"])
	}

	items := inputItems(body)
	types := make([]string, 0, len(items))
	for _, item := range items {
		typ, _ := item["type"].(string)
		if typ == "" {
			typ = "message:" + item["role"].(string)
		}
		types = append(types, typ)
	}
	want := []string{"message:user", "reasoning", "message:assistant", "function_call", "function_call_output"}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("input item order = %v, want %v", types, want)
	}
	reasoning := items[1]
	if reasoning["id"] != "rs_1" || reasoning["encrypted_content"] != "ENCRYPTED-1" {
		t.Fatalf("replayed reasoning item = %v", reasoning)
	}
	if items[2]["phase"] != "commentary" {
		t.Fatalf("assistant message phase = %v, want commentary", items[2]["phase"])
	}
}

// Encrypted reasoning is only valid for the model that wrote it: after a
// fallback to another model it is not sent.
func TestChat_DoesNotReplayAnotherModelsReasoning(t *testing.T) {
	var reqs capturedRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.add(r)
		writeValidResponse(w)
	}))
	defer server.Close()

	history := []Message{
		{Role: "user", Content: "status?"},
		{
			Role:      "assistant",
			ToolCalls: []protocoltypes.ToolCall{{ID: "call_1", Name: "get_job", Arguments: map[string]any{}}},
			ReasoningItems: []protocoltypes.ReasoningItem{{
				ID: "rs_1", EncryptedContent: "ENCRYPTED-1", Model: "gpt-6.1-sol",
			}},
		},
		{Role: "tool", ToolCallID: "call_1", Content: "done"},
	}

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	if _, err := p.Chat(t.Context(), history, nil, "gpt-5.5", map[string]any{"thinking_level": "high"}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	for _, item := range inputItems(reqs.bodies[0]) {
		if item["type"] == "reasoning" {
			t.Fatalf("gpt-6.1-sol reasoning was replayed to gpt-5.5: %v", item)
		}
	}
}

// If the API rejects the replayed reasoning, the call is retried once with
// plain messages, so a replay problem never fails the turn.
func TestChat_RetriesWithoutReplayWhenAPIRejectsIt(t *testing.T) {
	var reqs capturedRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.add(r)
		reqs.mu.Lock()
		n := len(reqs.bodies)
		reqs.mu.Unlock()
		if n == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"The encrypted content for item rs_1 could not be verified.",` +
				`"type":"invalid_request_error","param":"input[1].encrypted_content","code":"invalid_encrypted_content"}}`))
			return
		}
		writeValidResponse(w)
	}))
	defer server.Close()

	history := []Message{
		{Role: "user", Content: "status?"},
		{
			Role:      "assistant",
			Content:   "Checking.",
			Phase:     "commentary",
			ToolCalls: []protocoltypes.ToolCall{{ID: "call_1", Name: "get_job", Arguments: map[string]any{}}},
			ReasoningItems: []protocoltypes.ReasoningItem{{
				ID: "rs_1", EncryptedContent: "STALE", Model: "gpt-6.1-sol",
			}},
		},
		{Role: "tool", ToolCallID: "call_1", Content: "done"},
	}

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	resp, err := p.Chat(t.Context(), history, nil, "gpt-6.1-sol", map[string]any{"thinking_level": "high"})
	if err != nil {
		t.Fatalf("Chat() error = %v, want the retry to succeed", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("content = %q, want ok", resp.Content)
	}
	if len(reqs.bodies) != 2 {
		t.Fatalf("requests = %d, want 2 (rejected, then plain retry)", len(reqs.bodies))
	}
	for _, item := range inputItems(reqs.bodies[1]) {
		if item["type"] == "reasoning" || item["phase"] != nil {
			t.Fatalf("retry still replays reasoning or phase: %v", item)
		}
	}
}

// An unrelated 400 is returned as an error without a second request.
func TestChat_UnrelatedBadRequestIsNotRetried(t *testing.T) {
	var reqs capturedRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.add(r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(
			[]byte(`{"error":{"message":"Invalid value for max_output_tokens","type":"invalid_request_error"}}`),
		)
	}))
	defer server.Close()

	history := []Message{
		{Role: "user", Content: "status?"},
		{
			Role:      "assistant",
			ToolCalls: []protocoltypes.ToolCall{{ID: "call_1", Name: "get_job", Arguments: map[string]any{}}},
			ReasoningItems: []protocoltypes.ReasoningItem{{
				ID: "rs_1", EncryptedContent: "ENC", Model: "gpt-6.1-sol",
			}},
		},
		{Role: "tool", ToolCallID: "call_1", Content: "done"},
	}

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	_, err := p.Chat(t.Context(), history, nil, "gpt-6.1-sol", map[string]any{"thinking_level": "high"})
	if err == nil || !strings.Contains(err.Error(), "max_output_tokens") {
		t.Fatalf("Chat() error = %v, want the API's 400", err)
	}
	if len(reqs.bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(reqs.bodies))
	}
}

// Without reasoning (thinking off or unset) nothing extra is requested.
func TestChat_NoIncludeWithoutReasoning(t *testing.T) {
	var reqs capturedRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.add(r)
		writeValidResponse(w)
	}))
	defer server.Close()

	p := NewProvider("k", server.URL, "", "", WithResponsesAPIPath("v1/responses"))
	for _, opts := range []map[string]any{nil, {"thinking_level": "off"}} {
		if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4.1", opts); err != nil {
			t.Fatalf("Chat() error = %v", err)
		}
	}
	for i, body := range reqs.bodies {
		if _, ok := body["include"]; ok {
			t.Fatalf("request %d asked for include %v without reasoning", i, body["include"])
		}
	}
}
