package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// fakeResponsesServer is a local stand-in for the OpenAI Responses API
// (POST /v1/responses). Each model answers with either a refusal content part
// or a plain output_text part, so the real openai-responses provider and the
// real Responses parser run end to end without network access.
type fakeResponsesServer struct {
	mu        sync.Mutex
	models    []string
	refusing  map[string]bool
	answerFor map[string]string
	srv       *httptest.Server
}

func newFakeResponsesServer(t *testing.T, refusing map[string]bool, answerFor map[string]string) *fakeResponsesServer {
	t.Helper()
	f := &fakeResponsesServer{refusing: refusing, answerFor: answerFor}
	f.srv = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeResponsesServer) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != "/v1/responses" {
		http.Error(w, "unexpected route "+r.Method+" "+r.URL.Path, http.StatusNotFound)
		return
	}
	raw, _ := io.ReadAll(r.Body)
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.models = append(f.models, req.Model)
	f.mu.Unlock()

	var part string
	if f.refusing[req.Model] {
		part = `{"type":"refusal","refusal":"I'm sorry, but I can't help with that."}`
	} else {
		part = fmt.Sprintf(`{"type":"output_text","text":%q,"annotations":[]}`, f.answerFor[req.Model])
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{
		"id": "resp_fake",
		"object": "response",
		"status": "completed",
		"model": %q,
		"output": [
			{"type": "message", "id": "msg_fake", "role": "assistant", "status": "completed", "content": [%s]}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`, req.Model, part)
}

func (f *fakeResponsesServer) requestedModels() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.models...)
}

// injectedProviderMustNotRun fails the call so a test notices when the agent
// uses the injected default provider instead of the model_list providers.
type injectedProviderMustNotRun struct{}

func (injectedProviderMustNotRun) Chat(
	context.Context, []providers.Message, []providers.ToolDefinition, string, map[string]any,
) (*providers.LLMResponse, error) {
	return nil, fmt.Errorf("injected provider called; want the openai-responses model_list provider")
}

func (injectedProviderMustNotRun) GetDefaultModel() string { return "unused" }

// newResponsesRefusalFailoverConfig points two openai-responses model_list
// entries at the fake server: "test-model" (gpt-primary) as the agent primary
// and "failover-model" (gpt-failover) as the refusal failover model.
func newResponsesRefusalFailoverConfig(t *testing.T, apiBase string, withFailover bool) *config.Config {
	t.Helper()
	cfg := newConfiguredStreamingTestConfig(t, false, false, nil)
	primary := &config.ModelConfig{
		ModelName:     "test-model",
		Provider:      "openai-responses",
		Model:         "gpt-primary",
		APIBase:       apiBase,
		ThinkingLevel: "high",
	}
	primary.SetAPIKey("test-key")
	failover := &config.ModelConfig{
		ModelName:     "failover-model",
		Provider:      "openai-responses",
		Model:         "gpt-failover",
		APIBase:       apiBase,
		ThinkingLevel: "high",
	}
	failover.SetAPIKey("test-key")
	cfg.ModelList = []*config.ModelConfig{primary, failover}
	if withFailover {
		cfg.Agents.Defaults.RefusalFailover = &config.RefusalFailoverConfig{
			Model:       "failover-model",
			HoldMinutes: 360,
		}
	}
	return cfg
}

func TestResponsesAPIRefusalFailsOverToConfiguredModelAndArmsHold(t *testing.T) {
	fake := newFakeResponsesServer(t,
		map[string]bool{"gpt-primary": true},
		map[string]string{"gpt-failover": "answer from the failover model"},
	)
	cfg := newResponsesRefusalFailoverConfig(t, fake.srv.URL, true)
	al := NewAgentLoop(cfg, bus.NewMessageBus(), injectedProviderMustNotRun{})
	agent := al.GetRegistry().GetDefaultAgent()
	if len(agent.RefusalFailoverCandidates) == 0 {
		t.Fatal("refusal failover model did not resolve from model_list")
	}

	before := time.Now()
	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != "answer from the failover model" {
		t.Fatalf("response = %q, want the failover model's answer", got)
	}
	if models := fake.requestedModels(); len(models) != 2 ||
		models[0] != "gpt-primary" || models[1] != "gpt-failover" {
		t.Fatalf("requested models = %v, want [gpt-primary gpt-failover]", models)
	}
	holdUntil, active := agent.RefusalHoldUntil()
	if !active {
		t.Fatal("refusal hold should be armed after the Responses API primary refused")
	}
	if earliest, latest := before.Add(359*time.Minute), time.Now().Add(361*time.Minute); holdUntil.Before(earliest) ||
		holdUntil.After(latest) {
		t.Fatalf("hold until = %v, want about 360 minutes from now", holdUntil)
	}

	// The armed hold makes the next turn start on the failover model and skip
	// the refusing primary entirely.
	got = runConfiguredStreamingTurn(t, al, "pico")
	if got != "answer from the failover model" {
		t.Fatalf("second turn response = %q, want the failover model's answer", got)
	}
	if models := fake.requestedModels(); len(models) != 3 || models[2] != "gpt-failover" {
		t.Fatalf("requested models = %v, want the second turn on gpt-failover only", models)
	}
}

func TestResponsesAPIRefusalWithoutFailoverShowsRefusalText(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]bool{"gpt-primary": true}, nil)
	cfg := newResponsesRefusalFailoverConfig(t, fake.srv.URL, false)
	al := NewAgentLoop(cfg, bus.NewMessageBus(), injectedProviderMustNotRun{})

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != "I'm sorry, but I can't help with that." {
		t.Fatalf("response = %q, want the model's own refusal text", got)
	}
	if models := fake.requestedModels(); len(models) != 1 {
		t.Fatalf("requested models = %v, want one call (no failover configured)", models)
	}
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); active {
		t.Fatal("refusal hold should not arm without a configured failover")
	}
}

func TestResponsesAPIRefusalFromFailoverModelShowsTextWithoutLooping(t *testing.T) {
	fake := newFakeResponsesServer(t,
		map[string]bool{"gpt-primary": true, "gpt-failover": true},
		nil,
	)
	cfg := newResponsesRefusalFailoverConfig(t, fake.srv.URL, true)
	al := NewAgentLoop(cfg, bus.NewMessageBus(), injectedProviderMustNotRun{})

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != "I'm sorry, but I can't help with that." {
		t.Fatalf("response = %q, want the failover model's refusal text as is", got)
	}
	if models := fake.requestedModels(); len(models) != 2 ||
		models[0] != "gpt-primary" || models[1] != "gpt-failover" {
		t.Fatalf("requested models = %v, want exactly [gpt-primary gpt-failover]", models)
	}
}
