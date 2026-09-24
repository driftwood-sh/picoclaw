package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

const fakeRefusalText = "I'm sorry, but I can't help with that."

// fakeResponsesServer is a local stand-in for the OpenAI Responses API
// (POST /v1/responses). Each model answers per its mode: "refuse" returns a
// refusal content part, "500" returns an HTTP 500, anything else returns an
// output_text part "ok from <model>". The real openai-responses provider and
// the real Responses parser run end to end without network access.
type fakeResponsesServer struct {
	mu     sync.Mutex
	models []string
	mode   map[string]string
	srv    *httptest.Server
}

func newFakeResponsesServer(t *testing.T, mode map[string]string) *fakeResponsesServer {
	t.Helper()
	f := &fakeResponsesServer{mode: mode}
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
	mode := f.mode[req.Model]
	f.mu.Unlock()

	var part string
	switch mode {
	case "500":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":{"message":"server error","type":"server_error"}}`)
		return
	case "refuse":
		part = fmt.Sprintf(`{"type":"refusal","refusal":%q}`, fakeRefusalText)
	default:
		part = fmt.Sprintf(`{"type":"output_text","text":%q,"annotations":[]}`, "ok from "+req.Model)
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

// setModes replaces the per-model modes and clears the request log.
func (f *fakeResponsesServer) setModes(mode map[string]string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.mode = mode
	f.models = nil
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

func responsesModelConfig(name, model, apiBase string) *config.ModelConfig {
	mc := &config.ModelConfig{
		ModelName:     name,
		Provider:      "openai-responses",
		Model:         model,
		APIBase:       apiBase,
		ThinkingLevel: "high",
	}
	mc.SetAPIKey("test-key")
	return mc
}

// newProdShapeRefusalConfig mirrors the fleet config: primary gpt-6-sol with
// the given fallbacks, refusal_failover gpt-5-5 with a 360-minute hold, and
// every model an openai-responses entry pointed at the fake server.
func newProdShapeRefusalConfig(t *testing.T, apiBase string, fallbacks []string) *config.Config {
	t.Helper()
	cfg := newConfiguredStreamingTestConfig(t, false, false, nil)
	cfg.Agents.Defaults.ModelName = "gpt-6-sol"
	cfg.Agents.Defaults.ModelFallbacks = fallbacks
	cfg.Agents.Defaults.RefusalFailover = &config.RefusalFailoverConfig{Model: "gpt-5-5", HoldMinutes: 360}
	cfg.Agents.Defaults.LLMRetryBackoffSecs = 1
	cfg.ModelList = []*config.ModelConfig{
		responsesModelConfig("gpt-6-sol", "gpt-6-sol", apiBase),
		responsesModelConfig("gpt-5-5", "gpt-5.5", apiBase),
		responsesModelConfig("minimax-m3", "minimax-m3", apiBase),
	}
	return cfg
}

var (
	mainShapeFallbacks   = []string{"gpt-5-5"}
	workerShapeFallbacks = []string{"gpt-5-5", "minimax-m3"}
)

func newProdShapeLoop(t *testing.T, fake *fakeResponsesServer, fallbacks []string) *AgentLoop {
	t.Helper()
	al := NewAgentLoop(
		newProdShapeRefusalConfig(t, fake.srv.URL, fallbacks),
		bus.NewMessageBus(),
		injectedProviderMustNotRun{},
	)
	if len(al.GetRegistry().GetDefaultAgent().RefusalFailoverCandidates) == 0 {
		t.Fatal("refusal failover model did not resolve from model_list")
	}
	return al
}

func assertModels(t *testing.T, fake *fakeResponsesServer, want ...string) {
	t.Helper()
	if got := fake.requestedModels(); !reflect.DeepEqual(got, want) {
		t.Fatalf("requested models = %v, want %v", got, want)
	}
}

func assertHoldNear(t *testing.T, al *AgentLoop, earliest, latest time.Time) {
	t.Helper()
	holdUntil, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil()
	if !active {
		t.Fatal("refusal hold should be armed")
	}
	if holdUntil.Before(earliest) || holdUntil.After(latest) {
		t.Fatalf("hold until = %v, want between %v and %v", holdUntil, earliest, latest)
	}
}

func runTurnExpectingError(t *testing.T, al *AgentLoop) (string, error) {
	t.Helper()
	return al.runAgentLoop(
		context.Background(),
		al.GetRegistry().GetDefaultAgent(),
		configuredStreamingProcessOptions("pico"),
	)
}

// (a) gpt-6-sol refuses: the turn moves to gpt-5-5, the hold arms for 360
// minutes, and the next turn starts on gpt-5-5.
func TestProdShapeRefusalFailsOverToGPT55AndArmsHold(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)

	before := time.Now()
	got := runConfiguredStreamingTurn(t, al, "pico")
	if got != "ok from gpt-5.5" {
		t.Fatalf("response = %q, want the gpt-5.5 answer", got)
	}
	assertModels(t, fake, "gpt-6-sol", "gpt-5.5")
	assertHoldNear(t, al, before.Add(359*time.Minute), time.Now().Add(361*time.Minute))

	fake.setModes(map[string]string{"gpt-6-sol": "refuse"})
	got = runConfiguredStreamingTurn(t, al, "pico")
	if got != "ok from gpt-5.5" {
		t.Fatalf("second turn response = %q, want the gpt-5.5 answer", got)
	}
	assertModels(t, fake, "gpt-5.5")
}

// (b) During the hold gpt-5-5 fails with a 500: the turn falls back to the
// healthy primary instead of failing the turn.
func TestProdShapeHoldFallsBackToPrimaryWhenFailoverErrors(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)
	_ = runConfiguredStreamingTurn(t, al, "pico")
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); !active {
		t.Fatal("refusal hold should be armed after the first turn")
	}

	fake.setModes(map[string]string{"gpt-6-sol": "ok", "gpt-5.5": "500"})
	got, err := runTurnExpectingError(t, al)
	if err != nil {
		t.Fatalf("turn during hold failed: %v (requests %v)", err, fake.requestedModels())
	}
	if got != "ok from gpt-6-sol" {
		t.Fatalf("response = %q, want the gpt-6-sol answer", got)
	}
	assertModels(t, fake, "gpt-5.5", "gpt-6-sol")
}

// (c) gpt-6-sol fails with a 500 and the fallback gpt-5-5 refuses: gpt-6-sol
// never refused, so no hold arms, gpt-5-5 runs once, and its refusal text is
// the reply.
func TestProdShapeFallbackRefusalAfterPrimaryErrorArmsNoHold(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "500", "gpt-5.5": "refuse"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != fakeRefusalText {
		t.Fatalf("response = %q, want the gpt-5.5 refusal text as is", got)
	}
	assertModels(t, fake, "gpt-6-sol", "gpt-5.5")
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); active {
		t.Fatal("refusal hold armed, but the primary never refused")
	}
}

// (d) Worker shape [gpt-6-sol, gpt-5-5, minimax-m3]: during the hold the chain
// is [gpt-5-5, minimax-m3, gpt-6-sol]. When gpt-5-5 errors the turn reaches
// minimax-m3. When minimax-m3 errors too (gpt-5-5 now in cooldown), the turn
// reaches gpt-6-sol.
func TestProdShapeWorkerHoldKeepsWholeFallbackChain(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse"})
	al := newProdShapeLoop(t, fake, workerShapeFallbacks)
	_ = runConfiguredStreamingTurn(t, al, "pico")
	assertModels(t, fake, "gpt-6-sol", "gpt-5.5")

	fake.setModes(map[string]string{"gpt-6-sol": "ok", "gpt-5.5": "500"})
	got, err := runTurnExpectingError(t, al)
	if err != nil {
		t.Fatalf("turn during hold failed: %v (requests %v)", err, fake.requestedModels())
	}
	if got != "ok from minimax-m3" {
		t.Fatalf("response = %q, want the minimax-m3 answer", got)
	}
	assertModels(t, fake, "gpt-5.5", "minimax-m3")

	fake.setModes(map[string]string{"gpt-6-sol": "ok", "gpt-5.5": "500", "minimax-m3": "500"})
	got, err = runTurnExpectingError(t, al)
	if err != nil {
		t.Fatalf("turn during hold failed: %v (requests %v)", err, fake.requestedModels())
	}
	if got != "ok from gpt-6-sol" {
		t.Fatalf("response = %q, want the gpt-6-sol answer", got)
	}
	// gpt-5.5 is in cooldown after its 500 and is skipped without a request.
	assertModels(t, fake, "minimax-m3", "gpt-6-sol")
}

// After the switch, gpt-5-5 errors and the chain reaches gpt-6-sol at the
// tail, which refuses again. That refusal is the reply: no second switch.
func TestProdShapeTailRefusalAfterSwitchDoesNotSwitchAgain(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse", "gpt-5.5": "500"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)

	got, err := runTurnExpectingError(t, al)
	if err != nil {
		t.Fatalf("turn failed: %v (requests %v)", err, fake.requestedModels())
	}
	if got != fakeRefusalText {
		t.Fatalf("response = %q, want the gpt-6-sol refusal text as is", got)
	}
	assertModels(t, fake, "gpt-6-sol", "gpt-5.5", "gpt-6-sol")
}

// During an armed hold, gpt-5-5 errors and gpt-6-sol refuses at the tail. The
// refusal is the reply, the hold is not re-armed, and nothing switches again.
func TestProdShapeTailRefusalDuringHoldDoesNotRearm(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse", "gpt-5.5": "500"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)
	agent := al.GetRegistry().GetDefaultAgent()
	agent.ArmRefusalHold(time.Hour)
	armedUntil, _ := agent.RefusalHoldUntil()

	got, err := runTurnExpectingError(t, al)
	if err != nil {
		t.Fatalf("turn during hold failed: %v (requests %v)", err, fake.requestedModels())
	}
	if got != fakeRefusalText {
		t.Fatalf("response = %q, want the gpt-6-sol refusal text as is", got)
	}
	assertModels(t, fake, "gpt-5.5", "gpt-6-sol")
	if holdUntil, _ := agent.RefusalHoldUntil(); !holdUntil.Equal(armedUntil) {
		t.Fatalf("hold until = %v, want the original %v (no re-arm)", holdUntil, armedUntil)
	}
}

// With no refusal_failover configured, a Responses API refusal is the reply.
func TestResponsesAPIRefusalWithoutFailoverShowsRefusalText(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse"})
	cfg := newProdShapeRefusalConfig(t, fake.srv.URL, nil)
	cfg.Agents.Defaults.RefusalFailover = nil
	al := NewAgentLoop(cfg, bus.NewMessageBus(), injectedProviderMustNotRun{})

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != fakeRefusalText {
		t.Fatalf("response = %q, want the model's own refusal text", got)
	}
	assertModels(t, fake, "gpt-6-sol")
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); active {
		t.Fatal("refusal hold should not arm without a configured failover")
	}
}

// When gpt-5-5 refuses after the switch, its refusal text is the reply after
// exactly two calls.
func TestResponsesAPIRefusalFromFailoverModelShowsTextWithoutLooping(t *testing.T) {
	fake := newFakeResponsesServer(t, map[string]string{"gpt-6-sol": "refuse", "gpt-5.5": "refuse"})
	al := newProdShapeLoop(t, fake, mainShapeFallbacks)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != fakeRefusalText {
		t.Fatalf("response = %q, want the gpt-5.5 refusal text as is", got)
	}
	assertModels(t, fake, "gpt-6-sol", "gpt-5.5")
}

func TestRefusalFailoverChainOrder(t *testing.T) {
	c := func(model string) providers.FallbackCandidate {
		return providers.FallbackCandidate{Provider: "openai-responses", Model: model}
	}
	names := func(chain []providers.FallbackCandidate) string {
		out := make([]string, 0, len(chain))
		for _, candidate := range chain {
			out = append(out, candidate.Model)
		}
		return strings.Join(out, ",")
	}
	tests := []struct {
		name       string
		candidates []providers.FallbackCandidate
		want       string
	}{
		{"main shape", []providers.FallbackCandidate{c("sol"), c("g55")}, "g55,sol"},
		{"worker shape", []providers.FallbackCandidate{c("sol"), c("g55"), c("mm3")}, "g55,mm3,sol"},
		{"failover not in chain", []providers.FallbackCandidate{c("sol"), c("mm3")}, "g55,mm3,sol"},
		{"primary is the failover", []providers.FallbackCandidate{c("g55"), c("mm3")}, "g55,mm3"},
		{"primary only", []providers.FallbackCandidate{c("sol")}, "g55,sol"},
		{"no candidates", nil, "g55"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := names(refusalFailoverChain(c("g55"), tt.candidates)); got != tt.want {
				t.Fatalf("chain = %s, want %s", got, tt.want)
			}
		})
	}
}
