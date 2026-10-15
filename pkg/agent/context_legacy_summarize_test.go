package agent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/tools"
)

type summarizerCall struct {
	model    string
	opts     map[string]any
	messages []providers.Message
}

// summarizerRecordingProvider records every call; it reports thinking support
// like the OpenAI Responses provider does.
type summarizerRecordingProvider struct {
	mu     sync.Mutex
	calls  []summarizerCall
	err    error
	onCall func()
}

func (p *summarizerRecordingProvider) Chat(
	_ context.Context,
	messages []providers.Message,
	_ []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	p.mu.Lock()
	p.calls = append(p.calls, summarizerCall{
		model:    model,
		opts:     maps.Clone(opts),
		messages: append([]providers.Message(nil), messages...),
	})
	n := len(p.calls)
	p.mu.Unlock()
	if p.onCall != nil {
		p.onCall()
	}
	if p.err != nil {
		return nil, p.err
	}
	return &providers.LLMResponse{Content: fmt.Sprintf("summary %d", n)}, nil
}

func (p *summarizerRecordingProvider) GetDefaultModel() string { return "unused" }
func (p *summarizerRecordingProvider) SupportsThinking() bool  { return true }

// newSummarizerTestLoop mirrors the fleet config shape: the agent names a
// config alias whose API model id differs ("gpt-6-1-sol" -> "gpt-6.1-sol")
// and whose thinking level is high.
func newSummarizerTestLoop(t *testing.T, provider providers.LLMProvider) (*AgentLoop, *AgentInstance) {
	t.Helper()
	t.Setenv("PICOCLAW_BUILTIN_SKILLS", t.TempDir())
	temp := 1.0
	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:                 t.TempDir(),
				ModelName:                 "gpt-6-1-sol",
				MaxTokens:                 4096,
				MaxToolIterations:         10,
				ContextWindow:             16000,
				SummarizeMessageThreshold: 2,
				SummarizeTokenPercent:     75,
				Temperature:               &temp,
			},
		},
		ModelList: []*config.ModelConfig{{
			ModelName:     "gpt-6-1-sol",
			Provider:      "openai-responses",
			Model:         "gpt-6.1-sol",
			ThinkingLevel: "high",
		}},
	}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	agent := al.GetRegistry().GetDefaultAgent()
	if agent == nil {
		t.Fatal("expected default agent")
	}
	return al, agent
}

func sixMessageHistory() []providers.Message {
	return []providers.Message{
		{Role: "user", Content: "Question one"},
		{Role: "assistant", Content: "Answer one"},
		{Role: "user", Content: "Question two"},
		{Role: "assistant", Content: "Answer two"},
		{Role: "user", Content: "Question three"},
		{Role: "assistant", Content: "Answer three"},
	}
}

// TestSummarizer_UsesAPIModelIDAndThinkingWithoutTemperature reproduces the
// audit finding: the summarizer sent the config alias ("gpt-6-1-sol") and
// temperature 0.3, which OpenAI rejects for a reasoning model, so it could
// never succeed. It must send the API model id the main call path sends, low
// reasoning effort (summaries run every turn), and no temperature.
func TestSummarizer_UsesAPIModelIDAndThinkingWithoutTemperature(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	agent.Sessions.SetHistory("s1", sixMessageHistory())

	lcm := &legacyContextManager{al: al}
	lcm.summarizeSession(agent, "s1")

	if len(provider.calls) == 0 {
		t.Fatal("summarizer made no model call")
	}
	for i, call := range provider.calls {
		if call.model != "gpt-6.1-sol" {
			t.Errorf("call %d model = %q, want the API id %q", i, call.model, "gpt-6.1-sol")
		}
		if _, ok := call.opts["temperature"]; ok {
			t.Errorf("call %d sent temperature %v; reasoning models reject it", i, call.opts["temperature"])
		}
		// The model is configured "high"; summaries are capped at low effort.
		if got := call.opts["thinking_level"]; got != "low" {
			t.Errorf("call %d thinking_level = %#v, want %q", i, got, "low")
		}
	}
	if got := agent.Sessions.GetSummary("s1"); got != "summary 1" {
		t.Fatalf("stored summary = %q, want the model's summary", got)
	}
}

// TestSummarizer_LargeBatchKeepsPreviousSummary covers the >10 message path,
// which used to summarize both halves without the existing summary and then
// replace it, dropping every earlier turn's context.
func TestSummarizer_LargeBatchKeepsPreviousSummary(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	var history []providers.Message
	for i := 0; i < 8; i++ {
		history = append(history,
			providers.Message{Role: "user", Content: fmt.Sprintf("question %d", i)},
			providers.Message{Role: "assistant", Content: fmt.Sprintf("answer %d", i)},
		)
	}
	agent.Sessions.SetHistory("s1", history)
	agent.Sessions.SetSummary("s1", "PRIOR-SUMMARY-MARKER")

	lcm := &legacyContextManager{al: al}
	lcm.summarizeSession(agent, "s1")

	carried := false
	for _, call := range provider.calls {
		if strings.Contains(call.messages[0].Content, "PRIOR-SUMMARY-MARKER") {
			carried = true
		}
	}
	if !carried {
		t.Fatal("no summarization call carried the existing summary; earlier context would be lost")
	}
}

// TestSummarizer_FailureIsLoggedAndFallsBack: a failing summarizer still
// produces the fallback summary (unchanged behavior) — the change is that the
// failure is no longer silent. The warning path must not panic and the error
// must reach retryLLMCall's caller.
func TestSummarizer_FailureIsLoggedAndFallsBack(t *testing.T) {
	provider := &summarizerRecordingProvider{err: errors.New("400 temperature not supported")}
	al, agent := newSummarizerTestLoop(t, provider)
	agent.Sessions.SetHistory("s1", sixMessageHistory())

	lcm := &legacyContextManager{al: al}
	if _, err := lcm.retryLLMCall(context.Background(), agent, "prompt", 2); err == nil ||
		!strings.Contains(err.Error(), "temperature not supported") {
		t.Fatalf("retryLLMCall() error = %v, want the provider error", err)
	}
	lcm.summarizeSession(agent, "s1")
	if got := agent.Sessions.GetSummary("s1"); !strings.HasPrefix(got, "Conversation summary:") {
		t.Fatalf("summary = %q, want the truncated-transcript fallback", got)
	}
}

// TestSummarizer_KeepsMessagesAppendedDuringTheCall: a real summarizer call
// takes seconds, and the next turn may append messages meanwhile. Truncation
// must drop only the summarized prefix, not the newly appended messages.
func TestSummarizer_KeepsMessagesAppendedDuringTheCall(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	agent.Sessions.SetHistory("s1", sixMessageHistory())
	var once sync.Once
	provider.onCall = func() {
		once.Do(func() {
			agent.Sessions.AddMessage("s1", "user", "Question four")
			agent.Sessions.AddMessage("s1", "assistant", "Answer four")
		})
	}

	lcm := &legacyContextManager{al: al}
	lcm.summarizeSession(agent, "s1")

	history := agent.Sessions.GetHistory("s1")
	contents := make([]string, 0, len(history))
	for _, m := range history {
		contents = append(contents, m.Content)
	}
	got := strings.Join(contents, " | ")
	// Q1/A1 were summarized; Q2..A3 were kept back; Q4/A4 arrived meanwhile.
	// Truncating by the count computed before the call (4) would also have
	// dropped Q2/A2, which no summary covers.
	want := "Question two | Answer two | Question three | Answer three | Question four | Answer four"
	if got != want {
		t.Fatalf("history after summarization = %q, want %q", got, want)
	}
}

// TestSummarizer_DiscardsSummaryWhenSessionWasCompacted: if the session was
// rewritten (forced compression) while the summary was being written, the
// summary no longer describes the history's prefix and must not be applied.
func TestSummarizer_DiscardsSummaryWhenSessionWasCompacted(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	agent.Sessions.SetHistory("s1", sixMessageHistory())
	replaced := []providers.Message{
		{Role: "user", Content: "Question three"},
		{Role: "assistant", Content: "Answer three"},
	}
	var once sync.Once
	provider.onCall = func() {
		once.Do(func() { agent.Sessions.SetHistory("s1", replaced) })
	}

	lcm := &legacyContextManager{al: al}
	lcm.summarizeSession(agent, "s1")

	if got := agent.Sessions.GetSummary("s1"); got != "" {
		t.Fatalf("summary = %q, want none applied to a rewritten session", got)
	}
	if got := agent.Sessions.GetHistory("s1"); len(got) != len(replaced) {
		t.Fatalf("history len = %d, want the compacted %d untouched", len(got), len(replaced))
	}
}

func TestSummarizerThinking_CapsAtLowWithoutEnablingThinking(t *testing.T) {
	for _, tc := range []struct {
		in, want thinkingSettings
	}{
		{thinkingSettings{level: ThinkingHigh, configured: true}, thinkingSettings{level: ThinkingLow, configured: true}},
		{thinkingSettings{level: ThinkingXHigh, configured: true}, thinkingSettings{level: ThinkingLow, configured: true}},
		{thinkingSettings{level: ThinkingOff, configured: true}, thinkingSettings{level: ThinkingOff, configured: true}},
		{thinkingSettings{}, thinkingSettings{}},
	} {
		if got := summarizerThinking(tc.in); got != tc.want {
			t.Errorf("summarizerThinking(%+v) = %+v, want %+v", tc.in, got, tc.want)
		}
	}
}

// TestCronSessionsAreNotSummarized: every cron run uses a fresh
// agent:cron-<job>-<uuid> session that is never read again, so post-turn
// summarization there is a wasted model call. A normal explicit session with
// the same history is the control: it does get summarized.
func TestCronSessionsAreNotSummarized(t *testing.T) {
	for _, tc := range []struct {
		key           string
		wantSummarize bool
	}{
		{key: tools.CronSessionKeyPrefix + "job1-0b5e", wantSummarize: false},
		{key: "agent:main:control-session", wantSummarize: true},
	} {
		t.Run(tc.key, func(t *testing.T) {
			provider := &summarizerRecordingProvider{}
			al, agent := newSummarizerTestLoop(t, provider)
			agent.Sessions.SetHistory(tc.key, sixMessageHistory())

			_, err := al.ProcessDirectWithChannel(context.Background(), "run the job", tc.key, "pico", "chat-1")
			if err != nil {
				t.Fatalf("ProcessDirectWithChannel() error = %v", err)
			}

			deadline := time.Now().Add(2 * time.Second)
			for agent.Sessions.GetSummary(tc.key) == "" && time.Now().Before(deadline) {
				if !tc.wantSummarize {
					break
				}
				time.Sleep(20 * time.Millisecond)
			}
			if !tc.wantSummarize {
				time.Sleep(200 * time.Millisecond) // give a (wrongly) started summarizer time to land
			}
			summarized := agent.Sessions.GetSummary(tc.key) != ""
			if summarized != tc.wantSummarize {
				t.Fatalf("session %q summarized = %v, want %v", tc.key, summarized, tc.wantSummarize)
			}
		})
	}
}
