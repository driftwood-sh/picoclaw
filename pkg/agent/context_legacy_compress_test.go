package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/providers"
)

// legacyDropNotes reproduces what older builds left in a session summary:
// one "[Emergency compression dropped …]" line per forced compression.
func legacyDropNotes(n int) string {
	notes := make([]string, 0, n)
	for i := 0; i < n; i++ {
		notes = append(notes, fmt.Sprintf(
			"[Emergency compression dropped %d oldest messages due to context limit]", 10+i%7,
		))
	}
	return strings.Join(notes, "\n\n")
}

func eightTurnHistory() []providers.Message {
	var h []providers.Message
	for i := 1; i <= 8; i++ {
		h = append(h,
			providers.Message{Role: "user", Content: fmt.Sprintf("Question %d", i)},
			providers.Message{Role: "assistant", Content: fmt.Sprintf("Answer %d", i)},
		)
	}
	return h
}

func forceCompress(t *testing.T, lcm *legacyContextManager, sessionKey string) {
	t.Helper()
	if err := lcm.Compact(context.Background(), &CompactRequest{
		SessionKey: sessionKey,
		Reason:     ContextCompressReasonRetry,
	}); err != nil {
		t.Fatalf("Compact: %v", err)
	}
}

// The autosana summary was 1,766 stacked drop notes and nothing else. A
// summary that is only notes must read back empty, and notes mixed into real
// text must be removed without touching the text.
func TestCleanSessionSummary_StripsLegacyDropNotes(t *testing.T) {
	if got := cleanSessionSummary(legacyDropNotes(1766)); got != "" {
		t.Fatalf("notes-only summary cleaned to %d bytes, want empty: %.120q", len(got), got)
	}

	mixed := "Pending approval apr_42 for send_email(to=a@b.co).\n\n" + legacyDropNotes(3) +
		"\n\nAayush said: \"hold the Faire demo\"."
	got := cleanSessionSummary(mixed)
	if strings.Contains(got, "Emergency compression") {
		t.Fatalf("drop note survived cleaning: %q", got)
	}
	for _, keep := range []string{"apr_42", `"hold the Faire demo"`} {
		if !strings.Contains(got, keep) {
			t.Fatalf("cleaning lost %q: %q", keep, got)
		}
	}
	if strings.Contains(got, "\n\n\n") {
		t.Fatalf("cleaning left a run of blank lines: %q", got)
	}
}

func TestCapSessionSummary_KeepsNewestAndNeverStacksMarker(t *testing.T) {
	var lines []string
	for i := 0; i < 2000; i++ {
		lines = append(lines, fmt.Sprintf("line %04d", i))
	}
	long := strings.Join(lines, "\n")

	got := capSessionSummary(long, 1024)
	if len(got) > 1024 {
		t.Fatalf("capped summary is %d bytes, want <= 1024", len(got))
	}
	if !strings.HasPrefix(got, summaryTrimmedMarker+"\n") {
		t.Fatalf("capped summary does not open with the marker: %.80q", got)
	}
	if !strings.HasSuffix(got, "line 1999") {
		t.Fatalf("capped summary lost the newest text: ...%q", got[len(got)-20:])
	}
	if strings.Count(got, summaryTrimmedMarker) != 1 {
		t.Fatalf("marker count = %d, want 1", strings.Count(got, summaryTrimmedMarker))
	}

	again := capSessionSummary(got+"\n"+long, 1024)
	if strings.Count(again, summaryTrimmedMarker) != 1 {
		t.Fatalf("re-capping stacked the marker: %d copies", strings.Count(again, summaryTrimmedMarker))
	}

	short := "short summary"
	if got := capSessionSummary(short, 1024); got != short {
		t.Fatalf("summary under the cap changed: %q", got)
	}
}

// Assemble is what the turn reads. A stored summary of drop notes must not
// reach the prompt, even before anything rewrites the stored copy.
func TestLegacyAssemble_CleansStoredSummary(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	agent.Sessions.SetHistory("s1", sixMessageHistory())
	agent.Sessions.SetSummary("s1", "Real context line.\n\n"+legacyDropNotes(50))

	lcm := &legacyContextManager{al: al}
	resp, err := lcm.Assemble(context.Background(), &AssembleRequest{SessionKey: "s1"})
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if resp.Summary != "Real context line." {
		t.Fatalf("assembled summary = %q, want only the real line", resp.Summary)
	}
}

// Emergency compression folds the dropped messages into the summary with one
// model call, giving the model the existing summary, the dropped text, tool
// calls and (truncated) tool results, and the keep-list.
func TestForceCompression_SummarizesDroppedMessages(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	lcm := &legacyContextManager{al: al}

	bigResult := strings.Repeat("x", 5000)
	history := append([]providers.Message{
		{Role: "user", Content: "Queue the Faire demo, then tell me the job id."},
		{Role: "assistant", ToolCalls: []providers.ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: &providers.FunctionCall{Name: "run_code", Arguments: `{"script":"render.py"}`},
		}}},
		{Role: "tool", ToolCallID: "call_1", Content: "job_id=job_777 queued " + bigResult},
		{Role: "assistant", Content: "Queued as job_777."},
	}, eightTurnHistory()...)
	agent.Sessions.SetHistory("s1", history)
	agent.Sessions.SetSummary("s1", "Earlier: Aayush approved the Carta copy.\n\n"+legacyDropNotes(5))

	forceCompress(t, lcm, "s1")

	if len(provider.calls) != 1 {
		t.Fatalf("summary calls = %d, want 1", len(provider.calls))
	}
	prompt := provider.calls[0].messages[0].Content
	for _, want := range []string{
		"pending approval ids with the exact arguments",
		"quoted verbatim",
		"Earlier: Aayush approved the Carta copy.",
		"user: Queue the Faire demo, then tell me the job id.",
		`assistant called run_code({"script":"render.py"})`,
		"tool result: job_id=job_777 queued",
		"assistant: Queued as job_777.",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("summary prompt is missing %q", want)
		}
	}
	if strings.Contains(prompt, "Emergency compression") {
		t.Error("summary prompt still carries the legacy drop notes")
	}
	if strings.Contains(prompt, bigResult) {
		t.Error("summary prompt carries the full 5,000-char tool result; want it truncated")
	}
	if got := agent.Sessions.GetSummary("s1"); got != "summary 1" {
		t.Fatalf("stored summary = %q, want the model's summary", got)
	}
	if got := len(agent.Sessions.GetHistory("s1")); got >= len(history) {
		t.Fatalf("history not compressed: %d messages (was %d)", got, len(history))
	}
}

// A failed summary call leaves the previous summary (cleaned) in place and
// adds nothing to it: a real summary or nothing.
func TestForceCompression_FailedSummaryKeepsPreviousSummary(t *testing.T) {
	provider := &summarizerRecordingProvider{err: errors.New("upstream 500")}
	al, agent := newSummarizerTestLoop(t, provider)
	lcm := &legacyContextManager{al: al}

	agent.Sessions.SetHistory("s1", eightTurnHistory())
	agent.Sessions.SetSummary("s1", "Keep: job_9 is rendering.\n\n"+legacyDropNotes(3))

	forceCompress(t, lcm, "s1")

	if got := agent.Sessions.GetSummary("s1"); got != "Keep: job_9 is rendering." {
		t.Fatalf("summary after a failed summary call = %q, want the previous real text only", got)
	}
	if got := len(agent.Sessions.GetHistory("s1")); got >= len(eightTurnHistory()) {
		t.Fatalf("history not compressed after a failed summary: %d messages", got)
	}
}

// The loop that grew autosana's summary: compression after compression must
// never make the summary grow past the cap, whatever the model returns.
func TestForceCompression_RepeatedCompressionStaysBounded(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	lcm := &legacyContextManager{al: al}

	for i := 0; i < 60; i++ {
		agent.Sessions.SetHistory("s1", eightTurnHistory())
		forceCompress(t, lcm, "s1")
		if got := len(agent.Sessions.GetSummary("s1")); got > maxSessionSummaryBytes {
			t.Fatalf("after %d compressions the summary is %d bytes, cap %d", i+1, got, maxSessionSummaryBytes)
		}
	}
	if s := agent.Sessions.GetSummary("s1"); strings.Contains(s, "Emergency compression") {
		t.Fatalf("drop note written after repeated compressions: %q", s)
	}

	// Even a model that returns far more than asked is capped.
	verbose := &verboseSummaryProvider{size: 50000}
	al2, agent2 := newSummarizerTestLoop(t, verbose)
	lcm2 := &legacyContextManager{al: al2}
	agent2.Sessions.SetHistory("s2", eightTurnHistory())
	forceCompress(t, lcm2, "s2")
	if got := len(agent2.Sessions.GetSummary("s2")); got > maxSessionSummaryBytes {
		t.Fatalf("verbose summary stored at %d bytes, cap %d", got, maxSessionSummaryBytes)
	}
}

// Messages with nothing to summarize (tool plumbing only) make no model call.
func TestForceCompression_NothingToSummarizeMakesNoCall(t *testing.T) {
	provider := &summarizerRecordingProvider{}
	al, agent := newSummarizerTestLoop(t, provider)
	lcm := &legacyContextManager{al: al}

	history := []providers.Message{
		{Role: "user", Content: ""},
		{Role: "assistant"},
		{Role: "user", Content: ""},
		{Role: "assistant"},
		{Role: "user", Content: "latest"},
	}
	agent.Sessions.SetHistory("s1", history)
	agent.Sessions.SetSummary("s1", "previous")

	forceCompress(t, lcm, "s1")

	if len(provider.calls) != 0 {
		t.Fatalf("summary calls = %d, want 0 for an empty transcript", len(provider.calls))
	}
	if got := agent.Sessions.GetSummary("s1"); got != "previous" {
		t.Fatalf("summary = %q, want the previous summary unchanged", got)
	}
}

func TestDroppedTranscript_KeepsNewestWhenTooLong(t *testing.T) {
	var msgs []providers.Message
	for i := 0; i < 200; i++ {
		msgs = append(msgs, providers.Message{
			Role:    "user",
			Content: fmt.Sprintf("message %03d %s", i, strings.Repeat("y", 1500)),
		})
	}
	got := droppedTranscript(msgs)
	if len(got) > droppedTranscriptMaxChars+len("[earlier messages omitted]\n") {
		t.Fatalf("transcript is %d chars, want <= %d", len(got), droppedTranscriptMaxChars)
	}
	if !strings.HasPrefix(got, "[earlier messages omitted]\n") {
		t.Fatalf("long transcript does not say it omitted messages: %.60q", got)
	}
	if !strings.Contains(got, "message 199") || strings.Contains(got, "message 000") {
		t.Fatal("transcript did not keep the newest messages")
	}
}

// verboseSummaryProvider returns a summary far over the cap.
type verboseSummaryProvider struct {
	size int
}

func (p *verboseSummaryProvider) Chat(
	context.Context, []providers.Message, []providers.ToolDefinition, string, map[string]any,
) (*providers.LLMResponse, error) {
	return &providers.LLMResponse{Content: strings.Repeat("word ", p.size/5)}, nil
}

func (p *verboseSummaryProvider) GetDefaultModel() string { return "unused" }
func (p *verboseSummaryProvider) SupportsThinking() bool  { return true }
