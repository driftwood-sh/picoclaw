package agent

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// harnessLinesThatFightAPersona are the runtime-prompt lines a persona had to
// cancel: narration, a second identity, memory writes on a whim, a deprecated
// notes path, and ALWAYS/MUST wording for a non-invariant.
var harnessLinesThatFightAPersona = []string{
	"You are picoclaw, a helpful AI assistant",
	"briefly explain",
	"Briefly explain",
	"seems memorable",
	"Daily Notes",
	"ALWAYS use tools",
	"you MUST call",
	"Always defer to explicit user instructions",
}

func writePersona(t *testing.T, workspace, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(workspace, "AGENT.md"), []byte(body), 0o644); err != nil {
		t.Fatalf("write AGENT.md: %v", err)
	}
}

func TestHarnessPrompt_WithPersonaCarriesNoCompetingLines(t *testing.T) {
	workspace := t.TempDir()
	writePersona(t, workspace, "You are this customer's dedicated Driftwood agent.\n")
	cb := NewContextBuilder(workspace)

	prompt := cb.BuildSystemPrompt()

	for _, line := range harnessLinesThatFightAPersona {
		if strings.Contains(prompt, line) {
			t.Errorf("system prompt still carries %q", line)
		}
	}
	for _, want := range []string{
		"# Agent runtime (picoclaw ",
		"**Act through tools** - To do something, call the tool that does it; never say you did what no tool confirmed.",
		"**Be accurate** - Report only what a tool confirmed.",
		"If it disagrees with the current messages or your instructions, trust those.",
		"You are this customer's dedicated Driftwood agent.",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("system prompt is missing %q", want)
		}
	}
	// The persona's identity is the only one.
	if n := strings.Count(prompt, "You are "); n != 1 {
		t.Errorf("system prompt has %d identity sentences, want only the persona's", n)
	}
}

func TestHarnessPrompt_WithoutPersonaKeepsStockIdentity(t *testing.T) {
	cb := NewContextBuilder(t.TempDir())
	prompt := cb.BuildSystemPrompt()

	if !strings.Contains(prompt, "You are picoclaw, a helpful AI assistant.") {
		t.Errorf("stock identity missing without a persona:\n%s", prompt)
	}
	for _, line := range []string{"briefly explain", "seems memorable", "Daily Notes", "ALWAYS use tools"} {
		if strings.Contains(prompt, line) {
			t.Errorf("system prompt still carries %q", line)
		}
	}
}

func TestHarnessPrompt_EmptyPersonaKeepsStockIdentity(t *testing.T) {
	workspace := t.TempDir()
	writePersona(t, workspace, "   \n")
	cb := NewContextBuilder(workspace)

	if !strings.Contains(cb.BuildSystemPrompt(), "You are picoclaw") {
		t.Error("an empty AGENT.md must not remove the only identity")
	}
}

func TestSummaryPreamble_TrustsCurrentMessagesAndInstructions(t *testing.T) {
	cb := NewContextBuilder(t.TempDir())
	parts := cb.turnContextParts("Earlier: job_7 queued.")
	joined := strings.Join(parts, "\n")
	if !strings.Contains(joined, "CONTEXT_SUMMARY: ") {
		t.Fatalf("summary block lost its CONTEXT_SUMMARY prefix: %q", joined)
	}
	if !strings.Contains(joined, "trust those") || strings.Contains(joined, "defer to explicit") {
		t.Fatalf("summary preamble = %q, want the 'trust those' wording", joined)
	}
}

func TestDynamicContext_ClockCarriesZone(t *testing.T) {
	cb := NewContextBuilder(t.TempDir())
	ctx := cb.buildDynamicContext("pico_client", "C1", "", "")
	zoned := regexp.MustCompile(`## Current Time\n\d{4}-\d{2}-\d{2} \d{2}:\d{2} \S+ \(\w+day\)`)
	if !zoned.MatchString(ctx) {
		t.Fatalf("runtime clock has no zone:\n%s", ctx)
	}
}

func TestHookDeniedContent_SaysNotRunNotDenied(t *testing.T) {
	got := hookDeniedToolContent(approvalHookDeniedPrefix, "parked: waiting for a decision")
	if got != "Tool not run (approval hook): parked: waiting for a decision" {
		t.Fatalf("approval hook content = %q", got)
	}
	if strings.Contains(strings.ToLower(got), "denied") {
		t.Fatalf("parked call reads as a denial: %q", got)
	}
}
