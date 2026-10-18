package tools

import (
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/config"
)

// The stock description must not prescribe background=true for long work:
// that line sent agents into sleep-and-poll loops inside a turn, against a
// deployment rule to hand long work off and end the turn.
func TestExecToolDescription_GivesNoLongWorkRule(t *testing.T) {
	tool, err := NewExecToolWithConfig(t.TempDir(), false, config.DefaultConfig())
	if err != nil {
		t.Fatalf("NewExecToolWithConfig: %v", err)
	}
	desc := tool.Description()
	for _, banned := range []string{"long-running", "Use background=true for"} {
		if strings.Contains(desc, banned) {
			t.Errorf("exec description still says %q: %s", banned, desc)
		}
	}
	// It still documents what the parameters do.
	for _, want := range []string{"background=true", "sessionId", "pty=true", "Output buffer limit"} {
		if !strings.Contains(desc, want) {
			t.Errorf("exec description lost %q: %s", want, desc)
		}
	}
}

func TestExecToolDescription_AppendsConfiguredHint(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Tools.Exec.LongRunningHint = "  Work over a minute: start it with driftwood-batch and end the turn.  "
	tool, err := NewExecToolWithConfig(t.TempDir(), false, cfg)
	if err != nil {
		t.Fatalf("NewExecToolWithConfig: %v", err)
	}
	desc := tool.Description()
	if !strings.HasSuffix(desc, "\n\nWork over a minute: start it with driftwood-batch and end the turn.") {
		t.Fatalf("configured hint not appended (trimmed) to the description: %q", desc)
	}
	if !strings.HasPrefix(desc, execToolDescription) {
		t.Fatalf("description lost its stock text: %q", desc)
	}
}

func TestExecToolDescription_NilConfigHasNoHint(t *testing.T) {
	tool, err := NewExecTool(t.TempDir(), false)
	if err != nil {
		t.Fatalf("NewExecTool: %v", err)
	}
	if got := tool.Description(); got != execToolDescription {
		t.Fatalf("description = %q, want the stock text", got)
	}
}
