package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sipeed/picoclaw/pkg/config"
)

func writeMemoryFile(t *testing.T, workspace, content string) {
	t.Helper()
	path := filepath.Join(workspace, "memory", "MEMORY.md")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestMemoryContext_UnderBudgetIsInjectedWhole(t *testing.T) {
	ws := t.TempDir()
	content := strings.Repeat("- 2026-10-14: a fact\n", 10)
	writeMemoryFile(t, ws, content)

	ms := NewMemoryStore(ws)
	ms.SetMaxBytes(4096)
	got := ms.GetMemoryContext()
	if !strings.Contains(got, content) {
		t.Fatalf("memory under budget was altered:\n%s", got)
	}
	if strings.Contains(got, "over the") {
		t.Fatalf("unexpected over-budget notice:\n%s", got)
	}
}

// TestMemoryContext_OverBudgetIsCappedWithCurateNotice: autosana's MEMORY.md
// reached 918 KB and broke OpenAI's 1 MB instructions limit for five days.
// Over budget, only the first N bytes are injected, plus a notice telling the
// agent the real size, the budget, and to curate the file now.
func TestMemoryContext_OverBudgetIsCappedWithCurateNotice(t *testing.T) {
	ws := t.TempDir()
	var sb strings.Builder
	for i := 0; sb.Len() < 900*1024; i++ {
		sb.WriteString("- 2026-10-14: fact number ")
		sb.WriteString(strings.Repeat("x", i%40))
		sb.WriteString("\n")
	}
	writeMemoryFile(t, ws, sb.String())

	ms := NewMemoryStore(ws)
	ms.SetMaxBytes(32 * 1024)
	got := ms.GetMemoryContext()

	if len(got) > 33*1024 {
		t.Fatalf("memory context = %d bytes, want about the 32 KB budget", len(got))
	}
	wantSize := "is " + formatKB(sb.Len())
	for _, want := range []string{"MEMORY.md", wantSize, "over the 32 KB budget", "Curate it now"} {
		if !strings.Contains(got, want) {
			t.Errorf("notice missing %q:\n%s", want, got[len(got)-400:])
		}
	}
	if !strings.HasPrefix(got, "## Long-term Memory\n\n- 2026-10-14: fact number") {
		t.Fatalf("capped memory must keep the head of the file:\n%s", got[:200])
	}
	// Cut on a line boundary: the kept text ends with a complete line.
	kept := got[:strings.Index(got, "\n\n[Your MEMORY.md")]
	if last := kept[strings.LastIndexByte(kept, '\n')+1:]; !strings.HasPrefix(last, "- 2026-10-14: fact number") {
		t.Fatalf("kept text ends mid-line: %q", last)
	}
}

func TestMemoryContext_CapNeverSplitsARune(t *testing.T) {
	content := strings.Repeat("é", 3000) // 2 bytes each, no newlines
	got, capped := capMemoryBlock(content, 1001, "Your MEMORY.md", "Curate it.")
	if !capped {
		t.Fatal("expected the block to be capped")
	}
	if !utf8.ValidString(got) {
		t.Fatal("capped memory is not valid UTF-8")
	}
}

func TestMemoryContext_DailyNotesShareTheBudget(t *testing.T) {
	ws := t.TempDir()
	ms := NewMemoryStore(ws)
	ms.SetMaxBytes(2048)
	if err := ms.AppendToday(strings.Repeat("note line\n", 1000)); err != nil {
		t.Fatal(err)
	}
	got := ms.GetMemoryContext()
	if len(got) > 3*1024 {
		t.Fatalf("daily notes block = %d bytes, want about the 2 KB budget", len(got))
	}
	if !strings.Contains(got, "Your recent daily notes is") || !strings.Contains(got, "over the 2 KB budget") {
		t.Fatalf("daily notes over budget without a notice:\n%s", got)
	}
}

func TestMemoryMaxBytes_DefaultAndConfig(t *testing.T) {
	var d config.AgentDefaults
	if got := d.GetMemoryMaxBytes(); got != 32*1024 {
		t.Fatalf("default memory_max_bytes = %d, want 32768", got)
	}
	d.MemoryMaxBytes = 8192
	if got := d.GetMemoryMaxBytes(); got != 8192 {
		t.Fatalf("memory_max_bytes = %d, want 8192", got)
	}

	ws := t.TempDir()
	writeMemoryFile(t, ws, strings.Repeat("a line of memory\n", 1000)) // ~17 KB
	cb := NewContextBuilder(ws).WithMemoryMaxBytes(8192)
	msgs := cb.BuildMessages(nil, "", "hi", nil, "pico", "chat", "", "")
	turnCtx, ok := turnContextMessage(msgs)
	if !ok || !strings.Contains(turnCtx.Content, "over the 8 KB budget") {
		t.Fatalf("configured budget not applied to the turn context:\n%v", turnCtx.Content)
	}
}
