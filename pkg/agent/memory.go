// PicoClaw - Ultra-lightweight personal AI agent
// Inspired by and based on nanobot: https://github.com/HKUDS/nanobot
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/fileutil"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// MemoryStore manages persistent memory for the agent.
// - Long-term memory: memory/MEMORY.md
// - Daily notes: memory/YYYYMM/YYYYMMDD.md
type MemoryStore struct {
	workspace  string
	memoryDir  string
	memoryFile string
	maxBytes   int // injection budget per block; <= 0 uses config.DefaultMemoryMaxBytes
}

// NewMemoryStore creates a new MemoryStore with the given workspace path.
// It ensures the memory directory exists.
func NewMemoryStore(workspace string) *MemoryStore {
	memoryDir := filepath.Join(workspace, "memory")
	memoryFile := filepath.Join(memoryDir, "MEMORY.md")

	// Ensure memory directory exists
	os.MkdirAll(memoryDir, 0o755)

	return &MemoryStore{
		workspace:  workspace,
		memoryDir:  memoryDir,
		memoryFile: memoryFile,
	}
}

// getTodayFile returns the path to today's daily note file (memory/YYYYMM/YYYYMMDD.md).
func (ms *MemoryStore) getTodayFile() string {
	today := time.Now().Format("20060102") // YYYYMMDD
	monthDir := today[:6]                  // YYYYMM
	filePath := filepath.Join(ms.memoryDir, monthDir, today+".md")
	return filePath
}

// ReadLongTerm reads the long-term memory (MEMORY.md).
// Returns empty string if the file doesn't exist.
func (ms *MemoryStore) ReadLongTerm() string {
	if data, err := os.ReadFile(ms.memoryFile); err == nil {
		return string(data)
	}
	return ""
}

// WriteLongTerm writes content to the long-term memory file (MEMORY.md).
func (ms *MemoryStore) WriteLongTerm(content string) error {
	// Use unified atomic write utility with explicit sync for flash storage reliability.
	// Using 0o600 (owner read/write only) for secure default permissions.
	return fileutil.WriteFileAtomic(ms.memoryFile, []byte(content), 0o600)
}

// ReadToday reads today's daily note.
// Returns empty string if the file doesn't exist.
func (ms *MemoryStore) ReadToday() string {
	todayFile := ms.getTodayFile()
	if data, err := os.ReadFile(todayFile); err == nil {
		return string(data)
	}
	return ""
}

// AppendToday appends content to today's daily note.
// If the file doesn't exist, it creates a new file with a date header.
func (ms *MemoryStore) AppendToday(content string) error {
	todayFile := ms.getTodayFile()

	// Ensure month directory exists
	monthDir := filepath.Dir(todayFile)
	if err := os.MkdirAll(monthDir, 0o755); err != nil {
		return err
	}

	var existingContent string
	if data, err := os.ReadFile(todayFile); err == nil {
		existingContent = string(data)
	}

	var newContent string
	if existingContent == "" {
		// Add header for new day
		header := fmt.Sprintf("# %s\n\n", time.Now().Format("2006-01-02"))
		newContent = header + content
	} else {
		// Append to existing content
		newContent = existingContent + "\n" + content
	}

	// Use unified atomic write utility with explicit sync for flash storage reliability.
	return fileutil.WriteFileAtomic(todayFile, []byte(newContent), 0o600)
}

// GetRecentDailyNotes returns daily notes from the last N days.
// Contents are joined with "---" separator.
func (ms *MemoryStore) GetRecentDailyNotes(days int) string {
	var sb strings.Builder
	first := true

	for i := range days {
		date := time.Now().AddDate(0, 0, -i)
		dateStr := date.Format("20060102") // YYYYMMDD
		monthDir := dateStr[:6]            // YYYYMM
		filePath := filepath.Join(ms.memoryDir, monthDir, dateStr+".md")

		if data, err := os.ReadFile(filePath); err == nil {
			if !first {
				sb.WriteString("\n\n---\n\n")
			}
			sb.Write(data)
			first = false
		}
	}

	return sb.String()
}

// SetMaxBytes sets the injection budget applied to MEMORY.md and, separately,
// to the recent daily notes. Non-positive values use the config default.
func (ms *MemoryStore) SetMaxBytes(n int) {
	ms.maxBytes = n
}

func (ms *MemoryStore) budget() int {
	if ms.maxBytes > 0 {
		return ms.maxBytes
	}
	return config.DefaultMemoryMaxBytes
}

// capMemoryBlock keeps at most maxBytes of an over-budget block: the head
// (first third of the budget) and the tail (last two thirds), each cut on a
// line boundary and never inside a UTF-8 sequence, with a gap marker between
// them and a notice at the end. The head holds the file's oldest curated core
// and structure; the tail holds the newest facts, which agents append at the
// end — keeping only the head would hide every new fact and the agent would
// keep re-adding them. Memory is injected into every request, and an uncapped
// MEMORY.md once grew to 918 KB and pushed the request past OpenAI's 1 MB
// instructions limit, failing every turn until the file was trimmed by hand.
func capMemoryBlock(content string, maxBytes int, what, shrinkHint string) (string, bool) {
	if len(content) <= maxBytes {
		return content, false
	}
	headBudget := maxBytes / 3
	tailBudget := maxBytes - headBudget

	headEnd := headBudget
	for headEnd > 0 && !utf8.RuneStart(content[headEnd]) {
		headEnd--
	}
	if nl := strings.LastIndexByte(content[:headEnd], '\n'); nl >= headBudget/2 {
		headEnd = nl + 1
	}

	tailStart := len(content) - tailBudget
	for tailStart < len(content) && !utf8.RuneStart(content[tailStart]) {
		tailStart++
	}
	if tailStart > 0 && content[tailStart-1] != '\n' {
		if nl := strings.IndexByte(content[tailStart:], '\n'); nl >= 0 && nl < tailBudget/2 {
			tailStart += nl + 1
		}
	}

	head := strings.TrimRight(content[:headEnd], "\n")
	tail := strings.TrimRight(content[tailStart:], "\n")
	gap := fmt.Sprintf("[... %s omitted from the middle ...]", formatKB(tailStart-headEnd))
	notice := fmt.Sprintf(
		"[%s is %s, over the %s budget; only its start and end are shown. %s]",
		what, formatKB(len(content)), formatKB(maxBytes), shrinkHint,
	)
	return head + "\n\n" + gap + "\n\n" + tail + "\n\n" + notice, true
}

func formatKB(n int) string {
	return fmt.Sprintf("%d KB", (n+1023)/1024)
}

// GetMemoryContext returns formatted memory context for the agent prompt.
// Includes long-term memory and recent daily notes, each capped at the
// memory budget (agents.defaults.memory_max_bytes).
func (ms *MemoryStore) GetMemoryContext() string {
	longTerm := ms.ReadLongTerm()
	recentNotes := ms.GetRecentDailyNotes(3)

	if longTerm == "" && recentNotes == "" {
		return ""
	}

	maxBytes := ms.budget()
	var cappedLong, cappedNotes bool
	longTermSize, notesSize := len(longTerm), len(recentNotes)
	longTerm, cappedLong = capMemoryBlock(longTerm, maxBytes, "Your MEMORY.md ("+ms.memoryFile+")",
		"Shrink it below "+formatKB(maxBytes)+" now with edit_file or exec (delete stale and "+
			"duplicate lines, merge related ones); don't read the whole file.")
	recentNotes, cappedNotes = capMemoryBlock(recentNotes, maxBytes, "Your recent daily notes",
		"Shrink them below "+formatKB(maxBytes)+" now with edit_file or exec (move what still "+
			"matters into MEMORY.md, delete the rest); don't read the whole files.")
	if cappedLong || cappedNotes {
		logger.WarnCF("agent", "Workspace memory over budget; injecting a truncated copy", map[string]any{
			"workspace":        ms.workspace,
			"memory_bytes":     longTermSize,
			"daily_note_bytes": notesSize,
			"max_bytes":        maxBytes,
		})
	}

	var sb strings.Builder

	if longTerm != "" {
		sb.WriteString("## Long-term Memory\n\n")
		sb.WriteString(longTerm)
	}

	if recentNotes != "" {
		if longTerm != "" {
			sb.WriteString("\n\n---\n\n")
		}
		sb.WriteString("## Recent Daily Notes\n\n")
		sb.WriteString(recentNotes)
	}

	return sb.String()
}
