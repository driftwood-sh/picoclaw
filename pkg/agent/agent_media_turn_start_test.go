package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/media"
	"github.com/sipeed/picoclaw/pkg/providers"
)

const turnStartClampedMessage = "current turn start out of range, clamped"

// captureWarnLogs routes WARN-and-above log lines to a temp file and returns a
// function that reads what has been written so far.
func captureWarnLogs(t *testing.T) func() string {
	t.Helper()

	logFile := filepath.Join(t.TempDir(), "agent-warn.log")
	prevLevel := logger.GetLevel()
	logger.SetLevel(logger.WARN)
	if err := logger.EnableFileLogging(logFile); err != nil {
		t.Fatalf("EnableFileLogging() error = %v", err)
	}
	t.Cleanup(func() {
		logger.DisableFileLogging()
		logger.SetLevel(prevLevel)
	})

	return func() string {
		data, err := os.ReadFile(logFile)
		if err != nil {
			t.Fatalf("ReadFile(%q) error = %v", logFile, err)
		}
		return string(data)
	}
}

func TestResolveMediaRefs_WarnsWhenCurrentTurnStartIsOutOfRange(t *testing.T) {
	store := media.NewFileMediaStore()
	imagePath := filepath.Join(t.TempDir(), "secret-screenshot.png")
	pngHeader := []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
		0x00, 0x00, 0x00, 0x0D,
		0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02,
		0x00, 0x00, 0x00,
		0x90, 0x77, 0x53, 0xDE,
	}
	if err := os.WriteFile(imagePath, pngHeader, 0o644); err != nil {
		t.Fatal(err)
	}
	ref, _ := store.Store(imagePath, media.MediaMeta{}, "test")

	tests := []struct {
		name             string
		currentTurnStart int
		wantClamped      int
	}{
		{name: "past the end", currentTurnStart: 9, wantClamped: 3},
		{name: "negative", currentTurnStart: -2, wantClamped: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			readLogs := captureWarnLogs(t)

			messages := []providers.Message{
				{Role: "user", Content: "private question about my bank statement"},
				{Role: "assistant", Content: "loading the image"},
				toolResultPromptMessage("Image loaded", "call_out_of_range", []string{ref}),
			}
			ids := turnLogIDs{sessionKey: "session-out-of-range", turnID: "turn-out-of-range"}
			result := resolveMediaRefs(messages, store, config.DefaultMaxMediaSize, tt.currentTurnStart, ids)

			// Behavior is unchanged: the index is still clamped.
			wantLen := 3
			if tt.currentTurnStart < 0 {
				// Clamped to 0, the whole history is the current turn, so the
				// tool image is attached in a follow-up message.
				wantLen = 4
			}
			if len(result) != wantLen {
				t.Fatalf("len(result) = %d, want %d", len(result), wantLen)
			}

			logs := readLogs()
			if n := strings.Count(logs, turnStartClampedMessage); n != 1 {
				t.Fatalf("warning logged %d times, want once per call; log = %q", n, logs)
			}
			var entry map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(logs)), &entry); err != nil {
				t.Fatalf("log line is not JSON: %v; log = %q", err, logs)
			}
			want := map[string]any{
				"level":               "warn",
				"component":           "agent",
				"message":             turnStartClampedMessage,
				"current_turn_start":  float64(tt.currentTurnStart),
				"clamped_turn_start":  float64(tt.wantClamped),
				"message_count":       float64(3),
				"tool_media_messages": float64(1),
				"session_key":         "session-out-of-range",
				"turn_id":             "turn-out-of-range",
			}
			for key, wantValue := range want {
				if entry[key] != wantValue {
					t.Fatalf("log field %s = %#v, want %#v; log = %q", key, entry[key], wantValue, logs)
				}
			}
			for _, secret := range []string{imagePath, "private question", "bank statement", "base64"} {
				if strings.Contains(logs, secret) {
					t.Fatalf("log = %q, must not contain %q", logs, secret)
				}
			}
		})
	}
}

func TestResolveMediaRefs_DoesNotWarnWhenCurrentTurnStartIsInRange(t *testing.T) {
	readLogs := captureWarnLogs(t)
	store := media.NewFileMediaStore()

	messages := []providers.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "second"},
		{Role: "user", Content: "third"},
	}
	// 0 is the start of history and len(messages) is a turn that has no
	// messages of its own yet: both are legitimate.
	for _, start := range []int{0, 1, len(messages)} {
		resolveMediaRefs(messages, store, config.DefaultMaxMediaSize, start)
	}

	if logs := readLogs(); strings.Contains(logs, turnStartClampedMessage) {
		t.Fatalf("log = %q, want no out-of-range warning for in-range turn starts", logs)
	}
}

func TestResolveMediaRefs_WithoutTurnIDsStillWarns(t *testing.T) {
	readLogs := captureWarnLogs(t)
	store := media.NewFileMediaStore()

	resolveMediaRefs([]providers.Message{{Role: "user", Content: "hi"}}, store, config.DefaultMaxMediaSize, 5)

	if logs := readLogs(); !strings.Contains(logs, turnStartClampedMessage) {
		t.Fatalf("log = %q, want the out-of-range warning", logs)
	}
}
