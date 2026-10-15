package integrationtools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func writeArtifactWithAge(t *testing.T, dir, name string, age time.Duration) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("payload"), 0o600); err != nil {
		t.Fatal(err)
	}
	mtime := time.Now().Add(-age)
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	return path
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// TestMCPTool_LargeTextSpillPrunesExpiredArtifacts: spilling a large MCP
// result sweeps .artifacts/mcp/ and deletes files older than the retention
// window, keeping recent ones and the new spill.
func TestMCPTool_LargeTextSpillPrunesExpiredArtifacts(t *testing.T) {
	workspace := t.TempDir()
	dir := filepath.Join(workspace, ".artifacts", "mcp")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	expired := writeArtifactWithAge(t, dir, "srv_tool_old.txt", 8*24*time.Hour)
	recent := writeArtifactWithAge(t, dir, "srv_tool_recent.txt", 2*24*time.Hour)
	subdir := filepath.Join(dir, "keep-dir")
	if err := os.Mkdir(subdir, 0o700); err != nil {
		t.Fatal(err)
	}

	largeText := strings.Repeat("large MCP payload line\n", 1200)
	manager := &MockMCPManager{
		callToolFunc: func(context.Context, string, string, map[string]any) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: largeText}}}, nil
		},
	}
	tool := NewMCPTool(manager, "srv", &mcp.Tool{Name: "dump"})
	tool.SetWorkspace(workspace)

	result := tool.Execute(context.Background(), nil)
	if len(result.ArtifactTags) != 1 {
		t.Fatalf("expected the new spill to be saved, got tags %v", result.ArtifactTags)
	}
	newPath := strings.TrimSuffix(strings.TrimPrefix(result.ArtifactTags[0], "[file:"), "]")

	if exists(expired) {
		t.Error("artifact older than the retention window was not deleted")
	}
	if !exists(recent) {
		t.Error("artifact inside the retention window was deleted")
	}
	if !exists(newPath) {
		t.Error("the new spill was deleted")
	}
	if !exists(subdir) {
		t.Error("retention must only delete regular files")
	}
}

// TestPruneMCPArtifacts_ThrottledPerDirectory: a burst of spills sweeps the
// folder once per interval, not on every write.
func TestPruneMCPArtifacts_ThrottledPerDirectory(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	pruneMCPArtifacts(dir, now)

	expired := writeArtifactWithAge(t, dir, "late_old.txt", 30*24*time.Hour)
	pruneMCPArtifacts(dir, now.Add(time.Minute))
	if !exists(expired) {
		t.Fatal("sweep ran again inside the throttle interval")
	}
	pruneMCPArtifacts(dir, now.Add(mcpArtifactPruneInterval+time.Minute))
	if exists(expired) {
		t.Fatal("sweep after the throttle interval did not delete the expired artifact")
	}
}
