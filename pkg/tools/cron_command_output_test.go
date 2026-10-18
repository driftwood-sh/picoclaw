package tools

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/cron"
)

func cronCommandJob(command string) *cron.CronJob {
	job := &cron.CronJob{ID: "job-42", Name: "nightly check"}
	job.Payload.Channel = "pico_client"
	job.Payload.To = "C0123"
	job.Payload.Command = command
	return job
}

func cronToolWithCommandOutput(t *testing.T, mode string) *CronTool {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.Tools.Cron.CommandOutput = mode
	return newTestCronToolWithConfig(t, cfg)
}

func noOutbound(t *testing.T, tool *CronTool) {
	t.Helper()
	select {
	case msg := <-tool.msgBus.OutboundChan():
		t.Fatalf("command output was posted to the chat: %q", msg.Content)
	case <-time.After(200 * time.Millisecond):
	}
}

func noInbound(t *testing.T, tool *CronTool) {
	t.Helper()
	select {
	case msg := <-tool.msgBus.InboundChan():
		t.Fatalf("command output was handed to the agent: %q", msg.Content)
	case <-time.After(200 * time.Millisecond):
	}
}

func TestCronCommandOutput_AgentModeHandsOutputToAgentNotChat(t *testing.T) {
	tool := cronToolWithCommandOutput(t, "agent")

	if got := tool.ExecuteJob(context.Background(), cronCommandJob("echo cron-agent-ok")); got != "ok" {
		t.Fatalf("ExecuteJob() = %q, want ok", got)
	}

	var msg bus.InboundMessage
	select {
	case msg = <-tool.msgBus.InboundChan():
	case <-time.After(2 * time.Second):
		t.Fatal("command output never reached the agent")
	}
	if msg.Channel != "system" || msg.SenderID != "async:cron" {
		t.Fatalf("inbound channel/sender = %s/%s, want system/async:cron", msg.Channel, msg.SenderID)
	}
	if msg.ChatID != "pico_client:C0123" {
		t.Fatalf("inbound chat = %q, want the job's origin pico_client:C0123", msg.ChatID)
	}
	for _, want := range []string{"job-42", "nightly check", "cron-agent-ok"} {
		if !strings.Contains(msg.Content, want) {
			t.Errorf("agent message is missing %q: %q", want, msg.Content)
		}
	}
	noOutbound(t, tool)
}

func TestCronCommandOutput_NoneModePostsNothing(t *testing.T) {
	tool := cronToolWithCommandOutput(t, "none")

	if got := tool.ExecuteJob(context.Background(), cronCommandJob("echo cron-none-ok")); got != "ok" {
		t.Fatalf("ExecuteJob() = %q, want ok", got)
	}
	noOutbound(t, tool)
	noInbound(t, tool)
}

func TestCronCommandOutput_AgentModeRoutesExecDisabledError(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Tools.Exec.Enabled = false
	cfg.Tools.Cron.CommandOutput = "agent"
	tool := newTestCronToolWithConfig(t, cfg)

	tool.ExecuteJob(context.Background(), cronCommandJob("df -h"))

	select {
	case msg := <-tool.msgBus.InboundChan():
		if !strings.Contains(msg.Content, "command execution is disabled") {
			t.Fatalf("agent message = %q, want the exec-disabled error", msg.Content)
		}
	case <-time.After(time.Second):
		t.Fatal("exec-disabled error never reached the agent")
	}
	noOutbound(t, tool)
}

// The default stays today's behavior: output posted to the chat as-is.
func TestCronCommandOutput_DefaultPostsToChat(t *testing.T) {
	tool := cronToolWithCommandOutput(t, "")

	tool.ExecuteJob(context.Background(), cronCommandJob("echo cron-chat-ok"))

	select {
	case msg := <-tool.msgBus.OutboundChan():
		if !strings.Contains(msg.Content, "cron-chat-ok") {
			t.Fatalf("posted %q, want the command output", msg.Content)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("default mode did not post the output")
	}
}

// The model learns where command output goes before it schedules a command.
func TestCronCommandOutput_DescriptionSaysWhereOutputGoes(t *testing.T) {
	cases := map[string]string{
		"":      "its output is posted to this chat as-is.",
		"chat":  "its output is posted to this chat as-is.",
		"agent": "its output comes back to you as a background result and is not posted.",
		"none":  "its output is not posted anywhere.",
		"bogus": "its output is posted to this chat as-is.",
	}
	for mode, want := range cases {
		desc := cronToolWithCommandOutput(t, mode).Description()
		if !strings.HasSuffix(desc, want) {
			t.Errorf("command_output %q: description ends %q, want %q", mode, desc[len(desc)-60:], want)
		}
	}
}
