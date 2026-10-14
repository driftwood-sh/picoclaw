package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestWorkflowIdentitySurvivesChildAndCallback(t *testing.T) {
	al := &AgentLoop{}
	input := &TurnContext{Inbound: &bus.InboundContext{Channel: "pico", ChatID: "chat", MessageID: "request-1"}}
	root := al.newTurnEventScope("main", "session", input)
	child := al.newTurnEventScope("researcher", "subturn-1", root.context)
	callback := al.newTurnEventScope("main", "session", &TurnContext{Inbound: &bus.InboundContext{
		Channel: "system", Raw: map[string]string{
			"workflow_id": root.meta(1, "", "").WorkflowID, "parent_turn_id": root.turnID,
		},
	}})
	rootID := root.meta(1, "", "").WorkflowID
	if rootID == "" || child.meta(1, "", "").WorkflowID != rootID || callback.meta(1, "", "").WorkflowID != rootID {
		t.Fatal("root, worker and callback must share a nonempty workflow identity")
	}
	if input.Inbound.Raw != nil {
		t.Fatal("scope creation must not mutate the caller's context")
	}
	if callback.meta(1, "", "").ParentTurnID != root.turnID {
		t.Fatal("callback lost its parent turn")
	}
	restarted := (&AgentLoop{}).newTurnEventScope("main", "session", input)
	if restarted.meta(1, "", "").WorkflowID != rootID {
		t.Fatal("re-delivery after restart must keep assignment")
	}
	other := al.newTurnEventScope("main", "session", nil)
	other2 := (&AgentLoop{}).newTurnEventScope("main", "session", nil)
	if other.meta(1, "", "").WorkflowID == other2.meta(1, "", "").WorkflowID {
		t.Fatal("requests without platform ids must not collide across restarts")
	}
}

type trialModelHook struct{}

func (trialModelHook) BeforeLLM(_ context.Context, req *LLMHookRequest) (*LLMHookRequest, HookDecision, error) {
	next := req.Clone()
	if req.Meta.Iteration == 1 {
		next.Model = "test-model"
		next.Fallbacks = []string{"fallback-model"}
	}
	return next, HookDecision{}, nil
}

func (trialModelHook) AfterLLM(_ context.Context, resp *LLMHookResponse) (*LLMHookResponse, HookDecision, error) {
	return resp, HookDecision{}, nil
}

func TestTrialHookKeepsRecoveryChain(t *testing.T) {
	cfg := newConfiguredStreamingTestConfig(t, false, false, []string{"fallback-model"})
	provider := &sequencedChatProvider{plan: []*providers.LLMResponse{
		{}, {}, {Content: "recovered"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	if err := al.MountHook(NamedHook("trial", trialModelHook{})); err != nil {
		t.Fatal(err)
	}
	got := runConfiguredStreamingTurn(t, al, "pico")
	if got != "recovered" || len(provider.chatModels) < 2 {
		t.Fatalf("recovery failed: %q, %v", got, provider.chatModels)
	}
	if !reflect.DeepEqual(provider.chatModels, []string{"openai/test-model", "openai/test-model", "openai/fallback-model"}) {
		t.Fatalf("unexpected recovery chain: %v", provider.chatModels)
	}
}
