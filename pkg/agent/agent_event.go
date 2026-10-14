// PicoClaw - Ultra-lightweight personal AI agent

package agent

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/sipeed/picoclaw/pkg/bus"

	runtimeevents "github.com/sipeed/picoclaw/pkg/events"
)

func (al *AgentLoop) newTurnEventScope(agentID, sessionKey string, turnCtx *TurnContext) turnEventScope {
	seq := al.turnSeq.Add(1)
	turnCtx = cloneTurnContext(turnCtx)
	if turnCtx == nil {
		turnCtx = &TurnContext{}
	}
	if turnCtx.Inbound == nil {
		turnCtx.Inbound = &bus.InboundContext{}
	}
	if turnCtx.Inbound.Raw == nil {
		turnCtx.Inbound.Raw = make(map[string]string)
	}
	if turnCtx.Inbound.Raw["workflow_id"] == "" {
		in := turnCtx.Inbound
		id := uuid.NewString()
		if in.MessageID != "" {
			id = uuid.NewSHA1(uuid.NameSpaceURL, []byte(in.Channel+":"+in.ChatID+":"+in.MessageID)).String()
		}
		in.Raw["workflow_id"] = id
	}
	return turnEventScope{
		agentID:    agentID,
		sessionKey: sessionKey,
		turnID:     fmt.Sprintf("%s-turn-%d", agentID, seq),
		context:    cloneTurnContext(turnCtx),
	}
}

func (ts turnEventScope) meta(iteration int, source, tracePath string) HookMeta {
	meta := HookMeta{
		AgentID:     ts.agentID,
		TurnID:      ts.turnID,
		SessionKey:  ts.sessionKey,
		Iteration:   iteration,
		Source:      source,
		TracePath:   tracePath,
		turnContext: cloneTurnContext(ts.context),
	}
	if ts.context != nil && ts.context.Inbound != nil {
		meta.WorkflowID = ts.context.Inbound.Raw["workflow_id"]
		meta.ParentTurnID = ts.context.Inbound.Raw["parent_turn_id"]
	}
	return meta
}

func (al *AgentLoop) emitEvent(kind runtimeevents.Kind, meta HookMeta, payload any) {
	clonedMeta := cloneHookMeta(meta)
	eventCtx := cloneTurnContext(clonedMeta.turnContext)
	evt := runtimeevents.Event{
		Kind:        kind,
		Source:      runtimeevents.Source{Component: "agent", Name: clonedMeta.AgentID},
		Scope:       runtimeScopeFromHookMeta(clonedMeta, eventCtx),
		Correlation: runtimeCorrelationFromHookMeta(clonedMeta),
		Severity:    runtimeSeverityForAgentEvent(kind, payload),
		Payload:     payload,
		Attrs:       runtimeAttrsFromHookMeta(clonedMeta),
	}

	if al == nil {
		return
	}

	deliveredToEvolution := false
	if kind == runtimeevents.KindAgentTurnEnd {
		evolution := al.currentEvolutionBridge()
		if evolution != nil {
			deliveredToEvolution = evolution.handleRuntimeTurnEnd(evt)
		}
	}
	if deliveredToEvolution {
		if evt.Attrs == nil {
			evt.Attrs = make(map[string]any, 1)
		}
		evt.Attrs[evolutionDirectDeliveryAttr] = true
	}
	al.publishRuntimeEvent(evt)
}

func (al *AgentLoop) currentEvolutionBridge() *evolutionBridge {
	if al == nil {
		return nil
	}
	al.mu.RLock()
	defer al.mu.RUnlock()
	return al.evolution
}

func (al *AgentLoop) isCurrentEvolutionBridge(bridge *evolutionBridge) bool {
	if al == nil || bridge == nil {
		return false
	}
	al.mu.RLock()
	defer al.mu.RUnlock()
	return al.evolution == bridge
}

// MountHook registers an in-process hook on the agent loop.
func (al *AgentLoop) MountHook(reg HookRegistration) error {
	if al == nil || al.hooks == nil {
		return fmt.Errorf("hook manager is not initialized")
	}
	return al.hooks.Mount(reg)
}

// UnmountHook removes a previously registered in-process hook.
func (al *AgentLoop) UnmountHook(name string) {
	if al == nil || al.hooks == nil {
		return
	}
	al.hooks.Unmount(name)
}

// RuntimeEvents returns the root runtime event channel.
func (al *AgentLoop) RuntimeEvents() runtimeevents.EventChannel {
	if al == nil || al.runtimeEvents == nil {
		return nil
	}
	return al.runtimeEvents.Channel()
}

// RuntimeEventStats returns runtime event bus counters.
func (al *AgentLoop) RuntimeEventStats() runtimeevents.Stats {
	if al == nil || al.runtimeEvents == nil {
		return runtimeevents.Stats{Closed: true}
	}
	return al.runtimeEvents.Stats()
}

// RuntimeEventBus returns the runtime event bus used by the agent loop.
func (al *AgentLoop) RuntimeEventBus() runtimeevents.Bus {
	if al == nil {
		return nil
	}
	return al.runtimeEvents
}
