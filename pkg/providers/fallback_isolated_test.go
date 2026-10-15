package providers

import (
	"context"
	"errors"
	"testing"
)

// TestFallbackChain_WithIsolatedCooldownDoesNotTouchParent: retryable
// failures seen through an isolated chain must not put candidates into the
// parent chain's cooldown (the summarizer uses this so summary failures never
// reroute or block the main turn).
func TestFallbackChain_WithIsolatedCooldownDoesNotTouchParent(t *testing.T) {
	parentCooldown := NewCooldownTracker()
	parent := NewFallbackChain(parentCooldown, nil)
	candidates := []FallbackCandidate{
		{Provider: "openai", Model: "primary"},
		{Provider: "openai", Model: "backup"},
	}
	fail := func(context.Context, FallbackCandidate) (*LLMResponse, error) {
		return nil, errors.New(`API request failed: Status: 500 Body: {"error":{"message":"server error"}}`)
	}

	if _, err := parent.WithIsolatedCooldown().ExecuteCandidate(context.Background(), candidates, fail); err == nil {
		t.Fatal("expected every candidate to fail")
	}
	for _, c := range candidates {
		if !parentCooldown.IsAvailable(c.StableKey()) {
			t.Fatalf("%s is in the parent chain's cooldown after an isolated-chain failure", c.Model)
		}
	}

	var nilChain *FallbackChain
	if nilChain.WithIsolatedCooldown() == nil {
		t.Fatal("WithIsolatedCooldown on a nil chain must still return a usable chain")
	}
}
