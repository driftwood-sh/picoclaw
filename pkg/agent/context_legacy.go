package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/sipeed/picoclaw/pkg/config"
	runtimeevents "github.com/sipeed/picoclaw/pkg/events"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// legacyContextManager wraps the existing summarization/compression logic
// as a ContextManager implementation. It is the default when no other
// ContextManager is configured.
type legacyContextManager struct {
	al          *AgentLoop
	summarizing sync.Map // dedup for async Compact (post-turn)
}

func (m *legacyContextManager) Assemble(_ context.Context, req *AssembleRequest) (*AssembleResponse, error) {
	// Legacy: read history from session, return as-is.
	// Budget enforcement happens in BuildMessages caller via
	// isOverContextBudget + forceCompression.
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return &AssembleResponse{}, nil
	}
	history := agent.Sessions.GetHistory(req.SessionKey)
	// Clean on read: a summary written by an older build may be nothing but
	// stacked compression notes. Cleaning here fixes the prompt at once,
	// without waiting for the next write.
	summary := cleanSessionSummary(agent.Sessions.GetSummary(req.SessionKey))
	return &AssembleResponse{
		History: history,
		Summary: summary,
	}, nil
}

func (m *legacyContextManager) Compact(ctx context.Context, req *CompactRequest) error {
	switch req.Reason {
	case ContextCompressReasonProactive, ContextCompressReasonRetry:
		// Sync emergency compression — budget exceeded.
		if result, ok := m.forceCompression(ctx, req.SessionKey); ok {
			m.al.emitEvent(
				runtimeevents.KindAgentContextCompress,
				m.al.newTurnEventScope("", req.SessionKey, nil).meta(0, "forceCompression", "turn.context.compress"),
				ContextCompressPayload{
					Reason:            req.Reason,
					DroppedMessages:   result.DroppedMessages,
					RemainingMessages: result.RemainingMessages,
					Summarized:        result.Summarized,
				},
			)
		}
	case ContextCompressReasonSummarize:
		m.maybeSummarize(req.SessionKey)
	}
	return nil
}

func (m *legacyContextManager) Ingest(_ context.Context, _ *IngestRequest) error {
	// Legacy: no-op. Messages are persisted by Sessions JSONL.
	return nil
}

func (m *legacyContextManager) Clear(_ context.Context, sessionKey string) error {
	// Routed (non-default) agents keep history in their own session store,
	// so resolve the owning agent instead of assuming the default one.
	agent := m.al.agentForSession(sessionKey)
	if agent == nil || agent.Sessions == nil {
		return fmt.Errorf("sessions not initialized")
	}
	agent.Sessions.SetHistory(sessionKey, []providers.Message{})
	agent.Sessions.SetSummary(sessionKey, "")
	return agent.Sessions.Save(sessionKey)
}

// maybeSummarize triggers summarization if the session history exceeds thresholds.
// It runs asynchronously in a goroutine.
func (m *legacyContextManager) maybeSummarize(sessionKey string) {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return
	}

	newHistory := agent.Sessions.GetHistory(sessionKey)
	tokenEstimate := m.estimateTokens(newHistory)
	threshold := agent.ContextWindow * agent.SummarizeTokenPercent / 100

	if len(newHistory) > agent.SummarizeMessageThreshold || tokenEstimate > threshold {
		summarizeKey := agent.ID + ":" + sessionKey
		if _, loading := m.summarizing.LoadOrStore(summarizeKey, true); !loading {
			go func() {
				defer m.summarizing.Delete(summarizeKey)
				defer func() {
					if r := recover(); r != nil {
						logger.WarnCF("agent", "Summarization panic recovered", map[string]any{
							"session_key": sessionKey,
							"panic":       r,
						})
					}
				}()
				logger.Debug("Memory threshold reached. Optimizing conversation history...")
				m.summarizeSession(agent, sessionKey)
			}()
		}
	}
}

type compressionResult struct {
	DroppedMessages   int
	RemainingMessages int
	// Summarized reports that the dropped messages were folded into the
	// session summary. False means the summary was left as it was.
	Summarized bool
}

// forceCompression aggressively reduces context when the limit is hit.
// It drops the oldest ~50% of Turns (a Turn is a complete user→LLM→response
// cycle, as defined in #1316), so tool-call sequences are never split.
//
// The dropped messages are folded into the session summary with one model
// call. If that call fails, the summary stays as it was: the summary only
// ever holds a real summary, never a note that something was dropped. (Older
// builds appended "[Emergency compression dropped N oldest messages…]" on
// every compression and never trimmed it; one agent's summary grew to 129 KB
// of nothing but those notes, re-sent on every call.)
func (m *legacyContextManager) forceCompression(ctx context.Context, sessionKey string) (compressionResult, bool) {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return compressionResult{}, false
	}

	history := agent.Sessions.GetHistory(sessionKey)
	if len(history) <= 2 {
		return compressionResult{}, false
	}

	turns := parseTurnBoundaries(history)
	var mid int
	if len(turns) >= 2 {
		mid = turns[len(turns)/2]
	} else {
		mid = findSafeBoundary(history, len(history)/2)
	}
	var keptHistory, dropped []providers.Message
	if mid <= 0 {
		for i := len(history) - 1; i >= 0; i-- {
			if history[i].Role == "user" {
				keptHistory = []providers.Message{history[i]}
				dropped = append(append([]providers.Message(nil), history[:i]...), history[i+1:]...)
				break
			}
		}
		if keptHistory == nil {
			dropped = history
		}
	} else {
		keptHistory = history[mid:]
		dropped = history[:mid]
	}

	droppedCount := len(history) - len(keptHistory)

	existingSummary := cleanSessionSummary(agent.Sessions.GetSummary(sessionKey))
	newSummary, err := m.summarizeDropped(ctx, agent, dropped, existingSummary)
	summarized := err == nil && newSummary != ""
	if !summarized {
		fields := map[string]any{
			"session_key":  sessionKey,
			"dropped_msgs": droppedCount,
		}
		if err != nil {
			fields["error"] = err.Error()
		}
		logger.WarnCF("agent", "Emergency compression could not summarize the dropped messages; "+
			"keeping the previous summary", fields)
		newSummary = existingSummary
	}

	// The summary call takes a while. Keep anything appended to the session
	// meanwhile; the turn that asked for this compression is blocked on it,
	// so in practice nothing is.
	if current := agent.Sessions.GetHistory(sessionKey); len(current) > len(history) &&
		messageSlicesEquivalent(current[:len(history)], history) {
		keptHistory = append(append([]providers.Message(nil), keptHistory...), current[len(history):]...)
	}

	agent.Sessions.SetSummary(sessionKey, cleanSessionSummary(newSummary))
	agent.Sessions.SetHistory(sessionKey, keptHistory)
	agent.Sessions.Save(sessionKey)

	logger.WarnCF("agent", "Forced compression executed", map[string]any{
		"session_key":  sessionKey,
		"dropped_msgs": droppedCount,
		"new_count":    len(keptHistory),
		"summarized":   summarized,
	})

	return compressionResult{
		DroppedMessages:   droppedCount,
		RemainingMessages: len(keptHistory),
		Summarized:        summarized,
	}, true
}

const (
	// maxSessionSummaryBytes caps the stored and injected conversation
	// summary. The summary rides in every request of the session, so a
	// summary that grows without bound costs tokens on every call.
	maxSessionSummaryBytes = 6 * 1024
	// summaryWordBudget is the length the summarizer is asked to stay under;
	// maxSessionSummaryBytes is only the backstop.
	summaryWordBudget = 500
	// summaryPromptPrefix opens every summary prompt.
	summaryPromptPrefix = "Write the conversation summary."
	// summaryTrimmedMarker opens a summary that was cut to the cap.
	summaryTrimmedMarker = "[Older summary text trimmed to fit.]"

	// emergencySummaryTimeout bounds the summary call made inside a turn by
	// forceCompression. The turn waits on it, so it is shorter than the
	// background summarizeTimeout.
	emergencySummaryTimeout = 2 * time.Minute
	// emergencySummaryAttempts is how often forceCompression tries the
	// summary call before keeping the previous summary.
	emergencySummaryAttempts = 2
	// Per-message caps for the transcript forceCompression summarizes, and
	// a cap on the whole transcript (its newest part is kept).
	droppedTextMaxChars       = 2000
	droppedToolResultMaxChars = 600
	droppedToolArgsMaxChars   = 300
	droppedTranscriptMaxChars = 120000
)

// compressionNotePattern matches the drop notes older builds appended to the
// summary on every emergency compression.
var compressionNotePattern = regexp.MustCompile(
	`\[Emergency compression dropped \d+ oldest messages due to context limit\]`,
)

var blankLinesPattern = regexp.MustCompile(`\n{3,}`)

// cleanSessionSummary removes the stacked compression notes older builds
// wrote into the summary and caps what is left at maxSessionSummaryBytes.
func cleanSessionSummary(summary string) string {
	if summary == "" {
		return ""
	}
	cleaned := compressionNotePattern.ReplaceAllString(summary, "")
	cleaned = blankLinesPattern.ReplaceAllString(cleaned, "\n\n")
	return capSessionSummary(strings.TrimSpace(cleaned), maxSessionSummaryBytes)
}

// capSessionSummary keeps the newest maxBytes of the summary (at a line
// boundary when one is near), behind a one-line marker. The marker never
// stacks: a summary that already carries it is re-cut from its body.
func capSessionSummary(summary string, maxBytes int) string {
	if len(summary) <= maxBytes {
		return summary
	}
	body := strings.TrimSpace(strings.TrimPrefix(summary, summaryTrimmedMarker))
	keep := maxBytes - len(summaryTrimmedMarker) - 1
	if keep <= 0 {
		return ""
	}
	if len(body) > keep {
		body = body[len(body)-keep:]
		for len(body) > 0 && !utf8.RuneStart(body[0]) {
			body = body[1:]
		}
		if i := strings.IndexByte(body, '\n'); i >= 0 && i < len(body)/4 {
			body = body[i+1:]
		}
	}
	return summaryTrimmedMarker + "\n" + body
}

// summaryInstructions is the instruction block for every summary call. The
// summary is read at the start of every later turn, so it keeps what later
// work depends on and drops the rest.
func summaryInstructions() string {
	return fmt.Sprintf(
		summaryPromptPrefix+` It is read at the start of every later turn of this conversation, so keep only what later work depends on:
- the user's instructions and decisions, quoted verbatim;
- open tasks and where each stands;
- ids the work depends on: pending approval ids with the exact arguments they were requested with, job ids, run ids, file paths and URLs;
- deadlines and dates.
Leave out routine tool chatter and finished work nothing depends on. Write plain text, at most %d words.
`,
		summaryWordBudget,
	)
}

// summarizeDropped folds the messages forceCompression drops into the
// existing summary. Tool calls and tool results are included in short form,
// because the ids later work needs (jobs, runs, approvals) usually arrive in
// them. Returns "" with nil error when there is nothing to summarize.
func (m *legacyContextManager) summarizeDropped(
	ctx context.Context,
	agent *AgentInstance,
	dropped []providers.Message,
	existingSummary string,
) (string, error) {
	transcript := droppedTranscript(dropped)
	if transcript == "" {
		return "", nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, emergencySummaryTimeout)
	defer cancel()

	var sb strings.Builder
	sb.WriteString(summaryInstructions())
	if existingSummary != "" {
		sb.WriteString("\nEXISTING SUMMARY (fold it in; it covers what came before these messages):\n")
		sb.WriteString(existingSummary)
		sb.WriteString("\n")
	}
	sb.WriteString("\nMESSAGES TO FOLD IN (oldest first):\n")
	sb.WriteString(transcript)

	resp, err := m.retryLLMCall(ctx, agent, sb.String(), emergencySummaryAttempts)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// droppedTranscript renders messages for the emergency summary: text in
// full up to a cap, tool calls as name(args), tool results truncated. The
// whole transcript keeps its newest droppedTranscriptMaxChars.
func droppedTranscript(messages []providers.Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "user", "assistant":
			if text := strings.TrimSpace(msg.Content); text != "" {
				fmt.Fprintf(&sb, "%s: %s\n", msg.Role, truncateRunes(text, droppedTextMaxChars))
			}
			for _, tc := range msg.ToolCalls {
				name := tc.Name
				args := ""
				if tc.Function != nil {
					if name == "" {
						name = tc.Function.Name
					}
					args = tc.Function.Arguments
				}
				if args == "" && len(tc.Arguments) > 0 {
					if raw, err := json.Marshal(tc.Arguments); err == nil {
						args = string(raw)
					}
				}
				if name == "" {
					continue
				}
				fmt.Fprintf(&sb, "assistant called %s(%s)\n", name, truncateRunes(args, droppedToolArgsMaxChars))
			}
		case "tool":
			if text := strings.TrimSpace(msg.Content); text != "" {
				fmt.Fprintf(&sb, "tool result: %s\n", truncateRunes(text, droppedToolResultMaxChars))
			}
		}
	}
	out := sb.String()
	if len(out) > droppedTranscriptMaxChars {
		out = out[len(out)-droppedTranscriptMaxChars:]
		for len(out) > 0 && !utf8.RuneStart(out[0]) {
			out = out[1:]
		}
		if i := strings.IndexByte(out, '\n'); i >= 0 {
			out = out[i+1:]
		}
		out = "[earlier messages omitted]\n" + out
	}
	return out
}

func truncateRunes(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}

// summarizeTimeout bounds one background summarization (up to three model
// calls with retries). Reasoning models at high effort routinely need more
// than the old 120s for a long transcript; model request timeouts run 600s.
const summarizeTimeout = 10 * time.Minute

func (m *legacyContextManager) summarizeSession(agent *AgentInstance, sessionKey string) {
	ctx, cancel := context.WithTimeout(context.Background(), summarizeTimeout)
	defer cancel()

	history := agent.Sessions.GetHistory(sessionKey)
	summary := cleanSessionSummary(agent.Sessions.GetSummary(sessionKey))

	if len(history) <= 4 {
		return
	}

	safeCut := findSafeBoundary(history, len(history)-4)
	if safeCut <= 0 {
		return
	}
	keepCount := len(history) - safeCut
	toSummarize := history[:safeCut]

	maxMessageTokens := agent.ContextWindow / 2
	validMessages := make([]providers.Message, 0)
	omitted := false

	for _, msg := range toSummarize {
		if msg.Role != "user" && msg.Role != "assistant" {
			continue
		}
		msgTokens := len(msg.Content) / 2
		if msgTokens > maxMessageTokens {
			omitted = true
			continue
		}
		validMessages = append(validMessages, msg)
	}

	if len(validMessages) == 0 {
		return
	}

	const (
		maxSummarizationMessages = 10
		llmMaxRetries            = 3
	)

	var finalSummary string
	if len(validMessages) > maxSummarizationMessages {
		mid := len(validMessages) / 2
		mid = m.findNearestUserMessage(validMessages, mid)

		part1 := validMessages[:mid]
		part2 := validMessages[mid:]

		// The existing summary covers everything before part1. It must ride
		// with the first half, or the merged result (which replaces it)
		// silently drops all earlier context.
		s1, _ := m.summarizeBatch(ctx, agent, part1, summary)
		s2, _ := m.summarizeBatch(ctx, agent, part2, "")

		mergePrompt := fmt.Sprintf(
			"%s\nMerge these two consecutive summaries of the conversation into one.\n\n1 (older): %s\n\n2 (newer): %s",
			summaryInstructions(), s1, s2,
		)

		resp, err := m.retryLLMCall(ctx, agent, mergePrompt, llmMaxRetries)
		if err == nil && resp.Content != "" {
			finalSummary = resp.Content
		} else {
			logger.WarnCF("agent", "Summary merge failed; concatenating the two halves", map[string]any{
				"agent_id":    agent.ID,
				"session_key": sessionKey,
			})
			finalSummary = s1 + " " + s2
		}
	} else {
		finalSummary, _ = m.summarizeBatch(ctx, agent, validMessages, summary)
	}

	if omitted && finalSummary != "" {
		finalSummary += "\n[Note: Some oversized messages were omitted from this summary for efficiency.]"
	}

	if finalSummary != "" {
		// The model calls above take a while, and the next turn may already
		// have appended messages (or compacted the session). Truncate against
		// the history as it is now: drop exactly the summarized prefix, and
		// only if it is still the prefix.
		current := agent.Sessions.GetHistory(sessionKey)
		if len(current) < safeCut || !messageSlicesEquivalent(current[:safeCut], history[:safeCut]) {
			logger.WarnCF("agent", "Session changed during summarization; discarding the summary", map[string]any{
				"agent_id":    agent.ID,
				"session_key": sessionKey,
			})
			return
		}
		keepCount = len(current) - safeCut
		finalSummary = cleanSessionSummary(finalSummary)
		agent.Sessions.SetSummary(sessionKey, finalSummary)
		agent.Sessions.TruncateHistory(sessionKey, keepCount)
		agent.Sessions.Save(sessionKey)
		m.al.emitEvent(
			runtimeevents.KindAgentSessionSummarize,
			m.al.newTurnEventScope(agent.ID, sessionKey, nil).meta(0, "summarizeSession", "turn.session.summarize"),
			SessionSummarizePayload{
				SummarizedMessages: len(validMessages),
				KeptMessages:       keepCount,
				SummaryLen:         len(finalSummary),
				OmittedOversized:   omitted,
			},
		)
	}
}

func (m *legacyContextManager) findNearestUserMessage(messages []providers.Message, mid int) int {
	originalMid := mid

	for mid > 0 && messages[mid].Role != "user" {
		mid--
	}

	if messages[mid].Role == "user" {
		return mid
	}

	mid = originalMid
	for mid < len(messages) && messages[mid].Role != "user" {
		mid++
	}

	if mid < len(messages) {
		return mid
	}

	return originalMid
}

func (m *legacyContextManager) retryLLMCall(
	ctx context.Context,
	agent *AgentInstance,
	prompt string,
	maxRetries int,
) (*providers.LLMResponse, error) {
	var resp *providers.LLMResponse
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		m.al.activeRequestsInc()
		resp, err = func() (*providers.LLMResponse, error) {
			defer m.al.activeRequestsDec()
			return m.summarizerChat(ctx, agent, prompt)
		}()

		if err == nil && resp != nil && resp.Content != "" {
			return resp, nil
		}
		fields := map[string]any{
			"agent_id": agent.ID,
			"attempt":  attempt + 1,
			"of":       maxRetries,
		}
		if err != nil {
			fields["error"] = err.Error()
		} else {
			fields["error"] = "empty response"
		}
		logger.WarnCF("agent", "Summarization LLM call failed", fields)
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
		}
	}

	if err == nil {
		err = fmt.Errorf("summarization returned an empty response")
	}
	return resp, err
}

// summarizerChat sends one summarization prompt the way the main turn calls
// its model: the agent's resolved candidates (API model id, not the config
// alias) and the candidate's provider, through the fallback chain when there
// is more than one candidate. Reasoning is capped at low (summarizerThinking).
// It sends no temperature: reasoning models reject any non-default value, and
// the old hard-coded 0.3 made every summarization request fail on gpt-6.1-sol.
func (m *legacyContextManager) summarizerChat(
	ctx context.Context,
	agent *AgentInstance,
	prompt string,
) (*providers.LLMResponse, error) {
	messages := []providers.Message{{Role: "user", Content: prompt}}
	baseOpts := map[string]any{
		"max_tokens":       agent.MaxTokens,
		"prompt_cache_key": agent.ID + ":summary",
	}

	candidates, _ := refusalHoldSelection(
		agent,
		agent.Candidates,
		resolvedCandidateModel(agent.Candidates, agent.Model),
	)
	if len(candidates) == 0 {
		// No resolved candidates (legacy single-provider setups): the agent's
		// provider and model are all there is.
		return agent.Provider.Chat(ctx, messages, nil, agent.Model, baseOpts)
	}

	var cfg *config.Config
	if m.al != nil {
		cfg = m.al.GetConfig()
	}
	run := func(ctx context.Context, candidate providers.FallbackCandidate) (*providers.LLMResponse, error) {
		provider, err := providerForFallbackCandidate(
			agent, agent.Provider, candidates, candidate.Provider, candidate.Model,
		)
		if err != nil {
			return nil, err
		}
		opts := shallowCloneLLMOptions(baseOpts)
		if cfg != nil {
			modelCfg := resolveActiveModelConfig(
				cfg,
				agent.Workspace,
				[]providers.FallbackCandidate{candidate},
				candidate.Model,
				cfg.Agents.Defaults.Provider,
			)
			thinking := summarizerThinking(activeThinkingSettings(agent, modelCfg))
			applyThinkingOption(opts, provider, thinking, false, agent.ID)
		}
		return provider.Chat(ctx, messages, nil, candidate.Model, opts)
	}

	if len(candidates) > 1 {
		// Never the main turn's chain: its cooldown tracker decides which
		// model the user's next turn runs on, and a retryable summary failure
		// would put the primary (or every candidate) into cooldown -- the next
		// turn then silently ran on the fallback model with a cold cache, or
		// failed without reaching the provider. A fresh tracker per call also
		// lets the summary retries actually retry each candidate.
		var shared *providers.FallbackChain
		if m.al != nil {
			shared = m.al.fallback
		}
		result, err := shared.WithIsolatedCooldown().ExecuteCandidate(ctx, candidates, run)
		if err != nil {
			logger.WarnCF("agent", "Summary failed on every candidate model", map[string]any{
				"agent_id":   agent.ID,
				"candidates": candidateModels(candidates),
				"error":      err.Error(),
			})
			return nil, err
		}
		if len(result.Attempts) > 0 {
			logger.WarnCF("agent", "Summary fell back to another model", map[string]any{
				"agent_id":        agent.ID,
				"primary_model":   candidates[0].Model,
				"answered_by":     result.Model,
				"failed_attempts": len(result.Attempts),
			})
		}
		return result.Response, nil
	}
	return run(ctx, candidates[0])
}

func candidateModels(candidates []providers.FallbackCandidate) []string {
	models := make([]string, 0, len(candidates))
	for _, c := range candidates {
		models = append(models, c.Model)
	}
	return models
}

func (m *legacyContextManager) summarizeBatch(
	ctx context.Context,
	agent *AgentInstance,
	batch []providers.Message,
	existingSummary string,
) (string, error) {
	const (
		llmMaxRetries             = 3
		fallbackMinContentLength  = 200
		fallbackMaxContentPercent = 10
	)

	var sb strings.Builder
	sb.WriteString(summaryInstructions())
	if existingSummary != "" {
		sb.WriteString("\nEXISTING SUMMARY (fold it in; it covers what came before this segment):\n")
		sb.WriteString(existingSummary)
		sb.WriteString("\n")
	}
	sb.WriteString("\nCONVERSATION:\n")
	for _, msg := range batch {
		fmt.Fprintf(&sb, "%s: %s\n", msg.Role, msg.Content)
	}
	prompt := sb.String()

	response, err := m.retryLLMCall(ctx, agent, prompt, llmMaxRetries)
	if err == nil && response.Content != "" {
		return strings.TrimSpace(response.Content), nil
	}
	errText := "empty response"
	if err != nil {
		errText = err.Error()
	}
	logger.WarnCF("agent", "Summarization failed; using truncated-transcript fallback summary", map[string]any{
		"agent_id": agent.ID,
		"messages": len(batch),
		"error":    errText,
	})

	var fallback strings.Builder
	fallback.WriteString("Conversation summary: ")
	for i, msg := range batch {
		if i > 0 {
			fallback.WriteString(" | ")
		}
		content := strings.TrimSpace(msg.Content)
		runes := []rune(content)
		if len(runes) == 0 {
			fallback.WriteString(fmt.Sprintf("%s: ", msg.Role))
			continue
		}

		keepLength := len(runes) * fallbackMaxContentPercent / 100
		if keepLength < fallbackMinContentLength {
			keepLength = fallbackMinContentLength
		}
		if keepLength > len(runes) {
			keepLength = len(runes)
		}

		content = string(runes[:keepLength])
		if keepLength < len(runes) {
			content += "..."
		}
		fallback.WriteString(fmt.Sprintf("%s: %s", msg.Role, content))
	}
	return fallback.String(), nil
}

func (m *legacyContextManager) estimateTokens(messages []providers.Message) int {
	total := 0
	for _, msg := range messages {
		total += EstimateMessageTokens(msg)
	}
	return total
}

// summarizerThinking caps reasoning for summary calls at low effort. A
// summary is a compression of text already in hand, and it runs after nearly
// every turn: at the main turn's "high" effort it costs ~$0.08-0.10 a run,
// which would eat most of the prompt-cache saving. A model with thinking off
// or not configured is left as is (no thinking is switched on).
func summarizerThinking(settings thinkingSettings) thinkingSettings {
	if settings.configured && settings.level != ThinkingOff {
		settings.level = ThinkingLow
	}
	return settings
}
