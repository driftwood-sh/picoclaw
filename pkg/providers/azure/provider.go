package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers/common"
	orc "github.com/sipeed/picoclaw/pkg/providers/openai_responses_common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

type (
	LLMResponse    = protocoltypes.LLMResponse
	Message        = protocoltypes.Message
	ToolDefinition = protocoltypes.ToolDefinition
)

const (
	defaultRequestTimeout = common.DefaultRequestTimeout
	responsesAPIPath      = "openai/v1/responses"
)

// Provider implements the LLM provider interface for Azure OpenAI endpoints.
// It handles Azure-specific authentication (Bearer token), URL construction
// (Responses API), and request/response formatting.
type Provider struct {
	apiKey        string
	apiBase       string
	responsesPath string
	httpClient    *http.Client
	userAgent     string
	tokenSource   func(ctx context.Context) (string, error)
}

// Option configures the Azure Provider.
type Option func(*Provider)

// WithRequestTimeout sets the HTTP request timeout.
func WithRequestTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		if timeout > 0 {
			p.httpClient.Timeout = timeout
		}
	}
}

// WithUserAgent sets the User-Agent header for requests.
func WithUserAgent(userAgent string) Option {
	return func(p *Provider) {
		p.userAgent = userAgent
	}
}

// WithTokenSource sets a callback that returns a bearer token per request.
// When set, it takes precedence over the static api key.
func WithTokenSource(ts func(ctx context.Context) (string, error)) Option {
	return func(p *Provider) {
		p.tokenSource = ts
	}
}

// WithResponsesAPIPath overrides the Responses API path joined onto api_base.
// Azure uses "openai/v1/responses"; direct OpenAI uses "v1/responses".
func WithResponsesAPIPath(path string) Option {
	return func(p *Provider) {
		if path != "" {
			p.responsesPath = path
		}
	}
}

// NewProvider creates a new Azure OpenAI provider.
func NewProvider(apiKey, apiBase, proxy, userAgent string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:        apiKey,
		apiBase:       strings.TrimRight(apiBase, "/"),
		responsesPath: responsesAPIPath,
		userAgent:     userAgent,
		httpClient:    common.NewHTTPClient(proxy),
	}

	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}

	return p
}

// NewProviderWithTimeout creates a new Azure OpenAI provider with a custom request timeout in seconds.
func NewProviderWithTimeout(apiKey, apiBase, proxy, userAgent string, requestTimeoutSeconds int) *Provider {
	return NewProvider(
		apiKey, apiBase, proxy, userAgent,
		WithRequestTimeout(time.Duration(requestTimeoutSeconds)*time.Second),
	)
}

// NewProviderWithTokenSource creates a new Azure OpenAI provider that obtains its
// bearer token from the supplied callback on every request. Used for Entra ID auth
// where tokens are short-lived and refreshed by the underlying credential.
func NewProviderWithTokenSource(
	apiBase, proxy, userAgent string,
	tokenSource func(ctx context.Context) (string, error),
	opts ...Option,
) *Provider {
	p := &Provider{
		apiBase:       strings.TrimRight(apiBase, "/"),
		responsesPath: responsesAPIPath,
		userAgent:     userAgent,
		httpClient:    common.NewHTTPClient(proxy),
		tokenSource:   tokenSource,
	}

	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}

	return p
}

// Chat sends a request to the Azure OpenAI Responses API endpoint.
// The model parameter is passed in the request body.
func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p.apiBase == "" {
		return nil, fmt.Errorf("Azure API base not configured")
	}

	requestURL, err := url.JoinPath(p.apiBase, p.responsesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build Azure request URL: %w", err)
	}

	input, instructions, replayed := orc.TranslateMessagesForModel(messages, model)

	requestBody := responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		},
		Store: openai.Opt(false),
	}

	if instructions != "" {
		requestBody.Instructions = openai.Opt(instructions)
	}

	if len(tools) > 0 {
		enableWebSearch, _ := options["native_search"].(bool)
		requestBody.Tools = orc.TranslateTools(tools, enableWebSearch)
		requestBody.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsAuto),
		}
	}

	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok {
		requestBody.MaxOutputTokens = openai.Opt(int64(maxTokens))
	}

	reasoningApplied := false
	if level, ok := options["thinking_level"].(string); ok {
		if effort, mapped := reasoningEffortForThinkingLevel(level); mapped {
			requestBody.Reasoning = shared.ReasoningParam{Effort: effort}
			reasoningApplied = true
			if effort != shared.ReasoningEffortNone {
				// store=false keeps nothing server-side, so ask for the
				// reasoning back (encrypted) to replay on the next call of
				// the turn. Current OpenAI models return it by default when
				// store is false; the include makes it explicit.
				requestBody.Include = []responses.ResponseIncludable{
					responses.ResponseIncludableReasoningEncryptedContent,
				}
			}
		}
	}

	// Reasoning requests reject sampling parameters: OpenAI 400s any
	// non-default temperature on reasoning models ("'temperature' is not
	// supported with this model"), and the agent loop always sends one
	// (default 0.7). Only forward temperature on non-reasoning requests.
	if temperature, ok := common.AsFloat(options["temperature"]); ok && !reasoningApplied {
		requestBody.Temperature = openai.Opt(temperature)
	}

	if cacheKey, ok := options["prompt_cache_key"].(string); ok && cacheKey != "" {
		requestBody.PromptCacheKey = openai.Opt(cacheKey)
	}

	resp, err := p.send(ctx, requestURL, requestBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusBadRequest && replayed.Replayed() {
		// A replayed reasoning item or phase the API will not take (for
		// example encrypted reasoning it can no longer verify) must not
		// fail the turn: retry once with plain messages, as before replay.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		if replayRejected(body) {
			logger.WarnCF("provider.responses", "API rejected replayed reasoning; retrying without it",
				map[string]any{
					"model":           model,
					"reasoning_items": replayed.ReasoningItems,
					"phases":          replayed.Phases,
					"error":           common.ResponsePreview(body, 200),
				})
			plain, _ := orc.TranslateMessages(messages)
			requestBody.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: plain}
			retryResp, retryErr := p.send(ctx, requestURL, requestBody)
			if retryErr != nil {
				return nil, retryErr
			}
			defer retryResp.Body.Close()
			resp = retryResp
		} else {
			resp.Body = io.NopCloser(bytes.NewReader(body))
		}
	}

	if resp.StatusCode != http.StatusOK {
		return nil, common.HandleErrorResponse(resp, p.apiBase)
	}

	out, err := orc.ParseResponseBody(resp.Body)
	if err != nil {
		return nil, err
	}
	for i := range out.ReasoningItems {
		out.ReasoningItems[i].Model = model
	}
	return out, nil
}

// maxErrorBodyBytes bounds how much of an error body is read to classify it.
const maxErrorBodyBytes = 64 * 1024

// replayRejected reports whether a 400 body blames replayed reasoning or a
// message phase, rather than the request as a whole.
func replayRejected(body []byte) bool {
	lower := strings.ToLower(string(body))
	for _, marker := range []string{"encrypted_content", "encrypted content", "reasoning", "phase"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// send marshals and posts one Responses API request.
func (p *Provider) send(
	ctx context.Context,
	requestURL string,
	requestBody responses.ResponseNewParams,
) (*http.Response, error) {
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", requestURL, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	switch {
	case p.tokenSource != nil:
		tok, tokErr := p.tokenSource(ctx)
		if tokErr != nil {
			return nil, fmt.Errorf("acquiring azure identity token: %w", tokErr)
		}
		req.Header.Set("Authorization", "Bearer "+tok)
	case p.apiKey != "":
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	if p.userAgent != "" {
		req.Header.Set("User-Agent", p.userAgent)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	return resp, nil
}

// GetDefaultModel returns an empty string as Azure deployments are user-configured.
func (p *Provider) GetDefaultModel() string {
	return ""
}

// SupportsThinking reports that this provider can apply thinking_level: the
// Responses API takes a first-class reasoning.effort parameter. Without this,
// the agent loop strips thinking_level from the options before Chat is called.
func (p *Provider) SupportsThinking() bool {
	return true
}

// reasoningEffortForThinkingLevel maps picoclaw thinking levels onto Responses
// API reasoning efforts. "adaptive" (and anything unrecognized) reports
// unmapped so the API's own default effort applies.
func reasoningEffortForThinkingLevel(level string) (shared.ReasoningEffort, bool) {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "off":
		return shared.ReasoningEffortNone, true
	case "low":
		return shared.ReasoningEffortLow, true
	case "medium":
		return shared.ReasoningEffortMedium, true
	case "high":
		return shared.ReasoningEffortHigh, true
	case "xhigh":
		return shared.ReasoningEffortXhigh, true
	default:
		return "", false
	}
}
