// Package openai_responses_common provides shared utilities for providers
// that use the OpenAI Responses API (e.g., Azure, Codex).
package openai_responses_common

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	"github.com/sipeed/picoclaw/pkg/providers/common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

// TranslateMessages converts internal Message entries to the OpenAI Responses API
// input format. System messages are extracted as instructions (returned separately),
// user/assistant/tool messages become ResponseInputItemUnionParam entries.
// Supports multipart media (images, audio). It replays no reasoning items and
// no message phase; see TranslateMessagesForModel.
func TranslateMessages(messages []protocoltypes.Message) (input responses.ResponseInputParam, instructions string) {
	input, instructions, _ = translateMessages(messages, nil)
	return input, instructions
}

// ReplayStats reports what TranslateMessagesForModel replayed beyond plain
// messages, so a caller can retry without it if the API rejects it.
type ReplayStats struct {
	ReasoningItems int
	Phases         int
}

// Replayed reports whether anything beyond plain messages was sent.
func (s ReplayStats) Replayed() bool {
	return s.ReasoningItems > 0 || s.Phases > 0
}

type replayOptions struct {
	model string
}

// TranslateMessagesForModel is TranslateMessages for a stateless
// (store=false) request to model. It also sends back what the model produced
// earlier in the turn:
//   - each assistant message's encrypted reasoning items, placed before that
//     message's output, when they came from the same model (encrypted
//     reasoning is only valid for the model that wrote it);
//   - each assistant message's phase ("commentary" / "final_answer").
//
// OpenAI recommends passing reasoning items back with function call outputs;
// without them the model starts every step of a tool loop from scratch.
func TranslateMessagesForModel(
	messages []protocoltypes.Message,
	model string,
) (input responses.ResponseInputParam, instructions string, stats ReplayStats) {
	return translateMessages(messages, &replayOptions{model: model})
}

func translateMessages(
	messages []protocoltypes.Message,
	replay *replayOptions,
) (input responses.ResponseInputParam, instructions string, stats ReplayStats) {
	input = make(responses.ResponseInputParam, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			instructions = msg.Content
		case "user":
			if msg.ToolCallID != "" {
				input = append(input, responses.ResponseInputItemUnionParam{
					OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
						CallID: msg.ToolCallID,
						Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
							OfString: openai.Opt(msg.Content),
						},
					},
				})
			} else if len(msg.Media) > 0 {
				content := BuildMultipartContent(msg.Content, msg.Media)
				input = append(input, responses.ResponseInputItemUnionParam{
					OfInputMessage: &responses.ResponseInputItemMessageParam{
						Role:    "user",
						Content: content,
					},
				})
			} else {
				input = append(input, responses.ResponseInputItemUnionParam{
					OfMessage: &responses.EasyInputMessageParam{
						Role:    responses.EasyInputMessageRoleUser,
						Content: responses.EasyInputMessageContentUnionParam{OfString: openai.Opt(msg.Content)},
					},
				})
			}
		case "assistant":
			if replay != nil {
				for _, item := range msg.ReasoningItems {
					if item.EncryptedContent == "" || item.ID == "" || item.Model != replay.model {
						continue
					}
					input = append(input, reasoningInputItem(item))
					stats.ReasoningItems++
				}
			}
			if len(msg.ToolCalls) > 0 {
				if msg.Content != "" {
					input = append(input, assistantInputMessage(msg, replay, &stats))
				}
				for _, tc := range msg.ToolCalls {
					name, args, ok := ResolveToolCall(tc)
					if !ok {
						continue
					}
					input = append(input, responses.ResponseInputItemUnionParam{
						OfFunctionCall: &responses.ResponseFunctionToolCallParam{
							CallID:    tc.ID,
							Name:      name,
							Arguments: args,
						},
					})
				}
			} else {
				input = append(input, assistantInputMessage(msg, replay, &stats))
			}
		case "tool":
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: msg.ToolCallID,
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
						OfString: openai.Opt(msg.Content),
					},
				},
			})
		}
	}

	return input, instructions, stats
}

// assistantInputMessage replays an assistant message's text, with its phase
// when replaying for a stateless request and the API had set one.
func assistantInputMessage(
	msg protocoltypes.Message,
	replay *replayOptions,
	stats *ReplayStats,
) responses.ResponseInputItemUnionParam {
	param := &responses.EasyInputMessageParam{
		Role:    responses.EasyInputMessageRoleAssistant,
		Content: responses.EasyInputMessageContentUnionParam{OfString: openai.Opt(msg.Content)},
	}
	if replay != nil && msg.Phase != "" {
		param.SetExtraFields(map[string]any{"phase": msg.Phase})
		stats.Phases++
	}
	return responses.ResponseInputItemUnionParam{OfMessage: param}
}

// reasoningInputItem converts a stored reasoning item back to an input item.
func reasoningInputItem(item protocoltypes.ReasoningItem) responses.ResponseInputItemUnionParam {
	summary := make([]responses.ResponseReasoningItemSummaryParam, 0, len(item.Summary))
	for _, text := range item.Summary {
		summary = append(summary, responses.ResponseReasoningItemSummaryParam{Text: text})
	}
	return responses.ResponseInputItemUnionParam{
		OfReasoning: &responses.ResponseReasoningItemParam{
			ID:               item.ID,
			Summary:          summary,
			EncryptedContent: openai.Opt(item.EncryptedContent),
		},
	}
}

// BuildMultipartContent constructs a ResponseInputMessageContentListParam from
// text content and media URLs (data:image/... and data:audio/... URIs).
func BuildMultipartContent(text string, media []string) responses.ResponseInputMessageContentListParam {
	parts := make(responses.ResponseInputMessageContentListParam, 0, 1+len(media))

	if text != "" {
		parts = append(parts, responses.ResponseInputContentUnionParam{
			OfInputText: &responses.ResponseInputTextParam{
				Text: text,
			},
		})
	}

	for _, mediaURL := range media {
		if strings.HasPrefix(mediaURL, "data:image/") {
			parts = append(parts, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					ImageURL: openai.Opt(mediaURL),
					Detail:   responses.ResponseInputImageDetailAuto,
				},
			})
		} else if strings.HasPrefix(mediaURL, "data:audio/") {
			if format, data, ok := common.ParseDataAudioURL(mediaURL); ok {
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileData: openai.Opt(data),
						Filename: openai.Opt("audio." + format),
					},
				})
			}
		}
	}

	return parts
}

// ResolveToolCall extracts the function name and JSON arguments string from a ToolCall.
// Returns ok=false if the tool call has no name or if arguments fail to marshal.
func ResolveToolCall(tc protocoltypes.ToolCall) (name string, arguments string, ok bool) {
	name = tc.Name
	if name == "" && tc.Function != nil {
		name = tc.Function.Name
	}
	if name == "" {
		return "", "", false
	}

	if len(tc.Arguments) > 0 {
		argsJSON, err := json.Marshal(tc.Arguments)
		if err != nil {
			return "", "", false
		}
		return name, string(argsJSON), true
	}

	if tc.Function != nil && tc.Function.Arguments != "" {
		return name, tc.Function.Arguments, true
	}

	return name, "{}", true
}

// TranslateTools converts internal ToolDefinition entries to the OpenAI Responses API
// tool format. If enableWebSearch is true, a web_search tool is appended and any
// user-defined tool named "web_search" is skipped to avoid duplicates.
func TranslateTools(tools []protocoltypes.ToolDefinition, enableWebSearch bool) []responses.ToolUnionParam {
	capHint := len(tools)
	if enableWebSearch {
		capHint++
	}
	result := make([]responses.ToolUnionParam, 0, capHint)

	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		if enableWebSearch && strings.EqualFold(t.Function.Name, "web_search") {
			continue
		}
		ft := responses.FunctionToolParam{
			Name:       t.Function.Name,
			Parameters: t.Function.Parameters,
			Strict:     openai.Opt(false),
		}
		if t.Function.Description != "" {
			ft.Description = openai.Opt(t.Function.Description)
		}
		result = append(result, responses.ToolUnionParam{OfFunction: &ft})
	}

	if enableWebSearch {
		result = append(result, responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch))
	}

	return result
}

// ParseResponseBody parses an OpenAI Responses API JSON body into an LLMResponse.
// Handles output item types: "message" (output_text + refusal), "function_call", and "reasoning".
func ParseResponseBody(body io.Reader) (*protocoltypes.LLMResponse, error) {
	var apiResp responses.Response
	if err := json.NewDecoder(body).Decode(&apiResp); err != nil {
		return nil, err
	}

	return parseResponse(&apiResp), nil
}

// ParseResponseFromStruct converts a decoded responses.Response into an LLMResponse.
// Used by providers that receive the Response struct directly (e.g., via streaming SDK).
func ParseResponseFromStruct(resp *responses.Response) *protocoltypes.LLMResponse {
	return parseResponse(resp)
}

// parseResponse is the shared implementation for extracting LLMResponse fields
// from a decoded responses.Response.
func parseResponse(apiResp *responses.Response) *protocoltypes.LLMResponse {
	var content strings.Builder
	var reasoningContent strings.Builder
	var toolCalls []protocoltypes.ToolCall
	var reasoningItems []protocoltypes.ReasoningItem
	phase := ""
	refused := false

	for _, item := range apiResp.Output {
		switch item.Type {
		case "message":
			if p := outputItemPhase(item); p != "" {
				phase = p
			}
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					content.WriteString(c.Text)
				case "refusal":
					// The Responses API reports a safety refusal as a
					// {"type":"refusal","refusal":"<text>"} content part on an
					// otherwise completed response. Keep the text visible and
					// remember the refusal for FinishReason below.
					content.WriteString(c.Refusal)
					refused = true
				}
			}
		case "function_call":
			var args map[string]any
			if err := json.Unmarshal([]byte(item.Arguments), &args); err != nil {
				args = map[string]any{"raw": item.Arguments}
			}
			toolCalls = append(toolCalls, protocoltypes.ToolCall{
				ID:        item.CallID,
				Name:      item.Name,
				Arguments: args,
			})
		case "reasoning":
			summary := make([]string, 0, len(item.Summary))
			for _, s := range item.Summary {
				reasoningContent.WriteString(s.Text)
				summary = append(summary, s.Text)
			}
			// Only an item with encrypted content can be replayed when the
			// API stores nothing (store=false); an id alone is "not found".
			if item.ID != "" && item.EncryptedContent != "" {
				reasoningItems = append(reasoningItems, protocoltypes.ReasoningItem{
					ID:               item.ID,
					EncryptedContent: item.EncryptedContent,
					Summary:          summary,
				})
			}
		}
	}

	// Tool calls win over a refusal part: the turn still has work to run.
	// A refusal with no tool calls maps to "refusal", the same value the
	// Anthropic Messages provider sets, so the agent's refusal_failover
	// applies to Responses API models too. A failed, incomplete or canceled
	// status still overrides both below.
	finishReason := "stop"
	switch {
	case len(toolCalls) > 0:
		finishReason = "tool_calls"
	case refused:
		finishReason = "refusal"
	}
	switch apiResp.Status {
	case responses.ResponseStatusIncomplete:
		finishReason = "length"
	case responses.ResponseStatusFailed:
		finishReason = "error"
	case responses.ResponseStatusCancelled:
		finishReason = "canceled"
	}

	var usage *protocoltypes.UsageInfo
	if apiResp.Usage.TotalTokens > 0 {
		// OpenAI's input_tokens INCLUDES cache reads; UsageInfo.PromptTokens
		// excludes them (Anthropic convention), so split the cached share out.
		cached := int(apiResp.Usage.InputTokensDetails.CachedTokens)
		usage = &protocoltypes.UsageInfo{
			PromptTokens:         int(apiResp.Usage.InputTokens) - cached,
			CompletionTokens:     int(apiResp.Usage.OutputTokens),
			TotalTokens:          int(apiResp.Usage.TotalTokens),
			CacheReadInputTokens: cached,
		}
	}

	return &protocoltypes.LLMResponse{
		Content:          content.String(),
		ReasoningContent: reasoningContent.String(),
		ToolCalls:        toolCalls,
		FinishReason:     finishReason,
		Usage:            usage,
		ReasoningItems:   reasoningItems,
		Phase:            phase,
	}
}

// outputItemPhase reads the "phase" of an output message item. The SDK
// version in use has no typed field for it, so it is read from the item's
// raw JSON.
func outputItemPhase(item responses.ResponseOutputItemUnion) string {
	raw := item.RawJSON()
	if raw == "" || !strings.Contains(raw, `"phase"`) {
		return ""
	}
	var withPhase struct {
		Phase string `json:"phase"`
	}
	if err := json.Unmarshal([]byte(raw), &withPhase); err != nil {
		return ""
	}
	return withPhase.Phase
}
