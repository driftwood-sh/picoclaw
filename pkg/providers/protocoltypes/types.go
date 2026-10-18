package protocoltypes

import "time"

type ToolCall struct {
	ID               string         `json:"id"`
	Type             string         `json:"type,omitempty"`
	Function         *FunctionCall  `json:"function,omitempty"`
	Name             string         `json:"-"`
	Arguments        map[string]any `json:"-"`
	ThoughtSignature string         `json:"-"` // Internal use only
	ExtraContent     *ExtraContent  `json:"extra_content,omitempty"`
}

type ExtraContent struct {
	Google                  *GoogleExtra `json:"google,omitempty"`
	ToolFeedbackExplanation string       `json:"tool_feedback_explanation,omitempty"`
}

type GoogleExtra struct {
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type FunctionCall struct {
	Name             string `json:"name"`
	Arguments        string `json:"arguments"`
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type LLMResponse struct {
	Content          string            `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall        `json:"tool_calls,omitempty"`
	FinishReason     string            `json:"finish_reason"`
	Usage            *UsageInfo        `json:"usage,omitempty"`
	Reasoning        string            `json:"reasoning"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
	// ReasoningItems are the Responses API's opaque reasoning items for this
	// response (store=false: they carry encrypted_content). In memory only,
	// like Message.ReasoningItems: the after_llm process hook receives this
	// struct as JSON on every call, and encrypted reasoning has no business
	// leaving the runtime (a hook that rewrites the response just turns
	// replay off for that one step).
	ReasoningItems []ReasoningItem `json:"-"`
	// Phase is the Responses API phase of the assistant message
	// ("commentary" before tool calls, "final_answer"), when the API sets one.
	Phase string `json:"phase,omitempty"`
}

// ReasoningItem is one opaque reasoning item from the OpenAI Responses API.
// With store=false the API keeps nothing, so the model's reasoning before a
// tool call only reaches its next call if the caller sends the item back,
// encrypted content included.
type ReasoningItem struct {
	ID               string   `json:"id"`
	EncryptedContent string   `json:"encrypted_content"`
	Summary          []string `json:"summary,omitempty"`
	// Model is the model id the request that produced the item named.
	// Encrypted reasoning is replayed only to that same model.
	Model string `json:"model,omitempty"`
}

type StreamChunk struct {
	Content          string
	ReasoningContent string
}

type ReasoningDetail struct {
	Format string `json:"format"`
	Index  int    `json:"index"`
	Type   string `json:"type"`
	Text   string `json:"text"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// Prompt-cache accounting (Anthropic prompt caching). Zero when the
	// provider does not report cache usage. CacheReadInputTokens counts
	// tokens served from cache; CacheCreationInputTokens counts tokens
	// written to cache. Neither is included in PromptTokens by Anthropic:
	// total prompt size = PromptTokens + CacheReadInputTokens + CacheCreationInputTokens.
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// CacheControl marks a content block for LLM-side prefix caching.
// Currently only "ephemeral" is supported (used by Anthropic).
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// ContentBlock represents a structured segment of a system message.
// Adapters that understand SystemParts can use these blocks to set
// per-block cache control (e.g. Anthropic's cache_control: ephemeral).
type ContentBlock struct {
	Type         string        `json:"type"` // "text"
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`

	// Prompt metadata is internal to the agent runtime. It records which
	// structured prompt segment produced this block without changing provider
	// JSON.
	PromptLayer  string `json:"-"`
	PromptSlot   string `json:"-"`
	PromptSource string `json:"-"`
}

type Attachment struct {
	Type        string `json:"type,omitempty"`
	Ref         string `json:"ref,omitempty"`
	URL         string `json:"url,omitempty"`
	Filename    string `json:"filename,omitempty"`
	ContentType string `json:"content_type,omitempty"`
}

type Message struct {
	Role             string         `json:"role"`
	Content          string         `json:"content"`
	ModelName        string         `json:"model_name,omitempty"`
	CreatedAt        *time.Time     `json:"created_at,omitempty"`
	Media            []string       `json:"media,omitempty"`
	Attachments      []Attachment   `json:"attachments,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	SystemParts      []ContentBlock `json:"system_parts,omitempty"` // structured system blocks for cache-aware adapters
	ToolCalls        []ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	// Phase is the Responses API phase of an assistant message, replayed as
	// it was received ("commentary" / "final_answer").
	Phase string `json:"phase,omitempty"`

	// ReasoningItems are the encrypted reasoning items the model produced
	// with this assistant message (Responses API, store=false). They are
	// replayed on the next calls of the same turn so the model continues
	// from its own reasoning instead of re-planning after every tool call.
	// In memory only: never written to session history or hook payloads.
	ReasoningItems []ReasoningItem `json:"-"`

	// Prompt metadata is internal to the agent runtime. It records where a
	// message or system part came from without changing provider/session JSON.
	PromptLayer  string `json:"-"`
	PromptSlot   string `json:"-"`
	PromptSource string `json:"-"`
}

type ToolDefinition struct {
	Type     string                 `json:"type"`
	Function ToolFunctionDefinition `json:"function"`

	// Prompt metadata is internal to the agent runtime. Tool definitions are
	// model-visible capability prompts even though providers send them outside
	// the system message.
	PromptLayer  string `json:"-"`
	PromptSlot   string `json:"-"`
	PromptSource string `json:"-"`
}

type ToolFunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}
