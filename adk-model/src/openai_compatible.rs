//! Shared OpenAI-compatible provider implementation.

use crate::openai::convert;
use crate::retry::{RetryConfig, execute_with_retry, is_retryable_model_error};
use adk_core::{
    AdkError, Content, ErrorCategory, ErrorComponent, FinishReason, GenericSchemaAdapter, Llm,
    LlmRequest, LlmResponse, LlmResponseStream, Part, SchemaAdapter, SchemaCache, UsageMetadata,
};
use async_openai::types::chat::{
    CreateChatCompletionRequestArgs, ReasoningEffort, ResponseFormat, ResponseFormatJsonSchema,
};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for OpenAI-compatible providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleConfig {
    /// Provider display name used in error messages.
    pub provider_name: String,
    /// API key.
    pub api_key: String,
    /// Model name.
    pub model: String,
    /// Optional API base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Optional organization ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organization_id: Option<String>,
    /// Optional project ID for providers that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Optional reasoning effort for OpenAI reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Whether to allow the model to call multiple tools in a single turn.
    pub parallel_tool_calls: bool,
    /// Extra HTTP headers stamped on every request (gateway identity etc.),
    /// as ordered `(name, value)` pairs. Parsed once at client construction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_headers: Vec<(String, String)>,
}

impl OpenAICompatibleConfig {
    /// Create config for an OpenAI-compatible provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_name: "openai-compatible".to_string(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: None,
            organization_id: None,
            project_id: None,
            reasoning_effort: None,
            parallel_tool_calls: true,
            extra_headers: Vec::new(),
        }
    }

    /// Set provider display name used in errors.
    pub fn with_provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    /// Set a custom API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set organization ID.
    pub fn with_organization(mut self, organization_id: impl Into<String>) -> Self {
        self.organization_id = Some(organization_id.into());
        self
    }

    /// Set project ID.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set extra HTTP headers stamped on every request (gateway identity
    /// etc.). Invalid names/values surface as an error at client
    /// construction.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Set reasoning effort for reasoning models.
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set whether parallel tool calls are allowed.
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    // ── Provider presets ─────────────────────────────────────────

    /// Fireworks AI preset.
    ///
    /// Default model: `accounts/fireworks/models/llama-v3p1-8b-instruct`
    pub fn fireworks(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("fireworks")
            .with_base_url("https://api.fireworks.ai/inference/v1")
    }

    /// Together AI preset.
    ///
    /// Default model: `meta-llama/Llama-3.3-70B-Instruct-Turbo`
    pub fn together(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("together")
            .with_base_url("https://api.together.xyz/v1")
    }

    /// Mistral AI preset.
    ///
    /// Default model: `mistral-small-latest`
    pub fn mistral(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("mistral")
            .with_base_url("https://api.mistral.ai/v1")
    }

    /// Perplexity preset.
    ///
    /// Default model: `sonar`
    pub fn perplexity(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("perplexity")
            .with_base_url("https://api.perplexity.ai")
    }

    /// Cerebras preset.
    ///
    /// Default model: `llama-3.3-70b`
    pub fn cerebras(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("cerebras")
            .with_base_url("https://api.cerebras.ai/v1")
    }

    /// SambaNova preset.
    ///
    /// Default model: `Meta-Llama-3.3-70B-Instruct`
    pub fn sambanova(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("sambanova")
            .with_base_url("https://api.sambanova.ai/v1")
    }

    /// xAI (Grok) preset.
    ///
    /// Default model: `grok-3-mini`
    pub fn xai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model).with_provider_name("xai").with_base_url("https://api.x.ai/v1")
    }

    /// Google Gemini (OpenAI-compatible) preset.
    ///
    /// Targets Gemini's OpenAI-compatibility endpoint, letting you use a Gemini
    /// API key and a Gemini model (e.g. `gemini-3.5-flash`) through the OpenAI
    /// Chat Completions wire format. Use a `GEMINI_API_KEY` for the `api_key`.
    ///
    /// For native Gemini features (thinking levels, server-side tools, the
    /// Interactions API), prefer [`GeminiModel`](crate::gemini::GeminiModel).
    /// This preset is for callers who want a single OpenAI-compatible code path
    /// across providers.
    ///
    /// Default model suggestion: `gemini-3.5-flash`.
    pub fn gemini(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("gemini")
            .with_base_url("https://generativelanguage.googleapis.com/v1beta/openai")
    }

    /// MiniMax preset.
    ///
    /// Default model: `minimax-m2.7`
    pub fn minimax(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("minimax")
            .with_base_url("https://api.minimax.chat/v1")
    }

    /// ByteDance Doubao (Volcano Engine Ark) preset.
    ///
    /// Default model: `doubao-1-5-pro-256k`
    pub fn bytedance(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("bytedance")
            .with_base_url("https://ark.cn-beijing.volces.com/api/v3")
    }

    /// Zhipu AI (GLM) preset.
    ///
    /// Default model: `glm-5.1`
    pub fn zhipu(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("zhipu")
            .with_base_url("https://open.bigmodel.cn/api/paas/v4")
    }

    /// Baidu ERNIE (Qianfan) preset via OpenAI-compatible endpoint.
    ///
    /// Default model: `ernie-5`
    ///
    /// Note: Uses the Qianfan OpenAI-compatible endpoint. For the native
    /// Qianfan API with OAuth2 token exchange, use a dedicated client.
    pub fn baidu(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("baidu")
            .with_base_url("https://qianfan.baidubce.com/v2")
    }

    /// Cohere preset via OpenAI-compatible endpoint.
    ///
    /// Default model: `command-a-plus-05-2026`
    ///
    /// Note: For full Cohere features (citations, connectors, RAG), use
    /// the native Cohere API. This preset provides basic chat completions.
    pub fn cohere(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(api_key, model)
            .with_provider_name("cohere")
            .with_base_url("https://api.cohere.com/compatibility/v1")
    }
}

/// Shared OpenAI-compatible client implementation.
pub struct OpenAICompatible {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    provider_name: String,
    retry_config: RetryConfig,
    reasoning_effort: Option<ReasoningEffort>,
    organization_id: Option<String>,
    parallel_tool_calls: bool,
    extra_headers: reqwest::header::HeaderMap,
}

impl OpenAICompatible {
    /// Create a new OpenAI-compatible client.
    pub fn new(config: OpenAICompatibleConfig) -> Result<Self, AdkError> {
        let base_url = config.base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let extra_headers = crate::custom_headers::parse_extra_headers(&config.extra_headers)?;

        Ok(Self {
            http: reqwest::Client::new(),
            api_key: config.api_key,
            base_url,
            model: config.model,
            provider_name: config.provider_name,
            retry_config: RetryConfig::default(),
            reasoning_effort: config.reasoning_effort,
            organization_id: config.organization_id,
            parallel_tool_calls: config.parallel_tool_calls,
            extra_headers,
        })
    }

    /// Set the retry configuration (builder pattern).
    #[must_use]
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Set the retry configuration (mutable reference).
    pub fn set_retry_config(&mut self, retry_config: RetryConfig) {
        self.retry_config = retry_config;
    }

    /// Returns the current retry configuration.
    pub fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }
}

/// Build the serialized JSON request body from an `LlmRequest`.
///
/// This is shared between the streaming and non-streaming paths so that
/// request parameter construction is identical regardless of mode.
/// Also used by `AzureOpenAIClient` for consistent request building.
pub(crate) fn build_request_json(
    model: &str,
    request: &LlmRequest,
    reasoning_effort: &Option<ReasoningEffort>,
    parallel_tool_calls: bool,
    adapter: &dyn SchemaAdapter,
    cache: &SchemaCache,
) -> Result<serde_json::Value, AdkError> {
    let messages: Vec<_> = request.contents.iter().map(convert::content_to_message).collect();

    let mut request_builder = CreateChatCompletionRequestArgs::default();
    request_builder.model(model).messages(messages);

    if !request.tools.is_empty() {
        let tools = convert::convert_tools(&request.tools, adapter, cache);
        request_builder.tools(tools);
        // OpenAI defaults parallel_tool_calls to true.
        request_builder.parallel_tool_calls(parallel_tool_calls);
    }

    if let Some(effort) = reasoning_effort {
        request_builder.reasoning_effort(effort.clone());
    }

    if let Some(config) = &request.config {
        if let Some(temp) = config.temperature {
            request_builder.temperature(temp);
        }
        if let Some(top_p) = config.top_p {
            request_builder.top_p(top_p);
        }
        if let Some(max_tokens) = config.max_output_tokens {
            request_builder.max_completion_tokens(max_tokens as u32);
        }

        if let Some(schema) = &config.response_schema {
            let mut schema_with_strict = schema.clone();
            if let Some(obj) = schema_with_strict.as_object_mut() {
                obj.insert("additionalProperties".to_string(), serde_json::json!(false));
            }
            let json_schema = ResponseFormatJsonSchema {
                name: request.model.replace(['-', '.', '/'], "_"),
                description: None,
                schema: schema_with_strict,
                strict: Some(true),
            };
            request_builder.response_format(ResponseFormat::JsonSchema { json_schema });
        }
    }

    let openai_request = request_builder
        .build()
        .map_err(|e| AdkError::model(format!("failed to build request: {e}")))?;

    let mut body = serde_json::to_value(&openai_request)
        .map_err(|e| AdkError::model(format!("failed to serialize request: {e}")))?;

    // Merge provider-specific extensions from config.extensions["openai"] into
    // the request body.  This allows users to pass provider-specific fields
    // that the typed builder doesn't cover (e.g. provider-specific parameters
    // for OpenAI-compatible APIs like DeepSeek, Together, etc.).
    if let Some(config) = &request.config
        && let Some(openai_ext) = config.extensions.get("openai")
        && let (Some(body_obj), Some(ext_obj)) = (body.as_object_mut(), openai_ext.as_object())
    {
        for (key, value) in ext_obj {
            body_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(body)
}

/// Send an HTTP POST and handle error status codes.
///
/// Returns the raw `reqwest::Response` on success so the caller can decide
/// whether to parse it as JSON (non-streaming) or consume it as an SSE byte
/// stream (streaming).
async fn send_request(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    organization_id: &Option<String>,
    extra_headers: &reqwest::header::HeaderMap,
    body: &serde_json::Value,
    provider_name: &str,
) -> Result<reqwest::Response, AdkError> {
    let mut http_req = http.post(url).bearer_auth(api_key).json(body);

    if let Some(org_id) = organization_id {
        http_req = http_req.header("OpenAI-Organization", org_id);
    }
    http_req = http_req.headers(extra_headers.clone());

    let http_resp = http_req.send().await.map_err(|e| {
        AdkError::new(
            ErrorComponent::Model,
            ErrorCategory::Unavailable,
            "model.openai_compat.request",
            format!("{provider_name} request error: {e}"),
        )
        .with_provider(provider_name)
    })?;

    if !http_resp.status().is_success() {
        let status = http_resp.status();
        let status_code = status.as_u16();
        let body = http_resp.text().await.unwrap_or_default();
        let category = match status_code {
            401 => ErrorCategory::Unauthorized,
            403 => ErrorCategory::Forbidden,
            404 => ErrorCategory::NotFound,
            408 => ErrorCategory::Timeout,
            429 => ErrorCategory::RateLimited,
            503 | 529 => ErrorCategory::Unavailable,
            _ if status_code >= 500 => ErrorCategory::Internal,
            _ => ErrorCategory::InvalidInput,
        };
        return Err(AdkError::new(
            ErrorComponent::Model,
            category,
            "model.openai_compat.api_error",
            format!("{provider_name} API error (HTTP {status}): {body}"),
        )
        .with_upstream_status(status_code)
        .with_provider(provider_name));
    }

    Ok(http_resp)
}

/// Parse a finish_reason string into an ADK `FinishReason`.
fn parse_finish_reason(fr: &str) -> FinishReason {
    match fr {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::MaxTokens,
        "tool_calls" => FinishReason::Stop,
        "content_filter" => FinishReason::Safety,
        "function_call" => FinishReason::Stop,
        _ => FinishReason::Stop,
    }
}

/// Parse usage metadata from a raw SSE chunk JSON value.
///
/// Valid on ANY chunk of the stream, not just the `finish_reason` one:
/// providers split usage across the wire differently — DeepSeek-style attaches
/// it to the `finish_reason` chunk, strict OpenAI wire (after
/// `stream_options.include_usage`) delivers it on a trailing `choices: []`
/// chunk, and gateways may do either. Field fallbacks cover the dialects seen
/// in the wild (aligned with the anycms-llm billing extractor):
/// `prompt_tokens`→`input_tokens`, `completion_tokens`→`output_tokens`;
/// cache read prefers DeepSeek's `prompt_cache_hit_tokens`, then
/// `prompt_tokens_details.cached_tokens` (OpenAI chat), then
/// `input_tokens_details.cached_tokens` (responses style); DeepSeek-style
/// `prompt_cache_miss_tokens` maps to cache-creation.
///
/// Returns `None` when the usage is absent, `null` (every OpenAI choice chunk
/// carries `"usage": null`), or all-zero — placeholder shells must not
/// populate the holder.
fn parse_usage_from_chunk(chunk: &serde_json::Value) -> Option<UsageMetadata> {
    let u = chunk.get("usage")?;
    if !u.is_object() {
        return None;
    }
    // prompt/completion names → responses-style input/output fallback.
    let pick = |primary: &str, fallback: &str| {
        u.get(primary)
            .and_then(|v| v.as_i64())
            .or_else(|| u.get(fallback).and_then(|v| v.as_i64()))
            .unwrap_or(0)
    };
    let prompt_tokens = pick("prompt_tokens", "input_tokens");
    let completion_tokens = pick("completion_tokens", "output_tokens");
    let total_tokens =
        u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(prompt_tokens + completion_tokens);

    let prompt_details = u.get("prompt_tokens_details");
    let input_details = u.get("input_tokens_details");
    let completion_details = u.get("completion_tokens_details");
    let details_cached = |details: Option<&serde_json::Value>| {
        details.and_then(|d| d.get("cached_tokens")).and_then(|v| v.as_i64())
    };

    let cache_read = u
        .get("prompt_cache_hit_tokens")
        .and_then(|v| v.as_i64())
        .or_else(|| details_cached(prompt_details))
        .or_else(|| details_cached(input_details));
    let cache_creation = u.get("prompt_cache_miss_tokens").and_then(|v| v.as_i64());

    let usage = UsageMetadata {
        prompt_token_count: prompt_tokens as i32,
        candidates_token_count: completion_tokens as i32,
        total_token_count: total_tokens as i32,
        cache_read_input_token_count: cache_read.map(|v| v as i32),
        cache_creation_input_token_count: cache_creation.map(|v| v as i32),
        thinking_token_count: completion_details
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        audio_input_token_count: prompt_details
            .and_then(|d| d.get("audio_tokens"))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        audio_output_token_count: completion_details
            .and_then(|d| d.get("audio_tokens"))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        ..Default::default()
    };
    (usage.total_token_count > 0).then_some(usage)
}

#[async_trait]
impl Llm for OpenAICompatible {
    fn name(&self) -> &str {
        &self.model
    }

    #[tracing::instrument(
        name = "model.generate_content",
        skip_all,
        fields(
            model.name = %self.name(),
            stream = %stream,
            request.contents_count = %request.contents.len(),
            request.tools_count = %request.tools.len()
        )
    )]
    async fn generate_content(
        &self,
        request: LlmRequest,
        stream: bool,
    ) -> Result<LlmResponseStream, AdkError> {
        let model = self.model.clone();
        let provider_name = self.provider_name.clone();
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let retry_config = self.retry_config.clone();
        let reasoning_effort = self.reasoning_effort.clone();
        let organization_id = self.organization_id.clone();
        let extra_headers = self.extra_headers.clone();

        // Normalize tool schemas at request time using the schema adapter.
        let adapter = self.schema_adapter();
        use std::sync::LazyLock;
        static SCHEMA_CACHE: LazyLock<SchemaCache> =
            LazyLock::new(|| SchemaCache::for_adapter(std::sync::Arc::new(GenericSchemaAdapter)));
        let request_body = build_request_json(
            &model,
            &request,
            &reasoning_effort,
            self.parallel_tool_calls,
            adapter,
            &SCHEMA_CACHE,
        )?;

        let usage_span = adk_telemetry::llm_generate_span(&provider_name, &model, stream);

        if stream {
            // ── Streaming path ──────────────────────────────────────
            let response_stream = try_stream! {
                // Inject streaming fields into the pre-built request body.
                let mut body = request_body.clone();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("stream".to_string(), serde_json::json!(true));
                    obj.insert(
                        "stream_options".to_string(),
                        serde_json::json!({"include_usage": true}),
                    );
                }

                let url = format!("{base_url}/chat/completions");

                // Retry covers only the initial HTTP request, not stream consumption.
                let response = execute_with_retry(&retry_config, is_retryable_model_error, || {
                    let http = http.clone();
                    let url = url.clone();
                    let api_key = api_key.clone();
                    let organization_id = organization_id.clone();
                    let extra_headers = extra_headers.clone();
                    let body = body.clone();
                    let provider_name = provider_name.clone();
                    async move {
                        send_request(&http, &url, &api_key, &organization_id, &extra_headers, &body, &provider_name).await
                    }
                })
                .await?;

                // Process SSE byte stream (following DeepSeekClient pattern).
                let mut byte_stream = response.bytes_stream();
                let mut buffer = String::new();
                let mut tool_call_accumulators: HashMap<u32, (String, String, String)> =
                    HashMap::new();
                let mut text_tool_buffer = crate::tool_call_parser::ToolCallBuffer::new();
                // Content text streamed as partial `Part::Text` deltas. The
                // tool-call final below restates it as ONE settled Text part
                // so the final event honors the "partial=false carries the
                // complete content" contract — without it, narration streamed
                // before tool calls existed only as partials, never landing in
                // the settled event (session history, persistence, and every
                // frontend's settled-message view all lost it).
                let mut accumulated_text = String::new();
                // Usage seen on any chunk of this response. Strict OpenAI wire
                // delivers it on a trailing `choices: []` chunk AFTER the
                // `finish_reason` chunk (via `stream_options.include_usage`),
                // so the final response cannot carry it until it arrives —
                // hence the holder below.
                let mut pending_usage: Option<UsageMetadata> = None;
                // The `finish_reason` response, withheld until its usage
                // resolves: flushed when a later usage chunk lands, at
                // `[DONE]`, or at stream end. (Mirrors the OpenRouter
                // adapter's hold-back; DeepSeek-style usage riding the finish
                // chunk itself is taken eagerly below.)
                let mut held_final: Option<LlmResponse> = None;

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result.map_err(|e| {
                        AdkError::model(format!("stream read error: {e}"))
                    })?;

                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    // Process complete SSE lines. Lines are cut at a consumed
                    // offset and the buffer is drained once per network read —
                    // re-slicing the remainder per line copied the whole
                    // buffer for every line (quadratic per read).
                    let mut consumed = 0usize;
                    while let Some(line_end) = buffer[consumed..].find('\n') {
                        let line = buffer[consumed..consumed + line_end].trim().to_string();
                        consumed += line_end + 1;

                        if line.is_empty() {
                            continue;
                        }
                        if line == "data: [DONE]" {
                            // End of stream: any held finish goes out with
                            // whatever usage ever arrived (or none).
                            if let Some(mut final_response) = held_final.take() {
                                final_response.usage_metadata =
                                    final_response.usage_metadata.or(pending_usage.take());
                                yield final_response;
                            }
                            continue;
                        }

                        if let Some(data) = line.strip_prefix("data: ") {
                            let chunk_json: serde_json::Value = match serde_json::from_str(data) {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!(
                                        "failed to parse SSE chunk: {e} - {data}"
                                    );
                                    continue;
                                }
                            };

                            // Fold usage BEFORE the choices guard: the
                            // trailing usage chunk carries `"choices": []`
                            // and would be skipped below. Counts are
                            // monotonic across the stream (every usage frame
                            // reports the whole turn), so keep the larger.
                            if let Some(usage) = parse_usage_from_chunk(&chunk_json) {
                                pending_usage = match pending_usage {
                                    Some(prev) if prev.total_token_count >= usage.total_token_count => {
                                        Some(prev)
                                    }
                                    _ => Some(usage),
                                };
                                // A held finish that was waiting on exactly
                                // this usage can go out now.
                                if let Some(mut final_response) = held_final.take() {
                                    final_response.usage_metadata =
                                        final_response.usage_metadata.or(pending_usage.take());
                                    yield final_response;
                                }
                            }

                            let choice = match chunk_json.get("choices").and_then(|c| c.get(0)) {
                                Some(c) => c,
                                None => continue,
                            };
                            let delta = match choice.get("delta") {
                                Some(d) => d,
                                None => continue,
                            };

                            let finish_reason_str = choice
                                .get("finish_reason")
                                .and_then(|v| v.as_str())
                                .map(String::from);

                            // Accumulate tool calls by index.
                            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                for tc in tool_calls {
                                    let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    let entry = tool_call_accumulators
                                        .entry(index)
                                        .or_insert_with(|| {
                                            let call_id = tc
                                                .get("id")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("")
                                                .to_string();
                                            (call_id, String::new(), String::new())
                                        });

                                    if let Some(id) = tc.get("id").and_then(|v| v.as_str())
                                        && !id.is_empty() {
                                            entry.0 = id.to_string();
                                        }

                                    if let Some(func) = tc.get("function") {
                                        if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                            entry.1 = name.to_string();
                                        }
                                        if let Some(args_chunk) =
                                            func.get("arguments").and_then(|v| v.as_str())
                                        {
                                            entry.2.push_str(args_chunk);
                                        }
                                    }
                                }
                            }

                            // Check for finish_reason → build the final response
                            // and hold it for its usage (see `held_final`).
                            if let Some(ref fr) = finish_reason_str {
                                let finish_reason = Some(parse_finish_reason(fr));
                                // DeepSeek-style usage rode this same chunk and
                                // was folded above; strict OpenAI wire sends
                                // none here (the trailing chunk carries it).
                                let usage_metadata = pending_usage.take();

                                // Emit accumulated tool calls if any.
                                if !tool_call_accumulators.is_empty() {
                                    let mut sorted_calls: Vec<_> =
                                        tool_call_accumulators.drain().collect();
                                    sorted_calls.sort_by_key(|(idx, _)| *idx);

                                    // The finish chunk's own delta can still
                                    // carry a content tail (it `continue`s past
                                    // the partial-emission path below) — fold it
                                    // in so it is not dropped.
                                    if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                                        && !text.is_empty() {
                                        accumulated_text.push_str(text);
                                    }

                                    let mut parts: Vec<Part> = Vec::with_capacity(
                                        sorted_calls.len() + usize::from(!accumulated_text.is_empty()),
                                    );
                                    // Wire order is content before tool_calls:
                                    // restate the streamed text as ONE settled
                                    // Text part ahead of the calls. The agent's
                                    // `collapse_duplicated_snapshot` recognizes
                                    // the exact restate (deltas + snapshot with
                                    // prefix == snapshot) and keeps one copy in
                                    // the conversation history.
                                    if !accumulated_text.is_empty() {
                                        parts.push(Part::Text {
                                            text: std::mem::take(&mut accumulated_text),
                                        });
                                    }
                                    parts.extend(sorted_calls
                                        .into_iter()
                                        .map(|(_, (id, name, args_str))| {
                                            let args: serde_json::Value =
                                                serde_json::from_str(&args_str)
                                                    .unwrap_or(serde_json::json!({}));
                                            Part::FunctionCall {
                                                name,
                                                args,
                                                id: Some(id),
                                                thought_signature: None,
                                            }
                                        }));

                                    held_final = Some(LlmResponse {
                                        content: Some(Content {
                                            role: "model".to_string(),
                                            parts,
                                        }),
                                        usage_metadata,
                                        finish_reason,
                                        citation_metadata: None,
                                        partial: false,
                                        // Tool-call turns are not complete — tool
                                        // results must still be processed (issue #401).
                                        turn_complete: false,
                                        interrupted: false,
                                        error_code: None,
                                        error_message: None,
                                        provider_metadata: None,
                                        interaction_id: None,
                                    });
                                    continue;
                                }

                                // Final response without tool calls.
                                let mut parts = Vec::new();
                                if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                                    && !text.is_empty() {
                                        parts.push(Part::Text { text: text.to_string() });
                                    }

                                held_final = Some(LlmResponse {
                                    content: if parts.is_empty() { None } else {
                                        Some(Content {
                                            role: "model".to_string(),
                                            parts,
                                        })
                                    },
                                    usage_metadata,
                                    finish_reason,
                                    citation_metadata: None,
                                    partial: false,
                                    turn_complete: true,
                                    interrupted: false,
                                    error_code: None,
                                    error_message: None,
                                    provider_metadata: None,
                                    interaction_id: None,
                                });
                                continue;
                            }

                            // Emit partial reasoning_content as Part::Thinking.
                            // Fallback to "reasoning" field for OpenRouter, Kilo Gateway, SambaNova, Cerebras, Groq
                            let reasoning = delta.get("reasoning_content")
                                .or_else(|| delta.get("reasoning"))
                                .and_then(|v| v.as_str());
                            if let Some(reasoning) = reasoning
                                && !reasoning.is_empty() {
                                    yield LlmResponse {
                                        content: Some(Content {
                                            role: "model".to_string(),
                                            parts: vec![Part::Thinking {
                                                thinking: reasoning.to_string(),
                                                signature: None,
                                            }],
                                        }),
                                        usage_metadata: None,
                                        finish_reason: None,
                                        citation_metadata: None,
                                        partial: true,
                                        turn_complete: false,
                                        interrupted: false,
                                        error_code: None,
                                        error_message: None,
                                        provider_metadata: None,
                                        interaction_id: None,
                                    };
                                }

                            // Emit partial text content via tool call buffer.
                            // The buffer detects <tool_call> tags split across chunks
                            // and converts them to Part::FunctionCall.
                            if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                                && !text.is_empty() {
                                    match text_tool_buffer.push(text) {
                                        crate::tool_call_parser::BufferAction::Emit(parts) => {
                                            for part in parts {
                                                let is_tool = matches!(part, Part::FunctionCall { .. });
                                                if let Part::Text { text } = &part {
                                                    accumulated_text.push_str(text);
                                                }
                                                yield LlmResponse {
                                                    content: Some(Content {
                                                        role: "model".to_string(),
                                                        parts: vec![part],
                                                    }),
                                                    usage_metadata: None,
                                                    finish_reason: None,
                                                    citation_metadata: None,
                                                    partial: !is_tool,
                                                    turn_complete: false,
                                                    interrupted: false,
                                                    error_code: None,
                                                    error_message: None,
                                                    provider_metadata: None,
                                                    interaction_id: None,
                                                };
                                            }
                                        }
                                        crate::tool_call_parser::BufferAction::Buffering => {
                                            // Still accumulating a potential tool call
                                        }
                                    }
                                }
                        }
                    }
                    buffer.drain(..consumed);
                }

                // Stream ended without `[DONE]` (some servers just close):
                // a still-held finish goes out with whatever usage arrived.
                if let Some(mut final_response) = held_final.take() {
                    final_response.usage_metadata =
                        final_response.usage_metadata.or(pending_usage.take());
                    yield final_response;
                }

                // Flush any remaining buffered content from the tool call buffer
                for part in text_tool_buffer.flush() {
                    let is_tool = matches!(part, Part::FunctionCall { .. });
                    yield LlmResponse {
                        content: Some(Content {
                            role: "model".to_string(),
                            parts: vec![part],
                        }),
                        usage_metadata: None,
                        finish_reason: if is_tool { Some(adk_core::FinishReason::Stop) } else { None },
                        citation_metadata: None,
                        partial: !is_tool,
                        turn_complete: is_tool,
                        interrupted: false,
                        error_code: None,
                        error_message: None,
                        provider_metadata: None,
                        interaction_id: None,
                    };
                }
            };

            Ok(crate::usage_tracking::with_usage_tracking(Box::pin(response_stream), usage_span))
        } else {
            // ── Non-streaming path (preserved identically) ──────────
            let response_stream = try_stream! {
                let response = execute_with_retry(&retry_config, is_retryable_model_error, || {
                    let model = model.clone();
                    let provider_name = provider_name.clone();
                    let http = http.clone();
                    let api_key = api_key.clone();
                    let base_url = base_url.clone();
                    let body = request_body.clone();
                    let organization_id = organization_id.clone();
                    let extra_headers = extra_headers.clone();
                    async move {
                        let url = format!("{base_url}/chat/completions");
                        let http_resp =
                            send_request(&http, &url, &api_key, &organization_id, &extra_headers, &body, &provider_name)
                                .await?;

                        let raw_json: serde_json::Value = http_resp.json().await.map_err(|e| {
                            AdkError::new(
                                ErrorComponent::Model,
                                ErrorCategory::Internal,
                                "model.openai_compat.parse",
                                format!("{provider_name} response parse error: {e}"),
                            )
                            .with_provider(&provider_name)
                        })?;

                        tracing::debug!(
                            provider = %provider_name,
                            model = %model,
                            has_reasoning = raw_json
                                .pointer("/choices/0/message/reasoning_content")
                                .is_some(),
                            "openai chat completion response"
                        );

                        Ok(raw_json)
                    }
                })
                .await?;

                let adk_response = convert::from_raw_openai_response(&response);
                yield adk_response;
            };

            Ok(crate::usage_tracking::with_usage_tracking(Box::pin(response_stream), usage_span))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_headers_parse_at_client_construction() {
        let config = OpenAICompatibleConfig::new("test-key", "test-model")
            .with_extra_headers(vec![("X-Agent-Id".to_string(), "agent-7".to_string())]);
        let client = OpenAICompatible::new(config).expect("client creation failed");
        assert_eq!(client.extra_headers.get("x-agent-id").unwrap(), "agent-7");
    }

    #[test]
    fn invalid_extra_header_name_fails_construction() {
        let config = OpenAICompatibleConfig::new("test-key", "test-model")
            .with_extra_headers(vec![("Bad Name".to_string(), "v".to_string())]);
        assert!(OpenAICompatible::new(config).is_err());
    }

    #[test]
    fn test_parallel_tool_calls_config() {
        let config =
            OpenAICompatibleConfig::new("test-key", "test-model").with_parallel_tool_calls(false);

        assert!(!config.parallel_tool_calls, "parallel_tool_calls should be false in config");

        let client = OpenAICompatible::new(config).expect("client creation failed");
        assert!(!client.parallel_tool_calls, "parallel_tool_calls should be false in client");
    }

    #[test]
    fn test_parallel_tool_calls_default() {
        let config = OpenAICompatibleConfig::new("test-key", "test-model");

        assert!(config.parallel_tool_calls, "parallel_tool_calls should default to true");

        let client = OpenAICompatible::new(config).expect("client creation failed");
        assert!(client.parallel_tool_calls, "parallel_tool_calls should default to true in client");
    }

    #[test]
    fn gemini_preset_sets_endpoint_and_provider() {
        let config = OpenAICompatibleConfig::gemini("test-key", "gemini-3.5-flash");
        assert_eq!(config.provider_name, "gemini");
        assert_eq!(config.model, "gemini-3.5-flash");
        assert_eq!(
            config.base_url.as_deref(),
            Some("https://generativelanguage.googleapis.com/v1beta/openai")
        );
        assert_eq!(config.api_key, "test-key");
    }

    #[test]
    fn gemini_preset_supports_reasoning_effort() {
        // Gemini's OpenAI-compat layer maps reasoning_effort onto thinking levels.
        let config = OpenAICompatibleConfig::gemini("k", "gemini-3.5-flash")
            .with_reasoning_effort(ReasoningEffort::Low);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn gemini_preset_builds_client() {
        let config = OpenAICompatibleConfig::gemini("k", "gemini-3.5-flash");
        let client = OpenAICompatible::new(config).expect("client builds");
        assert_eq!(client.name(), "gemini-3.5-flash");
    }

    // ── streaming usage extraction ──────────────────────────────────────

    mod stream_usage {
        use super::*;
        use adk_core::{Content, LlmRequest};
        use futures::StreamExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        fn test_client(base_url: String) -> OpenAICompatible {
            OpenAICompatible::new(
                OpenAICompatibleConfig::new("test-key", "gpt-test")
                    .with_provider_name("openai-compat-test")
                    .with_base_url(base_url),
            )
            .expect("client should build")
        }

        fn request() -> LlmRequest {
            LlmRequest::new("gpt-test", vec![Content::new("user").with_text("hello")])
        }

        async fn drive(client: &OpenAICompatible) -> Vec<LlmResponse> {
            let mut stream =
                client.generate_content(request(), true).await.expect("generation should start");
            let mut responses = Vec::new();
            while let Some(item) = stream.next().await {
                responses.push(item.expect("chunk should succeed"));
            }
            responses
        }

        /// Strict OpenAI wire (`stream_options.include_usage`): every choice
        /// chunk carries `"usage": null` and the real counts arrive on a
        /// trailing `"choices": []` chunk AFTER `finish_reason`. The held
        /// final must pick them up — the regression this hold-back fixes.
        #[tokio::test]
        async fn usage_on_trailing_empty_choices_chunk_attaches_to_final() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}],\"usage\":null}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n\
                     data: [DONE]\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            assert_eq!(responses.len(), 2, "one partial + one final");
            assert!(responses[0].partial);
            assert!(!responses[1].partial);
            assert_eq!(responses[1].finish_reason, Some(FinishReason::Stop));
            let usage =
                responses[1].usage_metadata.as_ref().expect("final must carry the trailing usage");
            assert_eq!(usage.prompt_token_count, 10);
            assert_eq!(usage.candidates_token_count, 5);
            assert_eq!(usage.total_token_count, 15);
            assert_eq!(usage.cache_read_input_token_count, Some(4));
        }

        /// DeepSeek-style wire: usage rides the `finish_reason` chunk itself.
        /// Pre-existing behavior, preserved by the hold-back (flushed at
        /// `[DONE]`), with the DeepSeek cache dialect mapped.
        #[tokio::test]
        async fn usage_riding_finish_chunk_attaches_to_final() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10,\"prompt_cache_hit_tokens\":2,\"prompt_cache_miss_tokens\":5}}\n\n\
                     data: [DONE]\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            assert_eq!(responses.len(), 2);
            let usage =
                responses[1].usage_metadata.as_ref().expect("finish-chunk usage must survive");
            assert_eq!(usage.prompt_token_count, 7);
            assert_eq!(usage.total_token_count, 10);
            assert_eq!(usage.cache_read_input_token_count, Some(2));
            assert_eq!(usage.cache_creation_input_token_count, Some(5));
        }

        /// Tool-call finals are held too: a multi-step agent turn must credit
        /// the step that made the calls, not lose usage because the final
        /// lacks text.
        #[tokio::test]
        async fn tool_call_final_holds_for_trailing_usage() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":4,\"total_tokens\":24}}\n\n\
                     data: [DONE]\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            let final_response = responses
                .iter()
                .rev()
                .find(|r| !r.partial)
                .expect("a non-partial final must be emitted");
            assert!(!final_response.turn_complete, "tool-call turns stay open");
            assert!(final_response.content.as_ref().is_some_and(|c| c.has_function_calls()));
            assert_eq!(
                final_response.usage_metadata.as_ref().map(|u| u.total_token_count),
                Some(24)
            );
        }

        /// Narration streamed as content deltas BEFORE the tool calls must
        /// land in the settled final as one Text part ahead of the calls.
        /// The settled event is the only thing persistence (and every
        /// settled-message consumer — session history, the TUI's final
        /// replace, the web actor) keeps; before the accumulation the text
        /// existed only as partials and was silently dropped from the final.
        #[tokio::test]
        async fn streamed_text_lands_in_the_tool_call_final() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"我先看看目录。\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"再读 README。\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                     data: [DONE]\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            let final_response = responses
                .iter()
                .rev()
                .find(|r| !r.partial)
                .expect("a non-partial final must be emitted");
            let content =
                final_response.content.as_ref().expect("final must carry content");
            // Wire order: the streamed text first, then the call.
            match &content.parts[..] {
                [Part::Text { text }, Part::FunctionCall { name, .. }] => {
                    assert_eq!(text, "我先看看目录。再读 README。");
                    assert_eq!(name, "read_file");
                }
                other => panic!("expected [Text, FunctionCall], got {other:?}"),
            }
        }

        /// Servers that close the stream without `[DONE]`: the held final
        /// (and any usage) still flushes at stream end.
        #[tokio::test]
        async fn stream_end_without_done_marker_flushes_held_final() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            let final_response = responses
                .iter()
                .rev()
                .find(|r| !r.partial)
                .expect("held final must flush at stream end");
            assert_eq!(
                final_response.usage_metadata.as_ref().map(|u| u.total_token_count),
                Some(11)
            );
        }

        /// A provider that reports no usage at all keeps working: the final
        /// goes out at `[DONE]` with `usage_metadata: None`.
        #[tokio::test]
        async fn usage_free_stream_still_emits_final() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_string(
                    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
                     data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                     data: [DONE]\n\n",
                ))
                .mount(&server)
                .await;

            let responses = drive(&test_client(server.uri())).await;

            assert_eq!(responses.len(), 2);
            assert!(!responses[1].partial);
            assert!(responses[1].usage_metadata.is_none());
        }

        // ── parse_usage_from_chunk dialects ────────────────────────────

        /// OpenAI streams carry `"usage": null` on every choice chunk —
        /// those must parse to `None`, not a zeroed placeholder.
        #[test]
        fn parse_usage_rejects_null_and_zeroed_shells() {
            assert!(parse_usage_from_chunk(&serde_json::json!({})).is_none());
            assert!(
                parse_usage_from_chunk(&serde_json::json!({"choices": [], "usage": null}))
                    .is_none()
            );
            assert!(
                parse_usage_from_chunk(&serde_json::json!({"usage": {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
                }}))
                .is_none()
            );
        }

        /// Responses-style dialects name the buckets input/output and tuck
        /// cache under `input_tokens_details.cached_tokens`; total is
        /// derived when the wire omits it.
        #[test]
        fn parse_usage_falls_back_to_responses_style_fields() {
            let usage = parse_usage_from_chunk(&serde_json::json!({"usage": {
                "input_tokens": 8,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 6}
            }}))
            .expect("responses-style usage should parse");
            assert_eq!(usage.prompt_token_count, 8);
            assert_eq!(usage.candidates_token_count, 2);
            assert_eq!(usage.total_token_count, 10, "total derived as input+output");
            assert_eq!(usage.cache_read_input_token_count, Some(6));
        }

        /// The cache-read fallback order: DeepSeek's flat
        /// `prompt_cache_hit_tokens` wins over OpenAI's nested
        /// `prompt_tokens_details.cached_tokens`.
        #[test]
        fn parse_usage_prefers_deepseek_cache_fields() {
            let usage = parse_usage_from_chunk(&serde_json::json!({"usage": {
                "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
                "prompt_cache_hit_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3}
            }}))
            .expect("usage should parse");
            assert_eq!(usage.cache_read_input_token_count, Some(7));
            assert_eq!(usage.cache_creation_input_token_count, None);

            let nested = parse_usage_from_chunk(&serde_json::json!({"usage": {
                "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 3}
            }}))
            .expect("usage should parse");
            assert_eq!(nested.cache_read_input_token_count, Some(3));
        }
    }
}
