//! DeepSeek client implementation.
//!
//! Supports DeepSeek V4 models (`deepseek-v4-pro`, `deepseek-v4-flash`) and
//! legacy models (`deepseek-chat`, `deepseek-reasoner`).

use super::config::{DeepSeekConfig, ThinkingMode};
use super::convert::{
    self, ChatCompletionRequest, ChatCompletionResponse, ResponseFormat, ThinkingConfig,
};
use crate::retry::{RetryConfig, execute_with_retry, is_retryable_model_error};
use adk_core::{
    AdkError, ErrorCategory, ErrorComponent, FinishReason, GenericSchemaAdapter, Llm, LlmRequest,
    LlmResponse, LlmResponseStream, Part, RetryHint, SchemaAdapter,
};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;

/// DeepSeek client for V4 and legacy models.
///
/// # V4 Models
///
/// ```rust,ignore
/// use adk_model::deepseek::{DeepSeekClient, DeepSeekConfig, ReasoningEffort};
///
/// // V4 Pro with max reasoning
/// let pro = DeepSeekClient::new(
///     DeepSeekConfig::v4_pro("api-key")
///         .with_reasoning_effort(ReasoningEffort::Max)
/// )?;
///
/// // V4 Flash (fast; the API's default thinking is enabled — pass
/// // `with_thinking_mode(ThinkingMode::Disabled)` to turn it off)
/// let flash = DeepSeekClient::v4_flash("api-key")?;
/// ```
///
/// # Legacy Models
///
/// ```rust,ignore
/// // Still works — backward compatible
/// let chat = DeepSeekClient::chat("api-key")?;
/// let reasoner = DeepSeekClient::reasoner("api-key")?;
/// ```
pub struct DeepSeekClient {
    client: Client,
    config: DeepSeekConfig,
    retry_config: RetryConfig,
}

impl DeepSeekClient {
    /// Create a new DeepSeek client.
    pub fn new(config: DeepSeekConfig) -> Result<Self, AdkError> {
        let client = Client::builder()
            .build()
            .map_err(|e| AdkError::model(format!("failed to create HTTP client: {e}")))?;

        Ok(Self { client, config, retry_config: RetryConfig::default() })
    }

    /// Create a client for `deepseek-v4-pro` (strongest reasoning, thinking enabled).
    pub fn v4_pro(api_key: impl Into<String>) -> Result<Self, AdkError> {
        Self::new(DeepSeekConfig::v4_pro(api_key))
    }

    /// Create a client for `deepseek-v4-flash` (fast, cost-efficient).
    pub fn v4_flash(api_key: impl Into<String>) -> Result<Self, AdkError> {
        Self::new(DeepSeekConfig::v4_flash(api_key))
    }

    /// Create a client for `deepseek-chat` model (legacy).
    pub fn chat(api_key: impl Into<String>) -> Result<Self, AdkError> {
        Self::new(DeepSeekConfig::chat(api_key))
    }

    /// Create a client for `deepseek-reasoner` model with thinking enabled (legacy).
    pub fn reasoner(api_key: impl Into<String>) -> Result<Self, AdkError> {
        Self::new(DeepSeekConfig::reasoner(api_key))
    }

    /// Set retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Set retry configuration (mutable).
    pub fn set_retry_config(&mut self, retry_config: RetryConfig) {
        self.retry_config = retry_config;
    }

    /// Get the current retry configuration.
    pub fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }

    /// Build the API URL for chat completions.
    fn api_url(&self) -> String {
        let base = self.config.effective_base_url();
        format!("{}/chat/completions", base.trim_end_matches('/'))
    }

    /// Build a chat completion request from an LLM request.
    fn build_request(&self, request: &LlmRequest, stream: bool) -> ChatCompletionRequest {
        let mut messages: Vec<_> =
            request.contents.iter().map(convert::content_to_message).collect();

        // DeepSeek's structured-output contract is JSON Output: `response_format`
        // accepts only `{"type": "json_object"}` — there is no `json_schema` mode —
        // and the API requires the word "json" to appear in the system or user
        // prompt, otherwise it can return empty content.
        // https://api-docs.deepseek.com/guides/json_mode
        let response_format = match request.config.as_ref().and_then(|c| c.response_schema.as_ref())
        {
            Some(schema) => {
                if !mentions_json(&messages) {
                    messages.insert(
                        0,
                        convert::system_message(format!(
                            "Respond with a single json object matching this schema: {schema}"
                        )),
                    );
                }
                Some(ResponseFormat { format_type: "json_object".to_string() })
            }
            None => None,
        };

        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(convert::convert_tools(&request.tools, self.config.strict_tools))
        };

        // Get generation config
        let temperature = request.config.as_ref().and_then(|c| c.temperature);
        let top_p = request.config.as_ref().and_then(|c| c.top_p);
        let max_tokens = request
            .config
            .as_ref()
            .and_then(|c| c.max_output_tokens)
            .map(|t| t as u32)
            .or(self.config.max_tokens);

        // Build thinking config from the new ThinkingMode or legacy bool
        let thinking = match self.config.thinking {
            Some(ThinkingMode::Enabled) => Some(ThinkingConfig::enabled()),
            Some(ThinkingMode::Disabled) => Some(ThinkingConfig::disabled()),
            None => {
                if self.config.thinking_enabled {
                    Some(ThinkingConfig::enabled())
                } else {
                    None
                }
            }
        };

        // Reasoning effort
        let reasoning_effort = self.config.reasoning_effort.map(|e| e.to_string());

        ChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            temperature,
            top_p,
            max_tokens,
            stream: Some(stream),
            tools,
            response_format,
            thinking,
            reasoning_effort,
            stop: None,
        }
    }
}

/// Whether the conversation already satisfies DeepSeek's requirement that the word
/// "json" appear in the prompt when JSON Output is enabled.
fn mentions_json(messages: &[convert::Message]) -> bool {
    messages.iter().any(|message| {
        message.content.as_deref().is_some_and(|text| text.to_lowercase().contains("json"))
    })
}

#[async_trait]
impl Llm for DeepSeekClient {
    fn name(&self) -> &str {
        &self.config.model
    }

    fn schema_adapter(&self) -> &dyn SchemaAdapter {
        // DeepSeek uses the OpenAI-compatible API, so it uses the same transforms
        // as OpenAiSchemaAdapter (which is functionally identical to GenericSchemaAdapter).
        static ADAPTER: GenericSchemaAdapter = GenericSchemaAdapter;
        &ADAPTER
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
        let usage_span = adk_telemetry::llm_generate_span("deepseek", &self.config.model, stream);
        let api_url = self.api_url();
        let api_key = self.config.api_key.clone();
        let chat_request = self.build_request(&request, stream);
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();

        let response_stream = try_stream! {
            let response = execute_with_retry(&retry_config, is_retryable_model_error, || {
                let client = client.clone();
                let api_url = api_url.clone();
                let api_key = api_key.clone();
                let chat_request = chat_request.clone();
                async move {
                    let response = client
                        .post(&api_url)
                        .header("Authorization", format!("Bearer {api_key}"))
                        .header("Content-Type", "application/json")
                        .json(&chat_request)
                        .send()
                        .await
                        .map_err(|e| AdkError::new(
                            ErrorComponent::Model,
                            ErrorCategory::Unavailable,
                            "model.deepseek.request",
                            format!("DeepSeek API request failed: {e}"),
                        ).with_provider("deepseek"))?;

                    if !response.status().is_success() {
                        let status = response.status();
                        let status_code = status.as_u16();
                        let error_text = response.text().await.unwrap_or_default();
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
                            "model.deepseek.api_error",
                            format!("DeepSeek API error (HTTP {status}): {error_text}"),
                        ).with_upstream_status(status_code).with_provider("deepseek"));
                    }

                    Ok(response)
                }
            })
            .await?;

            if stream {
                let mut byte_stream = response.bytes_stream();
                let mut buffer = String::new();
                let mut tool_call_accumulators: std::collections::HashMap<u32, (String, String, String)> =
                    std::collections::HashMap::new();
                let mut reasoning_buffer = String::new();
                let mut text_buffer = String::new();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result
                        .map_err(|e| AdkError::model(format!("stream read error: {e}")))?;

                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    // Lines are cut at a consumed offset and the buffer is
                    // drained once per network read — re-slicing the remainder
                    // per line copied the whole buffer for every line
                    // (quadratic per read).
                    let mut consumed = 0usize;
                    while let Some(line_end) = buffer[consumed..].find('\n') {
                        let line = buffer[consumed..consumed + line_end].trim().to_string();
                        consumed += line_end + 1;

                        if line.is_empty() || line == "data: [DONE]" {
                            continue;
                        }

                        if let Some(data) = line.strip_prefix("data: ") {
                            match serde_json::from_str::<ChatCompletionResponse>(data) {
                                Ok(chunk_response) => {
                                    if let Some(choice) = chunk_response.choices.first() {
                                        if let Some(delta) = &choice.delta {
                                            // Accumulate reasoning content and stream it
                                            // out as partial Thinking events. Emission
                                            // mirrors what the server sends, not the
                                            // local thinking flag: a request without an
                                            // explicit `thinking` field (server default
                                            // = enabled on V4) still carries
                                            // reasoning_content deltas, and suppressing
                                            // them here would hide the model's
                                            // reasoning until the final event.
                                            if let Some(reasoning) = &delta.reasoning_content
                                                && !reasoning.is_empty() {
                                                    reasoning_buffer.push_str(reasoning);
                                                    yield LlmResponse {
                                                        content: Some(adk_core::Content {
                                                            role: "model".to_string(),
                                                            parts: vec![Part::Thinking {
                                                                thinking: reasoning.clone(),
                                                                signature: None,
                                                            }],
                                                        }),
                                                        partial: true,
                                                        turn_complete: false,
                                                        ..Default::default()
                                                    };
                                                }

                                            // Handle tool calls
                                            if let Some(tool_calls) = &delta.tool_calls {
                                                for tc in tool_calls {
                                                    let index = tc.index;
                                                    let entry = tool_call_accumulators
                                                        .entry(index)
                                                        .or_insert_with(|| {
                                                            let call_id = tc.id.clone().unwrap_or_else(|| {
                                                                format!("call_{index}")
                                                            });
                                                            (call_id, String::new(), String::new())
                                                        });
                                                    if let Some(id) = &tc.id {
                                                        entry.0.clone_from(id);
                                                    }
                                                    if let Some(func) = &tc.function {
                                                        // Some gateways repeat `function.name` as ""
                                                        // on every continuation delta (the OpenAI
                                                        // stream contract omits it instead); a blank
                                                        // must not clobber the name the first delta
                                                        // carried. The finish-side guard still fails
                                                        // loudly when NO delta ever named the call.
                                                        if let Some(name) = func
                                                            .name
                                                            .as_deref()
                                                            .filter(|n| !n.trim().is_empty())
                                                        {
                                                            entry.1 = name.to_string();
                                                        }
                                                        if let Some(args_chunk) = &func.arguments {
                                                            entry.2.push_str(args_chunk);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Check for finish
                                        if choice.finish_reason.is_some() {
                                            let finish_reason = choice.finish_reason.as_ref().map(|fr| {
                                                match fr.as_str() {
                                                    "stop" => FinishReason::Stop,
                                                    "length" => FinishReason::MaxTokens,
                                                    "tool_calls" => FinishReason::Stop,
                                                    "content_filter" => FinishReason::Safety,
                                                    _ => FinishReason::Stop,
                                                }
                                            });

                                            if !tool_call_accumulators.is_empty() {
                                                let mut sorted_calls: Vec<_> =
                                                    tool_call_accumulators.drain().collect();
                                                sorted_calls.sort_by_key(|(idx, _)| *idx);
                                                // Guard: an upstream (gateway) occasionally drops
                                                // `function.name` from the streamed tool-call
                                                // deltas, accumulating an empty name while the
                                                // arguments arrive intact. Emitting that call
                                                // poisons the turn: the dispatcher can't resolve
                                                // "" (a "Tool not found" result the model can't
                                                // act on), and the *next* request replays a
                                                // `name: ""` tool call that strict upstreams
                                                // reject with HTTP 400 "missing field `name`".
                                                // Failing here keeps the session history clean;
                                                // the error is marked retryable because the
                                                // drop is transient.
                                                if let Some((_, (id, _, _))) = sorted_calls
                                                    .iter()
                                                    .find(|(_, (_, name, _))| name.trim().is_empty())
                                                {
                                                    // try_stream!: `Err(e)?` yields the error item
                                                    // and ends the stream (a bare `yield Err(..)`
                                                    // would type as the Ok payload).
                                                    Err(AdkError::new(
                                                        ErrorComponent::Model,
                                                        ErrorCategory::Unavailable,
                                                        "model.deepseek.empty_tool_name",
                                                        format!(
                                                            "upstream streamed tool call `{id}` \
                                                             with an empty function.name; retry \
                                                             the turn"
                                                        ),
                                                    )
                                                    .with_retry(RetryHint {
                                                        should_retry: true,
                                                        ..Default::default()
                                                    })
                                                    .with_provider("deepseek"))?;
                                                    return;
                                                }
                                                let tool_calls: Vec<_> = sorted_calls
                                                    .into_iter()
                                                    .map(|(_, (id, name, args_str))| {
                                                        let args: Value =
                                                            serde_json::from_str(&args_str)
                                                                .unwrap_or(serde_json::json!({}));
                                                        (id, name, args)
                                                    })
                                                    .collect();
                                                // Buffered reasoning rides along when
                                                // present — the same server-default
                                                // reasoning a text turn keeps, so a
                                                // thinking→tool-call step does not
                                                // silently drop its trail.
                                                let tool_reasoning =
                                                    if reasoning_buffer.is_empty() {
                                                        None
                                                    } else {
                                                        Some(std::mem::take(
                                                            &mut reasoning_buffer,
                                                        ))
                                                    };
                                                yield convert::create_tool_call_response(
                                                    tool_calls,
                                                    finish_reason,
                                                    tool_reasoning,
                                                );
                                                continue;
                                            }

                                            let mut parts = Vec::new();
                                            if !reasoning_buffer.is_empty() {
                                                parts.push(Part::Thinking {
                                                    thinking: std::mem::take(&mut reasoning_buffer),
                                                    signature: None,
                                                });
                                            }
                                            if !text_buffer.is_empty() {
                                                parts.push(Part::Text {
                                                    text: std::mem::take(&mut text_buffer),
                                                });
                                            }

                                            let content = if parts.is_empty() {
                                                None
                                            } else {
                                                Some(adk_core::Content {
                                                    role: "model".to_string(),
                                                    parts,
                                                })
                                            };
                                            // Tool-call turns are not complete (issue #401).
                                            let turn_complete = content
                                                .as_ref()
                                                .is_none_or(|c| !c.has_function_calls());

                                            yield LlmResponse {
                                                content,
                                                usage_metadata: chunk_response.usage.map(|u| {
                                                    adk_core::UsageMetadata {
                                                        prompt_token_count: u.prompt_tokens as i32,
                                                        candidates_token_count: u.completion_tokens as i32,
                                                        total_token_count: u.total_tokens as i32,
                                                        thinking_token_count: u.reasoning_tokens.map(|t| t as i32),
                                                        cache_read_input_token_count: u.prompt_cache_hit_tokens.map(|t| t as i32),
                                                        cache_creation_input_token_count: u.prompt_cache_miss_tokens.map(|t| t as i32),
                                                        ..Default::default()
                                                    }
                                                }),
                                                finish_reason,
                                                partial: false,
                                                turn_complete,
                                                ..Default::default()
                                            };
                                        } else {
                                            // Emit partial text content and accumulate
                                            if let Some(delta) = &choice.delta
                                                && let Some(text) = &delta.content
                                                    && !text.is_empty() {
                                                        text_buffer.push_str(text);
                                                        yield LlmResponse {
                                                            content: Some(adk_core::Content {
                                                                role: "model".to_string(),
                                                                parts: vec![Part::Text {
                                                                    text: text.clone(),
                                                                }],
                                                            }),
                                                            partial: true,
                                                            turn_complete: false,
                                                            ..Default::default()
                                                        };
                                                    }
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("failed to parse DeepSeek chunk: {e} - {data}");
                                }
                            }
                        }
                    }
                    buffer.drain(..consumed);
                }
            } else {
                // Non-streaming mode
                let response_text = response.text().await
                    .map_err(|e| AdkError::model(format!("failed to read response: {e}")))?;

                let chat_response: ChatCompletionResponse = serde_json::from_str(&response_text)
                    .map_err(|e| AdkError::model(format!(
                        "failed to parse response: {e} - {response_text}"
                    )))?;

                yield convert::from_response(&chat_response);
            }
        };

        Ok(crate::usage_tracking::with_usage_tracking(Box::pin(response_stream), usage_span))
    }
}

#[cfg(test)]
mod response_format_tests {
    //! `build_request` read temperature, top-p, token limits, tools, thinking, and
    //! reasoning effort but always sent `response_format: None`, even with a
    //! `response_schema` present, while the module advertised structured JSON output.
    //! The schema reached the model only as the agent's textual instruction, so native
    //! enforcement was never requested and structured turns could cost retries.

    use super::*;
    use adk_core::{Content, GenerateContentConfig, LlmRequest};
    use serde_json::json;

    fn client() -> DeepSeekClient {
        DeepSeekClient::chat("test-key").expect("client builds")
    }

    fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"]
        })
    }

    fn request_with(config: Option<GenerateContentConfig>, prompt: &str) -> LlmRequest {
        let mut request =
            LlmRequest::new("deepseek-chat", vec![Content::new("user").with_text(prompt)]);
        request.config = config;
        request
    }

    #[test]
    fn a_response_schema_requests_json_output() {
        let request = request_with(
            Some(GenerateContentConfig { response_schema: Some(schema()), ..Default::default() }),
            "give me the answer as json",
        );

        let built = client().build_request(&request, false);
        let wire = serde_json::to_value(&built).expect("request serializes");

        // DeepSeek accepts only `json_object`; there is no `json_schema` mode.
        assert_eq!(
            wire["response_format"],
            json!({ "type": "json_object" }),
            "a response schema must request DeepSeek's JSON Output mode"
        );
    }

    #[test]
    fn no_schema_leaves_the_response_format_unset() {
        let request = request_with(None, "hello");
        let built = client().build_request(&request, false);
        let wire = serde_json::to_value(&built).expect("request serializes");

        assert!(
            wire.get("response_format").is_none(),
            "an ordinary turn must not request JSON Output"
        );
    }

    #[test]
    fn thinking_mode_and_effort_reach_the_wire() {
        // Explicit knobs serialize; an unset config omits both fields so the
        // server default applies.
        let explicit = DeepSeekClient::new(
            DeepSeekConfig::new("test-key", "deepseek-v4-flash")
                .with_thinking_mode(crate::deepseek::config::ThinkingMode::Enabled)
                .with_reasoning_effort(crate::deepseek::config::ReasoningEffort::Max),
        )
        .expect("client builds");
        let wire =
            serde_json::to_value(explicit.build_request(&request_with(None, "hi"), false))
                .expect("request serializes");
        assert_eq!(wire["thinking"], json!({ "type": "enabled" }));
        assert_eq!(wire["reasoning_effort"], json!("max"));

        let default_wire =
            serde_json::to_value(client().build_request(&request_with(None, "hi"), false))
                .expect("request serializes");
        assert!(default_wire.get("thinking").is_none());
        assert!(default_wire.get("reasoning_effort").is_none());
    }

    #[test]
    fn the_documented_json_keyword_requirement_is_satisfied() {
        // DeepSeek requires the word "json" in the system or user prompt whenever
        // JSON Output is enabled, or the API may return empty content.
        let request = request_with(
            Some(GenerateContentConfig { response_schema: Some(schema()), ..Default::default() }),
            "summarise the document",
        );

        let built = client().build_request(&request, false);

        assert!(
            built.messages.iter().any(|m| m
                .content
                .as_deref()
                .is_some_and(|text| text.to_lowercase().contains("json"))),
            "enabling JSON Output without the keyword risks empty responses"
        );
    }

    #[test]
    fn an_existing_json_mention_is_not_duplicated() {
        let request = request_with(
            Some(GenerateContentConfig { response_schema: Some(schema()), ..Default::default() }),
            "reply in json please",
        );

        let built = client().build_request(&request, false);

        assert_eq!(
            built.messages.len(),
            1,
            "the prompt already mentions json, so nothing needs to be added"
        );
    }
}

#[cfg(test)]
mod empty_tool_name_tests {
    //! An upstream gateway occasionally drops `function.name` from streamed
    //! tool-call deltas (arguments arrive intact). The client must fail the
    //! turn with a retryable error instead of emitting a `name: ""` call that
    //! the dispatcher can't resolve and that poisons the next request with a
    //! 400 "missing field `name`" on strict upstreams.

    use super::*;
    use adk_core::{Content, LlmRequest};
    use futures::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sse(chunks: &[serde_json::Value]) -> String {
        let mut body = String::new();
        for c in chunks {
            body.push_str(&format!("data: {}\n\n", serde_json::to_string(c).unwrap()));
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn chunk(delta: serde_json::Value, finish: Option<&str>) -> serde_json::Value {
        serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }]
        })
    }

    fn client_for(base: String) -> DeepSeekClient {
        DeepSeekClient::new(
            DeepSeekConfig::new("test-key", "deepseek-v4-flash").with_base_url(base),
        )
        .expect("client should build")
    }

    fn request() -> LlmRequest {
        LlmRequest::new("deepseek-v4-flash", vec![Content::new("user").with_text("hi")])
    }

    #[tokio::test]
    async fn empty_streamed_tool_name_fails_the_turn() {
        let server = MockServer::start().await;
        let body = sse(&[
            chunk(
                serde_json::json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_0",
                        "type": "function",
                        "function": { "name": "", "arguments": "{\"path\":\"/tmp\"}" }
                    }]
                }),
                None,
            ),
            chunk(serde_json::json!({}), Some("tool_calls")),
        ]);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("Content-Type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = client_for(server.uri());
        let mut stream = client.generate_content(request(), true).await.expect("stream starts");
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                assert_eq!(e.code, "model.deepseek.empty_tool_name", "unexpected error: {e}");
                assert!(e.is_retryable(), "the drop is transient, retry must be suggested");
                assert!(e.message.contains("call_0"), "error should name the call id: {e}");
                saw_err = true;
            }
        }
        assert!(saw_err, "stream must yield an error for an empty tool name");
    }

    #[tokio::test]
    async fn named_streamed_tool_call_still_passes() {
        let server = MockServer::start().await;
        let body = sse(&[
            chunk(
                serde_json::json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_0",
                        "type": "function",
                        "function": { "name": "read_file", "arguments": "{\"path\":\"/tmp\"}" }
                    }]
                }),
                None,
            ),
            chunk(serde_json::json!({}), Some("tool_calls")),
        ]);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("Content-Type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = client_for(server.uri());
        let mut stream = client.generate_content(request(), true).await.expect("stream starts");
        let mut saw_call = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(resp) => {
                    if let Some(c) = &resp.content
                        && c.has_function_calls()
                    {
                        saw_call = true;
                    }
                }
                Err(e) => panic!("a named tool call must not error: {e}"),
            }
        }
        assert!(saw_call, "the named tool call should reach the stream");
    }
}

#[cfg(test)]
mod reasoning_stream_tests {
    //! `reasoning_content` deltas must stream out as partial `Thinking` events
    //! and reach the final response — regardless of the local thinking flag.
    //! A request without an explicit `thinking` field (the `DeepSeekConfig::new`
    //! default) still gets server-default thinking on V4 models, so gating
    //! emission on the client flag hid the model's reasoning until the final
    //! event and dropped it entirely on thinking→tool-call turns.

    use super::*;
    use adk_core::{Content, LlmRequest, Part};
    use futures::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sse(chunks: &[serde_json::Value]) -> String {
        let mut body = String::new();
        for c in chunks {
            body.push_str(&format!("data: {}\n\n", serde_json::to_string(c).unwrap()));
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn chunk(delta: serde_json::Value, finish: Option<&str>) -> serde_json::Value {
        serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 40,
                "total_tokens": 52,
                "completion_tokens_details": { "reasoning_tokens": 25 }
            }
        })
    }

    fn client_for(base: String) -> DeepSeekClient {
        // `DeepSeekConfig::new` leaves `thinking: None` (server default) and
        // the legacy `thinking_enabled` false — the exact configuration whose
        // reasoning used to be suppressed.
        DeepSeekClient::new(
            DeepSeekConfig::new("test-key", "deepseek-v4-flash").with_base_url(base),
        )
        .expect("client should build")
    }

    fn request() -> LlmRequest {
        LlmRequest::new("deepseek-v4-flash", vec![Content::new("user").with_text("hi")])
    }

    async fn mount(server: &MockServer, body: String) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("Content-Type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(server)
            .await;
    }

    fn reasoning_text(resp: &LlmResponse) -> Option<String> {
        let content = resp.content.as_ref()?;
        let mut out = String::new();
        for part in &content.parts {
            if let Part::Thinking { thinking, .. } = part {
                out.push_str(thinking);
            }
        }
        (!out.is_empty()).then_some(out)
    }

    #[tokio::test]
    async fn reasoning_deltas_stream_without_local_thinking_flag() {
        let server = MockServer::start().await;
        mount(
            &server,
            sse(&[
                chunk(serde_json::json!({ "reasoning_content": "step one… " }), None),
                chunk(serde_json::json!({ "reasoning_content": "step two" }), None),
                chunk(serde_json::json!({ "content": "The answer." }), None),
                chunk(serde_json::json!({}), Some("stop")),
            ]),
        )
        .await;

        let client = client_for(server.uri());
        let mut stream = client.generate_content(request(), true).await.expect("stream starts");
        let mut partials = Vec::new();
        let mut final_thinking = None;
        let mut final_text = None;
        while let Some(item) = stream.next().await {
            let resp = item.expect("stream item");
            if resp.partial {
                let text = resp.content.as_ref().map(|c| {
                    c.parts
                        .iter()
                        .map(|p| match p {
                            Part::Text { text } => text.clone(),
                            _ => String::new(),
                        })
                        .collect::<String>()
                });
                partials.push((reasoning_text(&resp), text));
            } else {
                final_thinking = reasoning_text(&resp);
                final_text = resp.content.as_ref().map(|c| {
                    c.parts.iter().map(|p| match p {
                        Part::Text { text } => text.clone(),
                        _ => String::new(),
                    }).collect::<String>()
                });
            }
        }
        // Both reasoning deltas streamed out as partial Thinking events (the
        // answer-text delta is a third, reasoning-free partial).
        let reasoning_partials: Vec<_> =
            partials.iter().filter(|(r, _)| r.is_some()).collect();
        assert_eq!(
            reasoning_partials.len(),
            2,
            "both reasoning deltas are partial events"
        );
        assert_eq!(reasoning_partials[0].0.as_deref(), Some("step one… "));
        assert_eq!(reasoning_partials[1].0.as_deref(), Some("step two"));
        // The final response carries the assembled reasoning + text + usage.
        assert_eq!(final_thinking.as_deref(), Some("step one… step two"));
        assert_eq!(final_text.as_deref(), Some("The answer."));
    }

    #[tokio::test]
    async fn tool_call_turn_keeps_its_reasoning_without_local_flag() {
        let server = MockServer::start().await;
        mount(
            &server,
            sse(&[
                chunk(serde_json::json!({ "reasoning_content": "need a tool" }), None),
                chunk(
                    serde_json::json!({
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": { "name": "read_file", "arguments": "{\"path\":\"/tmp\"}" }
                        }]
                    }),
                    None,
                ),
                chunk(serde_json::json!({}), Some("tool_calls")),
            ]),
        )
        .await;

        let client = client_for(server.uri());
        let mut stream = client.generate_content(request(), true).await.expect("stream starts");
        let mut tool_thinking = None;
        while let Some(item) = stream.next().await {
            let resp = item.expect("stream item");
            if let Some(c) = &resp.content
                && c.has_function_calls()
            {
                tool_thinking = reasoning_text(&resp);
            }
        }
        assert_eq!(
            tool_thinking.as_deref(),
            Some("need a tool"),
            "a thinking→tool-call turn keeps its reasoning trail"
        );
    }
}
