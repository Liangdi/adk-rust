//! Streaming output from ACP agent sessions.
//!
//! Instead of collecting the full response into a string, streaming mode
//! yields chunks as they arrive from the agent — enabling real-time display
//! and lower time-to-first-token.

use std::sync::Arc;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionKind,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SessionNotification, SessionUpdate, ToolCallContent, ToolCallStatus, ToolCallUpdate, ToolKind,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Responder};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::connection::AcpAgentConfig;
use crate::error::Result;
use crate::host::{AcpHostHandler, capabilities};
use crate::permissions::{PermissionPolicy, PermissionRequest, outcome_for};
use crate::status::{AgentStatus, StatusTracker};

/// A chunk of output from the ACP agent.
#[derive(Debug, Clone)]
pub enum OutputChunk {
    /// A text chunk from the agent's response.
    Text(String),
    /// The agent is thinking (internal reasoning, not shown to user by default).
    Thought(String),
    /// A tool call was initiated (e.g., "Creating file app.rs").
    ToolCall {
        /// The tool-call identifier; [`ToolUpdate`](Self::ToolUpdate) chunks
        /// correlate back to this id (parallel calls must be matched by it).
        id: String,
        /// Human-readable title of the operation.
        title: String,
        /// MCP server attribution (`_meta.mcp_server`) when the tool comes
        /// from an MCP toolset exposed as `mcp__{server}__{tool}`.
        server: Option<String>,
    },
    /// A tool call completed.
    ToolCallComplete {
        /// Human-readable title.
        title: String,
    },
    /// An update on the status or results of a previously initiated tool call.
    ///
    /// Surfaces the External_Agent's `SessionUpdate::ToolCallUpdate`. The `id`
    /// correlates this update back to its originating [`OutputChunk::ToolCall`].
    ToolUpdate {
        /// The tool-call identifier this update correlates to.
        id: String,
        /// Execution status if reported (`pending`, `in_progress`, `completed`, `failed`).
        status: Option<String>,
        /// Tool kind if reported (`read`, `edit`, `execute`, ...).
        kind: Option<String>,
        /// Updated human-readable title if reported.
        title: Option<String>,
        /// Human-readable text summary extracted from the update content, if any.
        content: Option<String>,
        /// MCP server attribution (`_meta.mcp_server`), when reported.
        server: Option<String>,
        /// Absolute file paths the tool reported affecting.
        locations: Vec<String>,
    },
    /// A usage update reporting context-window consumption and (optionally) cost.
    ///
    /// Surfaces the External_Agent's `SessionUpdate::UsageUpdate`.
    Usage {
        /// Tokens currently in context.
        used: u64,
        /// Total context window size in tokens.
        size: u64,
        /// Cumulative session cost amount, if reported.
        cost: Option<f64>,
        /// ISO 4217 currency code accompanying `cost`, if reported.
        currency: Option<String>,
    },
    /// The agent requested permission (informational — decision already made by policy).
    PermissionRequested {
        /// What the agent wanted to do.
        title: String,
        /// Whether it was approved.
        approved: bool,
    },
    /// The agent finished responding.
    Done,
    /// An error occurred.
    Error(String),
}

/// A streaming receiver for ACP agent output.
///
/// Yields [`OutputChunk`]s as they arrive from the agent.
///
/// # Example
///
/// ```rust,ignore
/// use adk_acp::streaming::stream_prompt;
///
/// let mut stream = stream_prompt(&config, "Write a hello world", policy, status).await?;
/// while let Some(chunk) = stream.recv().await {
///     match chunk {
///         OutputChunk::Text(t) => print!("{t}"),
///         OutputChunk::ToolCall { title } => println!("\n[tool] {title}"),
///         OutputChunk::Done => break,
///         _ => {}
///     }
/// }
/// ```
pub type OutputStream = mpsc::Receiver<OutputChunk>;

/// Sink the client uses to surface an agent-initiated elicitation
/// (`elicitation/create`) to the host application and await the user's answer.
#[async_trait::async_trait]
pub trait ElicitationSink: std::fmt::Debug + Send + Sync {
    /// Resolve an elicitation request. Return `Err` to deny the request
    /// (the agent's tool receives an error).
    async fn resolve(
        &self,
        request: agent_client_protocol::schema::v1::CreateElicitationRequest,
    ) -> std::result::Result<
        agent_client_protocol::schema::v1::CreateElicitationResponse,
        String,
    >;
}

/// Send a prompt and stream the response chunks.
///
/// Returns a receiver that yields [`OutputChunk`]s as they arrive.
/// The agent process is terminated when the stream completes.
pub async fn stream_prompt(
    config: &AcpAgentConfig,
    prompt: &str,
    policy: Arc<PermissionPolicy>,
    status: StatusTracker,
) -> Result<OutputStream> {
    stream_prompt_inner(config, prompt, policy, status, None).await
}

/// Like [`stream_prompt`] but also routes agent-initiated elicitation
/// (`elicitation/create`) requests through `elicitation`. Requires the
/// `unstable_elicitation` feature on the `agent-client-protocol` dependency.
pub async fn stream_prompt_elicit(
    config: &AcpAgentConfig,
    prompt: &str,
    policy: Arc<PermissionPolicy>,
    status: StatusTracker,
    elicitation: Arc<dyn ElicitationSink>,
) -> Result<OutputStream> {
    stream_prompt_inner(config, prompt, policy, status, Some(elicitation)).await
}

async fn stream_prompt_inner(
    config: &AcpAgentConfig,
    prompt: &str,
    policy: Arc<PermissionPolicy>,
    status: StatusTracker,
    elicitation: Option<Arc<dyn ElicitationSink>>,
) -> Result<OutputStream> {
    info!(command = %config.command, "starting streaming ACP prompt");

    let agent = crate::connection::build_agent(config)?;

    let (chunk_tx, chunk_rx) = mpsc::channel::<OutputChunk>(64);
    let prompt_text = prompt.to_string();
    let working_dir = config.working_dir.clone();
    let mcp_servers = config.mcp_servers.clone();
    let filesystem = config.filesystem.clone();
    let terminal = config.terminal.clone();
    let client_capabilities = capabilities(filesystem.as_ref(), terminal.as_ref());

    status.set(AgentStatus::Starting);

    tokio::spawn(async move {
        let chunk_tx_err = chunk_tx.clone();
        let status_inner = status.clone();
        let policy_clone = policy.clone();
        let chunk_tx_perm = chunk_tx.clone();

        let outcome = Client
            .builder()
            .on_receive_notification(
                {
                    let tx = chunk_tx.clone();
                    async move |notif: SessionNotification, _cx: ConnectionTo<Agent>| {
                        if let Some(chunk) = map_update(notif.update) {
                            let _ = tx.send(chunk).await;
                        }
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                {
                    let status = status_inner.clone();
                    move |request: RequestPermissionRequest,
                          responder: Responder<RequestPermissionResponse>,
                          cx: ConnectionTo<Agent>| {
                        let status = status.clone();
                        let policy = policy_clone.clone();
                        let permission_tx = chunk_tx_perm.clone();
                        async move {
                            status.set(AgentStatus::WaitingPermission);
                            cx.spawn(async move {
                            let cancellation = responder.cancellation();
                            let perm_request = PermissionRequest::from_acp(&request);
                            let outcome = tokio::select! {
                                decision = policy.decide(&perm_request) => {
                                    outcome_for(&perm_request, &decision)
                                }
                                _ = cancellation.cancelled() => RequestPermissionOutcome::Cancelled,
                            };
                            let approved = match &outcome {
                                RequestPermissionOutcome::Selected(selected) => perm_request
                                    .options
                                    .iter()
                                    .find(|option| option.id == selected.option_id.to_string())
                                    .is_some_and(|option| {
                                        matches!(
                                            option.kind,
                                            PermissionOptionKind::AllowOnce
                                                | PermissionOptionKind::AllowAlways
                                        )
                                    }),
                                RequestPermissionOutcome::Cancelled => false,
                                _ => false,
                            };

                            let _ = permission_tx
                                .send(OutputChunk::PermissionRequested {
                                    title: perm_request.title.clone(),
                                    approved,
                                })
                                .await;
                            status.set(AgentStatus::Running);
                            responder.respond(RequestPermissionResponse::new(outcome))
                        })?;
                            Ok(())
                        }
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let elicitation = elicitation.clone();
                    move |request: agent_client_protocol::schema::v1::CreateElicitationRequest,
                          responder: agent_client_protocol::Responder<
                        agent_client_protocol::schema::v1::CreateElicitationResponse,
                    >,
                          cx: ConnectionTo<Agent>| {
                        let elicitation = elicitation.clone();
                        async move {
                            // No sink wired → reject (method not found) so the
                            // agent's tool degrades rather than hanging.
                            let Some(sink) = elicitation else {
                                return Ok(());
                            };
                            cx.spawn(async move {
                                match sink.resolve(request).await {
                                    Ok(response) => responder.respond(response),
                                    Err(message) => {
                                        responder.respond_with_result(Err(
                                            agent_client_protocol::Error::method_not_found()
                                                .data(message),
                                        ))
                                    }
                                }
                            })?;
                            Ok(())
                        }
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .with_handler(AcpHostHandler::new(filesystem, terminal))
            .connect_with(agent, {
                let status = status_inner.clone();
                let tx = chunk_tx.clone();
                |connection: ConnectionTo<Agent>| async move {
                    status.set(AgentStatus::Starting);

                    let initialization = connection
                        .send_request(
                            InitializeRequest::new(ProtocolVersion::V1)
                                .client_capabilities(client_capabilities),
                        )
                        .block_task()
                        .await?;
                    crate::connection::validate_initialization(&initialization, &mcp_servers)?;

                    status.set(AgentStatus::Running);

                    connection
                        .build_session_from(
                            NewSessionRequest::new(&working_dir).mcp_servers(mcp_servers),
                        )
                        .block_task()
                        .run_until(async |mut session| {
                            session.send_prompt(&prompt_text)?;
                            // read_to_string collects internally; notifications stream via callback
                            let _ = session.read_to_string().await?;
                            let _ = tx.send(OutputChunk::Done).await;
                            Ok(())
                        })
                        .await?;

                    status.set(AgentStatus::Idle);
                    Ok(())
                }
            })
            .await;

        if let Err(e) = outcome {
            warn!(error = %e, "streaming ACP session ended with error");
            let _ = chunk_tx_err.send(OutputChunk::Error(e.to_string())).await;
        }

        status_inner.set(AgentStatus::Stopped);
    });

    Ok(chunk_rx)
}

/// Map an incoming ACP [`SessionUpdate`] from an External_Agent to an
/// [`OutputChunk`] for the client's streaming surface.
///
/// This is a pure function so the mapping can be unit-tested without spawning a
/// subprocess or driving a live connection. Returns `None` for updates that
/// have no client-facing representation.
///
/// The text paths (`AgentMessageChunk` → [`OutputChunk::Text`],
/// `AgentThoughtChunk` → [`OutputChunk::Thought`]) are preserved exactly as they
/// were before richer updates were surfaced (see design property P12).
/// Map an ACP `SessionUpdate` notification to an [`OutputChunk`].
///
/// Public so HTTP-bridge style consumers (e.g. `agent_client_protocol_http`
/// clients that receive raw notifications) can reuse the exact stdio-path
/// semantics: text/thought chunks, tool-call start, tool-call updates
/// (status/kind/title/content/locations), and usage. Unmapped variants
/// yield `None`.
pub fn map_update(update: SessionUpdate) -> Option<OutputChunk> {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(text_content) => {
                Some(OutputChunk::Text(text_content.text.to_string()))
            }
            _ => None,
        },
        SessionUpdate::AgentThoughtChunk(chunk) => match chunk.content {
            ContentBlock::Text(text_content) => {
                Some(OutputChunk::Thought(text_content.text.to_string()))
            }
            _ => None,
        },
        SessionUpdate::ToolCall(tool_call) => Some(OutputChunk::ToolCall {
            id: tool_call.tool_call_id.to_string(),
            title: tool_call.title.to_string(),
            server: meta_mcp_server(&tool_call.meta),
        }),
        SessionUpdate::ToolCallUpdate(update) => Some(map_tool_call_update(update)),
        SessionUpdate::UsageUpdate(usage) => Some(OutputChunk::Usage {
            used: usage.used,
            size: usage.size,
            cost: usage.cost.as_ref().map(|cost| cost.amount),
            currency: usage.cost.as_ref().map(|cost| cost.currency.clone()),
        }),
        _ => None,
    }
}

/// Build an [`OutputChunk::ToolUpdate`] from an ACP [`ToolCallUpdate`],
/// surfacing the status, kind, title, extracted content text, and affected
/// file locations while preserving the tool-call id correlation.
fn map_tool_call_update(update: ToolCallUpdate) -> OutputChunk {
    let ToolCallUpdate { tool_call_id, fields, meta, .. } = update;

    let locations = fields
        .locations
        .unwrap_or_default()
        .into_iter()
        .map(|location| location.path.display().to_string())
        .collect();

    OutputChunk::ToolUpdate {
        id: tool_call_id.to_string(),
        status: fields.status.map(tool_status_str).map(str::to_string),
        kind: fields.kind.map(tool_kind_str).map(str::to_string),
        title: fields.title,
        content: extract_tool_content_text(fields.content.as_deref()),
        server: meta_mcp_server(&meta),
        locations,
    }
}

/// Read the `_meta.mcp_server` attribution from an ACP tool-call event, if the
/// agent set one. Key is private to this ADK implementation (the spec reserves
/// `_meta` for implementation-defined extensions).
fn meta_mcp_server(meta: &Option<agent_client_protocol::schema::v1::Meta>) -> Option<String> {
    meta.as_ref()
        .and_then(|meta| meta.get("mcp_server"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// Extract a human-readable text summary from tool-call update content blocks.
///
/// Joins the text of any standard text content blocks. Non-text content
/// (diffs, terminals, images) is ignored for the summary.
fn extract_tool_content_text(content: Option<&[ToolCallContent]>) -> Option<String> {
    let content = content?;
    let mut parts = Vec::new();
    for item in content {
        if let ToolCallContent::Content(block) = item
            && let ContentBlock::Text(text_content) = &block.content
        {
            parts.push(text_content.text.to_string());
        }
    }
    if parts.is_empty() { None } else { Some(parts.join("\n")) }
}

/// Stable wire string for a [`ToolCallStatus`].
fn tool_status_str(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
        _ => "unknown",
    }
}

/// Stable wire string for a [`ToolKind`].
fn tool_kind_str(kind: ToolKind) -> &'static str {
    match kind {
        ToolKind::Read => "read",
        ToolKind::Edit => "edit",
        ToolKind::Delete => "delete",
        ToolKind::Move => "move",
        ToolKind::Search => "search",
        ToolKind::Execute => "execute",
        ToolKind::Think => "think",
        ToolKind::Fetch => "fetch",
        ToolKind::SwitchMode => "switch_mode",
        ToolKind::Other => "other",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{
        ContentChunk, Cost, ToolCall, ToolCallContent, ToolCallLocation, ToolCallUpdate,
        ToolCallUpdateFields, UsageUpdate,
    };

    /// P12: agent message text must be surfaced exactly as before — the raw
    /// text payload of an `AgentMessageChunk`, byte-for-byte.
    #[test]
    fn agent_message_chunk_surfaces_text_unchanged() {
        let update =
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::from("hello world")));

        match map_update(update) {
            Some(OutputChunk::Text(text)) => assert_eq!(text, "hello world"),
            other => panic!("expected Text chunk, got {other:?}"),
        }
    }

    /// Thought chunks continue to surface as before.
    #[test]
    fn agent_thought_chunk_surfaces_thought() {
        let update =
            SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::from("let me think")));

        match map_update(update) {
            Some(OutputChunk::Thought(text)) => assert_eq!(text, "let me think"),
            other => panic!("expected Thought chunk, got {other:?}"),
        }
    }

    /// A non-text agent message block produces no chunk (unchanged behavior).
    #[test]
    fn non_text_agent_message_is_ignored() {
        use agent_client_protocol::schema::v1::ImageContent;
        let update = SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Image(
            ImageContent::new("AAAA", "image/png"),
        )));

        assert!(map_update(update).is_none());
    }

    /// Tool calls continue to surface their title.
    #[test]
    fn tool_call_surfaces_title() {
        let update = SessionUpdate::ToolCall(ToolCall::new("tc-1", "Reading app.rs"));

        match map_update(update) {
            Some(OutputChunk::ToolCall { id, title, server }) => {
                assert_eq!(id, "tc-1");
                assert_eq!(title, "Reading app.rs");
                assert!(server.is_none(), "no _meta.mcp_server was set");
            }
            other => panic!("expected ToolCall chunk, got {other:?}"),
        }
    }

    /// Tool-call updates now surface status, kind, title, content, and
    /// locations, preserving the tool-call id correlation.
    #[test]
    fn tool_call_update_surfaces_status_and_summary() {
        let fields = ToolCallUpdateFields::new()
            .status(ToolCallStatus::Completed)
            .kind(ToolKind::Edit)
            .title("Edited app.rs".to_string())
            .content(vec![ToolCallContent::from(ContentBlock::from("done"))])
            .locations(vec![ToolCallLocation::new("/tmp/app.rs")]);
        let update = SessionUpdate::ToolCallUpdate(ToolCallUpdate::new("tc-1", fields));

        match map_update(update) {
            Some(OutputChunk::ToolUpdate { id, status, kind, title, content, server, locations }) => {
                assert_eq!(id, "tc-1");
                assert_eq!(status.as_deref(), Some("completed"));
                assert_eq!(kind.as_deref(), Some("edit"));
                assert_eq!(title.as_deref(), Some("Edited app.rs"));
                assert_eq!(content.as_deref(), Some("done"));
                assert!(server.is_none(), "no _meta.mcp_server was set");
                assert_eq!(locations, vec!["/tmp/app.rs".to_string()]);
            }
            other => panic!("expected ToolUpdate chunk, got {other:?}"),
        }
    }

    /// A minimal tool-call update (id only) still surfaces with empty optionals.
    #[test]
    fn tool_call_update_minimal_surfaces_id() {
        let update =
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new("tc-9", ToolCallUpdateFields::new()));

        match map_update(update) {
            Some(OutputChunk::ToolUpdate { id, status, kind, title, content, server, locations }) => {
                assert_eq!(id, "tc-9");
                assert!(status.is_none());
                assert!(kind.is_none());
                assert!(title.is_none());
                assert!(content.is_none());
                assert!(server.is_none());
                assert!(locations.is_empty());
            }
            other => panic!("expected ToolUpdate chunk, got {other:?}"),
        }
    }

    /// Usage updates surface token counts and cost when present.
    #[test]
    fn usage_update_surfaces_tokens_and_cost() {
        let update = SessionUpdate::UsageUpdate(
            UsageUpdate::new(53_000, 200_000).cost(Cost::new(0.045, "USD")),
        );

        match map_update(update) {
            Some(OutputChunk::Usage { used, size, cost, currency }) => {
                assert_eq!(used, 53_000);
                assert_eq!(size, 200_000);
                assert_eq!(cost, Some(0.045));
                assert_eq!(currency.as_deref(), Some("USD"));
            }
            other => panic!("expected Usage chunk, got {other:?}"),
        }
    }

    /// Usage updates without cost surface token counts and no cost.
    #[test]
    fn usage_update_without_cost() {
        let update = SessionUpdate::UsageUpdate(UsageUpdate::new(10, 100));

        match map_update(update) {
            Some(OutputChunk::Usage { used, size, cost, currency }) => {
                assert_eq!(used, 10);
                assert_eq!(size, 100);
                assert!(cost.is_none());
                assert!(currency.is_none());
            }
            other => panic!("expected Usage chunk, got {other:?}"),
        }
    }
}
