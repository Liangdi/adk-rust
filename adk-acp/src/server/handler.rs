//! ACP session lifecycle and ADK-Rust Runner bridge.
//!
//! # Session activation semantics (for ACP client authors)
//!
//! The active-session map is process-level: one entry per activated session,
//! shared by every connection the server serves (the HTTP transport hands a
//! clone of the same handler to each inbound connection). An entry is a
//! marker, not a lease owned by a connection. `session/load` or
//! `session/resume` for a session whose last turn already finished — or whose
//! activating connection went away without sending `session/close` — simply
//! rebinds the entry to the requesting connection. A client that opens a
//! fresh connection per turn therefore needs no `session/close` to continue a
//! session; it can `session/load` the same id from each new connection.
//!
//! The rebind is refused only while a prompt is still executing in the
//! session. That liveness is tracked by a drop-guarded probe
//! ([`InFlightPrompt`]), so even a prompt abandoned mid-run when its
//! connection died stops blocking as soon as the SDK drops the task. Two
//! connections may still alternate prompts on one session; each individual
//! turn remains serial (the in-flight guard in
//! [`handle_prompt`](AcpSessionHandler::handle_prompt) rejects concurrent
//! turns).
//!
//! Explicit `session/close` keeps its original meaning: it marks the session
//! inactive without deleting its persisted history.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use adk_core::{
    Agent, Content, RunConfig, SessionId as AdkSessionId, ToolConfirmationDecision,
    ToolConfirmationHandler, ToolConfirmationRequest, Toolset, UserId,
};
use adk_runner::Runner;
use adk_session::{CreateRequest, DeleteRequest, GetRequest, ListRequest, SessionService};
use adk_tool::McpToolset;
#[cfg(feature = "mcp-http")]
use adk_tool::mcp::McpHttpClientBuilder;
use agent_client_protocol::RequestCancellation;
use agent_client_protocol::schema::v1::{
    ContentBlock, ListSessionsRequest, ListSessionsResponse, McpServer, Meta, NewSessionRequest,
    PromptRequest, SessionId, SessionInfo, SessionNotification, StopReason,
};
use agent_client_protocol::{Client, ConnectionTo};
use futures::StreamExt;
use rmcp::{ServiceExt, transport::TokioChildProcess};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::config::{AcpServerConfig, AgentFactory};
use super::error::AcpServerError;
use super::modes::{
    self, SessionControls, config_state_key, config_value_is_valid, mode_is_advertised,
    option_with_current_value,
};
use super::permission::PermissionBridge;
use super::streamer::ResponseStreamer;
use agent_client_protocol::schema::v1::{
    AvailableCommandsUpdate, ConfigOptionUpdate, CurrentModeUpdate, SessionConfigId,
    SessionConfigOption, SessionConfigOptionValue, SessionInfoUpdate, SessionModeId,
    SessionModeState, SessionUpdate,
};

const CWD_STATE_KEY: &str = "acp:cwd";
const ADDITIONAL_DIRS_STATE_KEY: &str = "acp:additional_directories";
/// Session-state key under which a human-readable session title is stored.
///
/// A [`SessionInfoUpdate`] is emitted from this key on session activation (when
/// present) and whenever [`set_session_title`](AcpSessionHandler::set_session_title)
/// changes it, so a title survives load / resume / fork like other ACP state.
const TITLE_STATE_KEY: &str = "acp:title";
const MCP_STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

struct SessionState {
    execution: Option<InFlightPrompt>,
    mcp: McpSessionResources,
    /// Per-session agent composed by the configured [`AgentFactory`] from the
    /// session's `_meta` map and cwd. `None` (no factory configured, or a
    /// load-without-`_meta` that chose to keep nothing) means the
    /// config-level agent serves this session; `create_session` composes
    /// every session when a factory exists (an absent `_meta` is an empty
    /// map there). Preserved across load/resume rebinds so a client that
    /// pinned a model on `session/new` keeps it on a later `session/load`
    /// without re-sending `_meta`.
    agent: Option<Arc<dyn Agent>>,
}

/// Liveness probe for the prompt currently executing in a session.
///
/// The session map holds one cheap clone; the prompt task's frame holds the
/// matching [`InFlightGuard`]. The probe reports "running" until the guard is
/// dropped, so every exit path — normal completion, early error return, and
/// the prompt future being dropped mid-execution when its ACP connection dies
/// (the official SDK tears down a connection's task actor on transport
/// failure, which drops `handle_prompt`'s future without running its tail) —
/// releases the session instead of leaving it permanently busy.
///
/// This is what makes idle-rebinding safe: a session can only be rebound by
/// `session/load` / `session/resume` when no live prompt holds it.
#[derive(Clone)]
struct InFlightPrompt {
    inner: Arc<InFlightInner>,
}

struct InFlightInner {
    token: CancellationToken,
    finished: AtomicBool,
}

/// Task-side half of [`InFlightPrompt`]. Held on the `handle_prompt` frame;
/// dropping it flips the probe to "not running" on every exit path, including
/// an abrupt drop of the future itself.
struct InFlightGuard {
    inner: Arc<InFlightInner>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.inner.finished.store(true, Ordering::Release);
    }
}

impl InFlightPrompt {
    /// Start tracking a prompt: returns the map-side probe plus the guard the
    /// prompt task must hold for as long as it executes.
    fn start() -> (Self, InFlightGuard) {
        let inner = Arc::new(InFlightInner {
            token: CancellationToken::new(),
            finished: AtomicBool::new(false),
        });
        (Self { inner: inner.clone() }, InFlightGuard { inner })
    }

    /// Whether the tracked prompt is still executing.
    fn is_running(&self) -> bool {
        !self.inner.finished.load(Ordering::Acquire)
    }

    /// Token that cancels the tracked prompt's runner.
    fn token(&self) -> &CancellationToken {
        &self.inner.token
    }
}

#[derive(Default)]
struct McpSessionResources {
    toolsets: Vec<Arc<dyn Toolset>>,
    cancellations: Vec<Option<rmcp::service::RunningServiceCancellationToken>>,
}

impl Drop for McpSessionResources {
    fn drop(&mut self) {
        for cancellation in &mut self.cancellations {
            if let Some(cancellation) = cancellation.take() {
                cancellation.cancel();
            }
        }
    }
}

/// Maps official ACP v1 session requests to the ADK-Rust Runner and session
/// service. One ACP session maps one-to-one to one ADK-Rust session.
pub struct AcpSessionHandler {
    agent: Arc<dyn Agent>,
    /// Optional per-session agent factory (request `_meta` overrides). When
    /// `None`, `agent` serves every session — unchanged legacy behaviour.
    agent_factory: Option<Arc<dyn AgentFactory>>,
    session_service: Arc<dyn SessionService>,
    app_name: String,
    user_id: String,
    max_sessions: usize,
    /// Process-level map of active sessions, shared by every ACP connection
    /// the server serves (the HTTP transport clones this handler per inbound
    /// connection).
    ///
    /// An entry marks a session as *activated*; it is not a lease owned by
    /// any one connection. `session/load` / `session/resume` arriving for an
    /// idle entry — the previous turn finished, or the activating connection
    /// is gone — rebind the entry to the requesting connection (replacing the
    /// entry tears down the previous entry's MCP server children). A rebind
    /// is refused only while a prompt is still executing; see
    /// [`InFlightPrompt`]. This is what lets per-turn HTTP clients continue a
    /// session from a fresh connection without sending `session/close`.
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    shutdown_token: CancellationToken,
    session_controls: Option<Arc<dyn SessionControls>>,
}

/// Live tool-confirmation handler that bridges ADK tool confirmations to ACP
/// `session/request_permission` **within the same runner turn** (no pause/resume).
///
/// When the runner reaches a tool call that requires confirmation, it calls
/// [`ToolConfirmationHandler::decide`] inline. This sends an ACP
/// `session/request_permission` to the client, awaits the outcome, and returns
/// the mapped decision. The runner then executes (or skips) the tool in the
/// same turn and appends the tool result, so providers never see an assistant
/// `tool_calls` message without a following tool result.
#[derive(Clone)]
struct LivePermissionBridge {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
}

#[async_trait::async_trait]
impl ToolConfirmationHandler for LivePermissionBridge {
    async fn decide(
        &self,
        request: &ToolConfirmationRequest,
    ) -> adk_core::Result<ToolConfirmationDecision> {
        PermissionBridge::request(&self.connection, &self.session_id, request)
            .await
            .map_err(|e| adk_core::AdkError::tool(format!("permission request failed: {e}")))
    }
}

impl std::fmt::Debug for LivePermissionBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LivePermissionBridge").finish_non_exhaustive()
    }
}

/// Live elicitation handler: bridges agent-initiated questions to ACP
/// `elicitation/create` within the same runner turn. Maps [`ElicitationRequest`]
/// (adk-core's provider-agnostic shape) to the ACP `ElicitationFormMode` schema
/// and the `ElicitationAction` response back to [`ElicitationResponse`].
#[derive(Clone)]
struct LiveElicitationBridge {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
}

#[async_trait::async_trait]
impl adk_core::ElicitationHandler for LiveElicitationBridge {
    async fn elicit(
        &self,
        request: &adk_core::ElicitationRequest,
    ) -> adk_core::Result<Option<adk_core::ElicitationResponse>> {
        use agent_client_protocol::schema::v1::{
            CreateElicitationRequest, ElicitationFormMode, ElicitationPropertySchema,
            ElicitationRequestScope, ElicitationSchema, ElicitationScope, EnumOption,
            MultiSelectPropertySchema, StringPropertySchema,
        };

        // Build an ACP object schema with one property per adk-core field.
        let mut schema = ElicitationSchema::new();
        for field in &request.fields {
            let prop = match &field.kind {
                adk_core::ElicitationFieldKind::Text => {
                    ElicitationPropertySchema::String(
                        StringPropertySchema::new().title(field.title.clone()),
                    )
                }
                adk_core::ElicitationFieldKind::SingleSelect { options } => {
                    let one_of: Vec<EnumOption> = options
                        .iter()
                        .map(|o| EnumOption::new(o.value.clone(), o.title.clone()))
                        .collect();
                    ElicitationPropertySchema::String(
                        StringPropertySchema::new().title(field.title.clone()).one_of(one_of),
                    )
                }
                adk_core::ElicitationFieldKind::MultiSelect { options } => {
                    let titled: Vec<EnumOption> = options
                        .iter()
                        .map(|o| EnumOption::new(o.value.clone(), o.title.clone()))
                        .collect();
                    ElicitationPropertySchema::Array(
                        MultiSelectPropertySchema::titled(titled).title(field.title.clone()),
                    )
                }
            };
            schema = schema.property(field.name.clone(), prop, true);
        }
        let scope = ElicitationScope::Request(ElicitationRequestScope::new(
            // request_id is opaque here; use the session id as a stable
            // correlation handle (the client correlates by session already).
            self.session_id.to_string(),
        ));
        let acp_request = CreateElicitationRequest::new(
            ElicitationFormMode::new(scope, schema),
            request.message.clone(),
        );

        let response = self
            .connection
            .send_request(acp_request)
            .block_task()
            .await
            .map_err(|e| adk_core::AdkError::tool(format!("elicitation request failed: {e}")))?;

        Ok(map_elicitation_action(response.action))
    }
}

impl std::fmt::Debug for LiveElicitationBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveElicitationBridge").finish_non_exhaustive()
    }
}

/// Map an ACP [`ElicitationAction`] to adk-core's [`ElicitationResponse`].
fn map_elicitation_action(
    action: agent_client_protocol::schema::v1::ElicitationAction,
) -> Option<adk_core::ElicitationResponse> {
    use agent_client_protocol::schema::v1::ElicitationAction;
    match action {
        ElicitationAction::Accept(accept) => {
            let mut values = std::collections::BTreeMap::new();
            if let Some(content) = accept.content {
                for (k, v) in content {
                    let mapped = match v {
                        agent_client_protocol::schema::v1::ElicitationContentValue::String(s) => {
                            adk_core::ElicitationValue::One(s)
                        }
                        agent_client_protocol::schema::v1::ElicitationContentValue::StringArray(a) => {
                            adk_core::ElicitationValue::Many(a)
                        }
                        agent_client_protocol::schema::v1::ElicitationContentValue::Integer(i) => {
                            adk_core::ElicitationValue::One(i.to_string())
                        }
                        agent_client_protocol::schema::v1::ElicitationContentValue::Number(n) => {
                            adk_core::ElicitationValue::One(n.to_string())
                        }
                        agent_client_protocol::schema::v1::ElicitationContentValue::Boolean(b) => {
                            adk_core::ElicitationValue::One(b.to_string())
                        }
                        _ => continue,
                    };
                    values.insert(k, mapped);
                }
            }
            Some(adk_core::ElicitationResponse { declined: false, values })
        }
        // Decline / Cancel / unknown → treat as declined (no content).
        _ => Some(adk_core::ElicitationResponse { declined: true, values: Default::default() }),
    }
}

impl AcpSessionHandler {
    /// Create a handler from validated server configuration.
    pub fn new(
        config: &AcpServerConfig,
        shutdown_token: CancellationToken,
    ) -> Result<Self, AcpServerError> {
        Ok(Self {
            agent: config.agent.clone(),
            agent_factory: config.agent_factory.clone(),
            session_service: config.session_service.clone(),
            app_name: config.agent_name.clone(),
            user_id: config.user_id.clone(),
            max_sessions: config.max_sessions,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            shutdown_token,
            session_controls: config.session_controls.clone(),
        })
    }

    /// Create an ACP session and its persistent ADK-Rust session.
    pub async fn create_session(
        &self,
        request: NewSessionRequest,
        request_cancellation: RequestCancellation,
    ) -> Result<SessionId, AcpServerError> {
        self.ensure_running()?;
        validate_absolute(&request.cwd, "cwd")?;
        for directory in &request.additional_directories {
            validate_absolute(directory, "additionalDirectories")?;
        }
        // Per-session agent first: a factory failure (invalid `_meta`
        // overrides) rejects the request before any registry or persistence
        // side effect needs unwinding. With a factory configured, **every**
        // new session routes through it — a request without `_meta` is
        // normalized to an empty map (the factory's base composition), so
        // per-cwd concerns (file-domain binding) cover plain sessions too,
        // not only override-carrying ones.
        let empty_meta = serde_json::Map::new();
        let session_agent = self
            .build_session_agent(Some(request.meta.as_ref().unwrap_or(&empty_meta)), &request.cwd)?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let mut state = HashMap::new();
        state.insert(
            CWD_STATE_KEY.to_string(),
            serde_json::Value::String(request.cwd.display().to_string()),
        );
        state.insert(
            ADDITIONAL_DIRS_STATE_KEY.to_string(),
            serde_json::to_value(&request.additional_directories)
                .map_err(|e| AcpServerError::Internal(e.to_string()))?,
        );

        {
            let mut sessions = self.sessions.lock().await;
            if sessions.len() >= self.max_sessions {
                return Err(AcpServerError::MaxSessionsReached(self.max_sessions));
            }
            sessions.insert(
                session_id.clone(),
                SessionState {
                    execution: None,
                    mcp: McpSessionResources::default(),
                    agent: session_agent,
                },
            );
        }

        let mcp = match start_mcp_servers(&request.mcp_servers, &request.cwd, &request_cancellation)
            .await
        {
            Ok(resources) => resources,
            Err(error) => {
                self.sessions.lock().await.remove(&session_id);
                return Err(error);
            }
        };

        if let Err(error) = self
            .session_service
            .create(CreateRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: Some(session_id.clone()),
                state,
            })
            .await
        {
            self.sessions.lock().await.remove(&session_id);
            return Err(AcpServerError::Internal(format!("failed to create session: {error}")));
        }

        if let Some(session) = self.sessions.lock().await.get_mut(&session_id) {
            session.mcp = mcp;
        }
        info!(session_id, "created ACP session");
        Ok(SessionId::new(session_id))
    }

    /// Resume a persisted session and make it active on this ACP connection.
    ///
    /// When the session is already active but idle — its previous turn
    /// finished, or the connection that activated it is gone — the activation
    /// is rebound to this connection instead of being rejected, so a client
    /// never needs `session/close` to recover from an unclean disconnect. A
    /// session with a prompt still executing is refused (see
    /// [`register_session_entry`](Self::register_session_entry)).
    pub async fn resume_session(
        &self,
        session_id: &SessionId,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: Vec<McpServer>,
        request_cancellation: RequestCancellation,
    ) -> Result<(), AcpServerError> {
        self.ensure_running()?;
        validate_absolute(&cwd, "cwd")?;
        for directory in &additional_directories {
            validate_absolute(directory, "additionalDirectories")?;
        }

        let id = session_id.to_string();
        let persisted = self
            .session_service
            .get(GetRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: id.clone(),
                num_recent_events: None,
                after: None,
            })
            .await
            .map_err(|_| AcpServerError::SessionNotFound(id.clone()))?;

        let stored_cwd = persisted
            .state()
            .get(CWD_STATE_KEY)
            .and_then(|value| value.as_str().map(PathBuf::from))
            .unwrap_or_else(|| cwd.clone());
        if stored_cwd != cwd {
            return Err(AcpServerError::MalformedMessage(
                "resumed session cwd does not match its original cwd".into(),
            ));
        }
        self.register_session_entry(&id).await?;
        let mcp = match start_mcp_servers(&mcp_servers, &cwd, &request_cancellation).await {
            Ok(resources) => resources,
            Err(error) => {
                self.sessions.lock().await.remove(&id);
                return Err(error);
            }
        };
        if let Some(session) = self.sessions.lock().await.get_mut(&id) {
            session.mcp = mcp;
        }
        Ok(())
    }

    /// Load a persisted session, reactivate it on this ACP connection, and
    /// replay its stored conversation as ordered `session/update` notifications
    /// before the request completes.
    ///
    /// Reactivation mirrors [`resume_session`](Self::resume_session): the
    /// working directory must be absolute and match the session's stored `cwd`,
    /// the session must exist for the configured application and user, and the
    /// session is registered on the connection (rebinding any idle activation
    /// left behind by a previous connection; a still-executing prompt is
    /// refused) with its MCP servers started. After reactivation, every stored
    /// event is mapped to its `SessionUpdate` variant through the shared
    /// [`ResponseStreamer`] and sent in chronological order.
    #[allow(clippy::too_many_arguments)]
    pub async fn load_session(
        &self,
        session_id: &SessionId,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: Vec<McpServer>,
        request_cancellation: RequestCancellation,
        connection: ConnectionTo<Client>,
        meta: Option<Meta>,
    ) -> Result<(), AcpServerError> {
        self.ensure_running()?;
        validate_absolute(&cwd, "cwd")?;
        for directory in &additional_directories {
            validate_absolute(directory, "additionalDirectories")?;
        }

        let id = session_id.to_string();
        let persisted = self
            .session_service
            .get(GetRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: id.clone(),
                num_recent_events: None,
                after: None,
            })
            .await
            .map_err(|_| AcpServerError::SessionNotFound(id.clone()))?;

        let stored_cwd = persisted
            .state()
            .get(CWD_STATE_KEY)
            .and_then(|value| value.as_str().map(PathBuf::from))
            .unwrap_or_else(|| cwd.clone());
        if stored_cwd != cwd {
            return Err(AcpServerError::MalformedMessage(
                "loaded session cwd does not match its original cwd".into(),
            ));
        }
        // `_meta` on load: a factory-equipped server rebuilds the session's
        // agent from the fresh overrides (clients re-send their pinned model
        // every turn). Absent `_meta` keeps whatever the entry already
        // carries — register_session_entry preserves it across the rebind,
        // so a plain reconnect never implicitly switches model or file
        // domain. Built before registration so a factory failure leaves the
        // session untouched.
        let rebuilt_agent = self.build_session_agent(meta.as_ref(), &cwd)?;
        self.register_session_entry(&id).await?;
        let mcp = match start_mcp_servers(&mcp_servers, &cwd, &request_cancellation).await {
            Ok(resources) => resources,
            Err(error) => {
                self.sessions.lock().await.remove(&id);
                return Err(error);
            }
        };
        if let Some(session) = self.sessions.lock().await.get_mut(&id) {
            session.mcp = mcp;
            if rebuilt_agent.is_some() {
                session.agent = rebuilt_agent;
            }
        }

        // Replay the stored conversation in chronological order. `events().all()`
        // returns events oldest-first, matching the order in which they were
        // originally streamed to the client.
        for event in persisted.events().all() {
            for update in ResponseStreamer::map_event(&event) {
                connection
                    .send_notification(SessionNotification::new(session_id.clone(), update))
                    .map_err(|e| AcpServerError::Transport(e.to_string()))?;
            }
        }
        info!(session_id = %id, "loaded ACP session and replayed history");
        Ok(())
    }

    /// Fork a persisted session into a new independent session and return the
    /// new session identifier.
    ///
    /// The source session is read (never mutated — Property P10): its stored
    /// events and relevant ACP state (`acp:cwd`, `acp:additional_directories`,
    /// `acp:mode`, and every `acp:config:<id>` value) are copied into a freshly
    /// created session id. The copied events are appended in their original
    /// chronological order so the fork's history matches the source's exactly.
    ///
    /// The forked session is registered on this ACP connection (subject to the
    /// max-session limit) and its MCP servers are started, mirroring
    /// [`create_session`](Self::create_session). The returned identifier is a
    /// new UUID distinct from the source.
    pub async fn fork_session(
        &self,
        source_session_id: &SessionId,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: Vec<McpServer>,
        request_cancellation: RequestCancellation,
    ) -> Result<SessionId, AcpServerError> {
        self.ensure_running()?;
        validate_absolute(&cwd, "cwd")?;
        for directory in &additional_directories {
            validate_absolute(directory, "additionalDirectories")?;
        }

        let source_id = source_session_id.to_string();
        let source = self
            .session_service
            .get(GetRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: source_id.clone(),
                num_recent_events: None,
                after: None,
            })
            .await
            .map_err(|_| AcpServerError::SessionNotFound(source_id.clone()))?;

        // Copy the relevant ACP state into the fork. `acp:cwd` and
        // `acp:additional_directories` live only in the initial create-state, so
        // they must be seeded explicitly. `acp:mode` and `acp:config:<id>` are
        // additionally carried by the replayed state-delta events, but seeding
        // them here keeps the fork's state correct even if the source recorded
        // them only through its initial state.
        let mut state = HashMap::new();
        for key in [CWD_STATE_KEY, ADDITIONAL_DIRS_STATE_KEY, modes::MODE_STATE_KEY] {
            if let Some(value) = source.state().get(key) {
                state.insert(key.to_string(), value);
            }
        }
        for (key, value) in source.state().all() {
            if key.starts_with(modes::CONFIG_STATE_KEY_PREFIX) {
                state.insert(key, value);
            }
        }

        // Snapshot the source's events up front so the copy is a point-in-time
        // branch; the source `Session` handle is then dropped without any write.
        let source_events = source.events().all();
        drop(source);

        let new_session_id = uuid::Uuid::new_v4().to_string();

        {
            let mut sessions = self.sessions.lock().await;
            if sessions.len() >= self.max_sessions {
                return Err(AcpServerError::MaxSessionsReached(self.max_sessions));
            }
            // The fork inherits the source's factory-composed agent (if any):
            // branching a conversation must not silently flip its model. A
            // fork request carries no `_meta`, so this is the only continuity
            // the fork gets.
            let inherited_agent =
                sessions.get(&source_id).and_then(|state| state.agent.clone());
            sessions.insert(
                new_session_id.clone(),
                SessionState {
                    execution: None,
                    mcp: McpSessionResources::default(),
                    agent: inherited_agent,
                },
            );
        }

        let mcp = match start_mcp_servers(&mcp_servers, &cwd, &request_cancellation).await {
            Ok(resources) => resources,
            Err(error) => {
                self.sessions.lock().await.remove(&new_session_id);
                return Err(error);
            }
        };

        if let Err(error) = self
            .session_service
            .create(CreateRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: Some(new_session_id.clone()),
                state,
            })
            .await
        {
            self.sessions.lock().await.remove(&new_session_id);
            return Err(AcpServerError::Internal(format!(
                "failed to create forked session: {error}"
            )));
        }

        // Copy the source history into the fork, preserving chronological order.
        for event in source_events {
            if let Err(error) = self.session_service.append_event(&new_session_id, event).await {
                // Best-effort cleanup so a partial fork is not left behind.
                let _ = self
                    .session_service
                    .delete(DeleteRequest {
                        app_name: self.app_name.clone(),
                        user_id: self.user_id.clone(),
                        session_id: new_session_id.clone(),
                    })
                    .await;
                self.sessions.lock().await.remove(&new_session_id);
                return Err(AcpServerError::Internal(format!(
                    "failed to copy session history into fork: {error}"
                )));
            }
        }

        if let Some(session) = self.sessions.lock().await.get_mut(&new_session_id) {
            session.mcp = mcp;
        }
        info!(
            source_session_id = %source_id,
            forked_session_id = %new_session_id,
            "forked ACP session"
        );
        Ok(SessionId::new(new_session_id))
    }

    /// List persisted ACP sessions for the configured ADK application and user.
    pub async fn list_sessions(
        &self,
        request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, AcpServerError> {
        let offset = request
            .cursor
            .as_deref()
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|_| AcpServerError::MalformedMessage("invalid session cursor".into()))?
            .unwrap_or(0);
        let page_size = 50;
        let sessions = self
            .session_service
            .list(ListRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                limit: Some(page_size + 1),
                offset: Some(offset),
            })
            .await
            .map_err(|e| AcpServerError::Internal(format!("failed to list sessions: {e}")))?;

        let has_more = sessions.len() > page_size;
        let mut result = Vec::new();
        for session in sessions.into_iter().take(page_size) {
            let cwd = session
                .state()
                .get(CWD_STATE_KEY)
                .and_then(|value| value.as_str().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("/"));
            if request.cwd.as_ref().is_some_and(|filter| filter != &cwd) {
                continue;
            }
            let additional_directories = session
                .state()
                .get(ADDITIONAL_DIRS_STATE_KEY)
                .and_then(|value| serde_json::from_value(value).ok())
                .unwrap_or_default();
            result.push(
                SessionInfo::new(session.id().to_string(), cwd)
                    .additional_directories(additional_directories)
                    .updated_at(session.last_update_time().to_rfc3339()),
            );
        }
        Ok(ListSessionsResponse::new(result)
            .next_cursor(has_more.then(|| (offset + page_size).to_string())))
    }

    /// Delete persisted session history and release any active execution.
    pub async fn delete_session(&self, session_id: &SessionId) -> Result<(), AcpServerError> {
        if let Some(state) = self.sessions.lock().await.remove(&session_id.to_string())
            && let Some(probe) = state.execution
            && probe.is_running()
        {
            probe.token().cancel();
        }
        self.session_service
            .delete(DeleteRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: session_id.to_string(),
            })
            .await
            .map_err(|_| AcpServerError::SessionNotFound(session_id.to_string()))
    }

    /// Execute one prompt and stream official `session/update` notifications
    /// through the SDK connection before returning the turn stop reason.
    pub async fn handle_prompt(
        &self,
        request: PromptRequest,
        connection: ConnectionTo<Client>,
    ) -> Result<StopReason, AcpServerError> {
        self.ensure_running()?;
        let id = request.session_id.to_string();
        let (cancellation_token, prompt_guard, runtime_toolsets, session_agent) = {
            let mut sessions = self.sessions.lock().await;
            let state =
                sessions.get_mut(&id).ok_or_else(|| AcpServerError::SessionNotFound(id.clone()))?;
            if state.execution.as_ref().is_some_and(InFlightPrompt::is_running) {
                return Err(AcpServerError::Execution(
                    "a prompt is already running in this session".into(),
                ));
            }
            let (probe, guard) = InFlightPrompt::start();
            let token = probe.token().clone();
            state.execution = Some(probe);
            let session_agent = state.agent.clone();
            (token, guard, state.mcp.toolsets.clone(), session_agent)
        };

        let result = self
            .execute_prompt(
                &request,
                connection,
                cancellation_token.clone(),
                runtime_toolsets,
                session_agent,
            )
            .await;

        // Dropping the guard marks the session idle on every exit path —
        // including this future being dropped mid-execution when its
        // connection dies, in which case this line never runs and Drop does
        // the work instead. No map mutation is needed here, which also keeps
        // a finished turn from touching an entry a later load rebound.
        drop(prompt_guard);

        if cancellation_token.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }
        result
    }

    async fn execute_prompt(
        &self,
        request: &PromptRequest,
        connection: ConnectionTo<Client>,
        cancellation_token: CancellationToken,
        runtime_toolsets: Vec<Arc<dyn Toolset>>,
        session_agent: Option<Arc<dyn Agent>>,
    ) -> Result<StopReason, AcpServerError> {
        let content = prompt_content(&request.prompt)?;
        let user_id = UserId::new(self.user_id.clone())
            .map_err(|e| AcpServerError::Execution(e.to_string()))?;
        let session_id = AdkSessionId::new(request.session_id.to_string())
            .map_err(|e| AcpServerError::Execution(e.to_string()))?;

        // Live tool-confirmation handler: when the agent's runner reaches a tool
        // call that requires confirmation (per the agent's tool_confirmation_policy),
        // it calls `decide()` inline within the same turn. We bridge that to an ACP
        // `session/request_permission` request to the client and await the outcome.
        //
        // This replaces the earlier pause/resume loop. Resuming replayed the
        // persisted history — including an assistant turn carrying `tool_calls`
        // whose tool result had not yet been produced — which providers like
        // DeepSeek reject ("tool_calls must be followed by tool messages"). The
        // live handler keeps the whole turn in one runner invocation, so the tool
        // result is produced and appended before any second LLM call.
        let confirmation_handler: Arc<dyn ToolConfirmationHandler> = Arc::new(
            LivePermissionBridge {
                connection: connection.clone(),
                session_id: request.session_id.clone(),
            },
        );
        let elicitation_handler: Arc<dyn adk_core::ElicitationHandler> = Arc::new(
            LiveElicitationBridge {
                connection: connection.clone(),
                session_id: request.session_id.clone(),
            },
        );

        let run_config = RunConfig::builder()
            .runtime_toolsets(runtime_toolsets.clone())
            .tool_confirmation_handler(confirmation_handler)
            .elicitation_handler(elicitation_handler)
            .build();
        let runner = Runner::builder()
            .app_name(&self.app_name)
            .agent(session_agent.unwrap_or_else(|| self.agent.clone()))
            .session_service(self.session_service.clone())
            .cancellation_token(cancellation_token.clone())
            .run_config(run_config)
            .build()
            .map_err(|e| AcpServerError::Execution(format!("failed to create runner: {e}")))?;

        let mut stream = runner
            .run(user_id.clone(), session_id.clone(), content.clone())
            .await
            .map_err(|e| AcpServerError::Execution(format!("runner.run failed: {e}")))?;

        // True once any *partial* (streaming delta) agent text event has been
        // seen this turn. Streaming providers send deltas as partials and then a
        // final event carrying the full reply; once a partial arrived we suppress
        // the final event's text chunk so the client does not receive the whole
        // answer twice.
        let mut saw_partial_agent_text = false;

        loop {
            let result = tokio::select! {
                _ = cancellation_token.cancelled() => return Ok(StopReason::Cancelled),
                result = stream.next() => result,
            };
            let Some(result) = result else {
                break;
            };
            match result {
                Ok(event) => {
                    // With the live confirmation handler wired above, the runner
                    // resolves tool confirmations inline (it calls the handler,
                    // not the interrupt path), so a `tool_confirmation` action
                    // should not surface here. If one ever does, skip it: the
                    // client learns about the request through the native
                    // `session/request_permission` the handler already sent.
                    if event.actions.tool_confirmation.is_some() {
                        continue;
                    }
                    if event.llm_response.partial {
                        saw_partial_agent_text = true;
                    }
                    // If the turn streamed partials, drop the final event's agent
                    // text (it is the accumulated reply = a duplicate).
                    let updates =
                        ResponseStreamer::map_event_filtered(&event, !saw_partial_agent_text);
                    for update in updates {
                        connection
                            .send_notification(SessionNotification::new(
                                request.session_id.clone(),
                                update,
                            ))
                            .map_err(|e| AcpServerError::Transport(e.to_string()))?;
                    }
                }
                Err(error) => {
                    if cancellation_token.is_cancelled() {
                        return Ok(StopReason::Cancelled);
                    }
                    warn!(%error, session_id = %request.session_id, "ACP Runner event failed");
                    return Err(AcpServerError::Execution(error.to_string()));
                }
            }
        }

        Ok(StopReason::EndTurn)
    }

    /// Cancel the prompt currently running in a session.
    pub async fn cancel_session(&self, session_id: &SessionId) {
        if let Some(token) = self
            .sessions
            .lock()
            .await
            .get(&session_id.to_string())
            .and_then(|state| state.execution.as_ref())
            .filter(|probe| probe.is_running())
            .map(|probe| probe.token().clone())
        {
            token.cancel();
        }
    }

    /// Close an active session without deleting its persisted history.
    ///
    /// The session is marked inactive (its MCP server children are torn down);
    /// a later `session/load` / `session/resume` reactivates it with its full
    /// history intact. Closing an idle session that was already rebound by
    /// another connection affects only the current activation.
    pub async fn close_session(&self, session_id: &SessionId) -> Result<(), AcpServerError> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions
            .remove(&session_id.to_string())
            .ok_or_else(|| AcpServerError::SessionNotFound(session_id.to_string()))?;
        if let Some(probe) = state.execution
            && probe.is_running()
        {
            probe.token().cancel();
        }
        Ok(())
    }

    /// The advertised mode state and configuration options for a session,
    /// reflecting any selections persisted in the session state.
    ///
    /// Returns `(None, None)` when the agent exposes no [`SessionControls`].
    /// When controls exist, the provider's declared modes/options supply the
    /// defaults and the persisted `acp:mode` / `acp:config:<id>` values are
    /// layered on top so `session/new`, `session/load`, `session/resume`, and
    /// `session/fork` responses surface the current selection.
    pub async fn session_controls_snapshot(
        &self,
        session_id: &SessionId,
    ) -> (Option<SessionModeState>, Option<Vec<SessionConfigOption>>) {
        let Some(controls) = &self.session_controls else {
            return (None, None);
        };
        // Best effort: absent or unreadable state falls back to provider
        // defaults rather than failing the surrounding request.
        let persisted = self
            .session_service
            .get(GetRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: session_id.to_string(),
                num_recent_events: None,
                after: None,
            })
            .await
            .ok();

        let modes = controls.modes().map(|mut state| {
            if let Some(session) = &persisted
                && let Some(value) = session.state().get(modes::MODE_STATE_KEY)
                && let Some(mode) = value.as_str()
            {
                state.current_mode_id = SessionModeId::new(mode);
            }
            state
        });

        let advertised = controls.config_options();
        let config_options = if advertised.is_empty() {
            None
        } else {
            Some(
                advertised
                    .into_iter()
                    .map(|option| {
                        if let Some(session) = &persisted
                            && let Some(value) = session.state().get(&config_state_key(&option.id))
                            && let Ok(parsed) =
                                serde_json::from_value::<SessionConfigOptionValue>(value.clone())
                        {
                            option_with_current_value(option, &parsed)
                        } else {
                            option
                        }
                    })
                    .collect(),
            )
        };

        (modes, config_options)
    }

    /// Emit the session-activation `session/update` notifications for available
    /// commands and session metadata.
    ///
    /// Called when a session becomes active — after `session/new`,
    /// `session/resume`, `session/load`, and `session/fork` register the session
    /// on the connection. It sends, in order:
    ///
    /// - a [`SessionUpdate::AvailableCommandsUpdate`] **only when** the agent's
    ///   [`SessionControls`] declares commands (Requirement 11.1); an agent that
    ///   declares none emits no such update (Requirement 11.4);
    /// - a [`SessionUpdate::SessionInfoUpdate`] **only when** the session carries
    ///   a title under `acp:title` (Requirement 11.2); no title means no
    ///   update (Requirement 11.4).
    ///
    /// The per-update helpers ([`emit_available_commands`](Self::emit_available_commands)
    /// and `emit_session_info`) are reusable so a
    /// future command-set or metadata change trigger emits the same
    /// notifications outside activation.
    pub async fn emit_session_activation_updates(
        &self,
        session_id: &SessionId,
        connection: &ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        self.emit_available_commands(session_id, connection).await?;
        self.emit_session_info(session_id, connection).await?;
        Ok(())
    }

    /// Emit an [`AvailableCommandsUpdate`] for the agent's declared commands.
    ///
    /// Sends nothing when the agent exposes no [`SessionControls`] or declares
    /// no commands, preserving capability accuracy for command-less agents.
    /// Reusable by a future command-set change trigger.
    pub async fn emit_available_commands(
        &self,
        session_id: &SessionId,
        connection: &ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        let commands = self
            .session_controls
            .as_ref()
            .map(|controls| controls.available_commands())
            .unwrap_or_default();
        if commands.is_empty() {
            return Ok(());
        }
        connection
            .send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
            ))
            .map_err(|e| AcpServerError::Transport(e.to_string()))
    }

    /// Emit a [`SessionInfoUpdate`] carrying the session's title, if it has one.
    ///
    /// The title is read from the `acp:title` session-state key. Sends nothing
    /// when no title is recorded, so a session with no metadata to report emits
    /// no update. Reusable by a metadata change trigger.
    async fn emit_session_info(
        &self,
        session_id: &SessionId,
        connection: &ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        let Some(title) = self.session_title(session_id).await else {
            return Ok(());
        };
        connection
            .send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title)),
            ))
            .map_err(|e| AcpServerError::Transport(e.to_string()))
    }

    /// Read the persisted title (`acp:title`) for a session, if any.
    ///
    /// Best-effort: an absent or unreadable session yields `None` rather than
    /// failing the surrounding request.
    async fn session_title(&self, session_id: &SessionId) -> Option<String> {
        let persisted = self
            .session_service
            .get(GetRequest {
                app_name: self.app_name.clone(),
                user_id: self.user_id.clone(),
                session_id: session_id.to_string(),
                num_recent_events: None,
                after: None,
            })
            .await
            .ok()?;
        persisted.state().get(TITLE_STATE_KEY).and_then(|value| value.as_str().map(str::to_string))
    }

    /// Set the session's human-readable title and notify the client.
    ///
    /// Persists the title under `acp:title` (surviving load / resume / fork) and
    /// sends a [`SessionUpdate::SessionInfoUpdate`] reflecting the change
    /// (Requirement 11.2). This is the metadata change trigger that reuses the
    /// same notification the activation path emits.
    pub async fn set_session_title(
        &self,
        session_id: &SessionId,
        title: impl Into<String>,
        connection: &ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        self.ensure_running()?;
        let title = title.into();
        self.write_session_state(
            &session_id.to_string(),
            TITLE_STATE_KEY,
            serde_json::Value::String(title.clone()),
        )
        .await?;
        connection
            .send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title)),
            ))
            .map_err(|e| AcpServerError::Transport(e.to_string()))?;
        info!(session_id = %session_id, "set ACP session title");
        Ok(())
    }

    /// Set the current session mode and notify the client.
    ///
    /// Validates `mode_id` against the modes advertised by the agent's
    /// [`SessionControls`]. On success the selection is persisted under
    /// `acp:mode` and a [`SessionUpdate::CurrentModeUpdate`] notification is
    /// sent. An unknown mode identifier (including the case where the agent
    /// advertises no modes) returns a descriptive error and leaves the current
    /// mode unchanged.
    pub async fn set_mode(
        &self,
        session_id: &SessionId,
        mode_id: &SessionModeId,
        connection: ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        self.ensure_running()?;
        let mode_state = self.session_controls.as_ref().and_then(|controls| controls.modes());
        let advertised =
            mode_state.as_ref().is_some_and(|state| mode_is_advertised(state, mode_id));
        if !advertised {
            return Err(AcpServerError::MalformedMessage(format!(
                "unknown session mode: {mode_id}"
            )));
        }
        self.write_session_state(
            &session_id.to_string(),
            modes::MODE_STATE_KEY,
            serde_json::Value::String(mode_id.to_string()),
        )
        .await?;
        connection
            .send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id.clone())),
            ))
            .map_err(|e| AcpServerError::Transport(e.to_string()))?;
        info!(session_id = %session_id, mode_id = %mode_id, "set ACP session mode");
        Ok(())
    }

    /// Set a session configuration option value and notify the client.
    ///
    /// Validates `config_id` and `value` against the options advertised by the
    /// agent's [`SessionControls`]. On success the value is persisted under
    /// `acp:config:<id>` and a [`SessionUpdate::ConfigOptionUpdate`] carrying
    /// the full current option set is sent. An unknown option or a value whose
    /// shape does not match the option returns a descriptive error and leaves
    /// the option unchanged.
    pub async fn set_config_option(
        &self,
        session_id: &SessionId,
        config_id: &SessionConfigId,
        value: SessionConfigOptionValue,
        connection: ConnectionTo<Client>,
    ) -> Result<(), AcpServerError> {
        self.ensure_running()?;
        let options = self
            .session_controls
            .as_ref()
            .map(|controls| controls.config_options())
            .unwrap_or_default();
        let option = options.iter().find(|option| &option.id == config_id).ok_or_else(|| {
            AcpServerError::MalformedMessage(format!("unknown configuration option: {config_id}"))
        })?;
        if !config_value_is_valid(option, &value) {
            return Err(AcpServerError::MalformedMessage(format!(
                "invalid value for configuration option: {config_id}"
            )));
        }
        let encoded =
            serde_json::to_value(&value).map_err(|e| AcpServerError::Internal(e.to_string()))?;
        self.write_session_state(&session_id.to_string(), &config_state_key(config_id), encoded)
            .await?;
        // Report the full option set with current values reflecting the write.
        let (_, config_options) = self.session_controls_snapshot(session_id).await;
        connection
            .send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                    config_options.unwrap_or_default(),
                )),
            ))
            .map_err(|e| AcpServerError::Transport(e.to_string()))?;
        info!(session_id = %session_id, config_id = %config_id, "set ACP session config option");
        Ok(())
    }

    /// Compose a per-session agent via the configured [`AgentFactory`], if any.
    ///
    /// Only an explicit `_meta` map (`Some`) reaches the factory here — but
    /// `create_session` normalizes an absent `_meta` to an empty map before
    /// calling, so on the create path a factory-equipped server composes
    /// **every** session (empty map = the factory's base composition plus
    /// whatever cwd binding it applies). `load_session` passes its `Option`
    /// through verbatim: a load without `_meta` keeps whatever agent the
    /// entry already carries (register_session_entry preserves it across the
    /// rebind). The session `cwd` rides along verbatim (the request-level
    /// `validate_absolute` already ran). A factory error is loud: the session
    /// request fails instead of degrading to the config-level agent.
    fn build_session_agent(
        &self,
        meta: Option<&Meta>,
        cwd: &Path,
    ) -> Result<Option<Arc<dyn Agent>>, AcpServerError> {
        let (Some(factory), Some(map)) = (self.agent_factory.as_ref(), meta) else {
            return Ok(None);
        };
        let agent = factory
            .build(map, cwd)
            .map_err(|e| AcpServerError::MalformedMessage(format!("session _meta rejected: {e}")))?;
        Ok(Some(agent))
    }

    /// Register a session as active in the process-level map, rebinding idle
    /// entries left behind by earlier connections.
    ///
    /// An entry whose prompt is still executing ([`InFlightPrompt::is_running`])
    /// blocks the registration: two connections may not run turns on one
    /// session at the same time, and the in-flight guard in
    /// [`handle_prompt`](Self::handle_prompt) keeps every other connection
    /// serial with the running turn too. An idle entry — the previous turn
    /// finished, or the activating connection went away — is replaced in
    /// place; dropping the old entry tears down its MCP server children, which
    /// is exactly what a fresh `session/load` / `session/resume` wants. A
    /// rebinding registration does not grow the map, so `max_sessions` is
    /// checked only for genuinely new entries.
    async fn register_session_entry(&self, id: &str) -> Result<(), AcpServerError> {
        let mut sessions = self.sessions.lock().await;
        if sessions
            .get(id)
            .and_then(|state| state.execution.as_ref())
            .is_some_and(InFlightPrompt::is_running)
        {
            return Err(AcpServerError::Execution(
                "the session is already active on this ACP connection: a prompt is running"
                    .into(),
            ));
        }
        let rebinding = sessions.contains_key(id);
        if !rebinding && sessions.len() >= self.max_sessions {
            return Err(AcpServerError::MaxSessionsReached(self.max_sessions));
        }
        // A rebind replaces the entry wholesale; carry any factory-composed
        // agent over so a session pinned on `session/new` keeps its model on
        // a later load/resume that re-registers it.
        let preserved_agent = sessions.get(id).and_then(|state| state.agent.clone());
        sessions.insert(
            id.to_string(),
            SessionState {
                execution: None,
                mcp: McpSessionResources::default(),
                agent: preserved_agent,
            },
        );
        Ok(())
    }

    /// Persist a single session-state key by appending a state-delta event.
    ///
    /// The event carries no content, so it produces no `session/update` on
    /// history replay; it exists only to record the state change under the
    /// session's persisted state (surviving load / resume / fork).
    async fn write_session_state(
        &self,
        session_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), AcpServerError> {
        let mut event = adk_core::Event::new(session_id);
        event.actions.state_delta.insert(key.to_string(), value);
        self.session_service
            .append_event(session_id, event)
            .await
            .map_err(|e| AcpServerError::Internal(format!("failed to persist session state: {e}")))
    }

    /// Number of active sessions on the connection.
    pub async fn active_session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Cancel active work and release all connection-scoped session state.
    pub async fn drain_sessions(&self, _timeout: std::time::Duration) {
        let mut sessions = self.sessions.lock().await;
        for state in sessions.values() {
            if let Some(probe) = &state.execution
                && probe.is_running()
            {
                probe.token().cancel();
            }
        }
        sessions.clear();
    }

    fn ensure_running(&self) -> Result<(), AcpServerError> {
        if self.shutdown_token.is_cancelled() { Err(AcpServerError::ShuttingDown) } else { Ok(()) }
    }
}

async fn start_mcp_servers(
    servers: &[McpServer],
    cwd: &Path,
    cancellation: &RequestCancellation,
) -> Result<McpSessionResources, AcpServerError> {
    validate_mcp_servers(servers)?;
    let mut resources = McpSessionResources {
        toolsets: Vec::with_capacity(servers.len()),
        cancellations: Vec::with_capacity(servers.len()),
    };
    for server in servers {
        match server {
            McpServer::Stdio(config) => {
                start_stdio_mcp_server(config, cwd, cancellation, &mut resources).await?;
            }
            #[cfg(feature = "mcp-http")]
            McpServer::Http(config) => {
                start_http_mcp_server(config, cancellation, &mut resources).await?;
            }
            unsupported => {
                return Err(AcpServerError::MalformedMessage(format!(
                    "this ACP agent does not support the {} MCP transport; only stdio{} was advertised",
                    transport_kind(unsupported),
                    if cfg!(feature = "mcp-http") { " and HTTP" } else { "" },
                )));
            }
        }
    }
    Ok(resources)
}

/// MCP transport name for error messages (`Sse` remains unsupported).
fn transport_kind(server: &McpServer) -> &'static str {
    match server {
        McpServer::Stdio(_) => "stdio",
        McpServer::Http(_) => "HTTP",
        McpServer::Sse(_) => "SSE",
        _ => "unknown",
    }
}

/// Start one stdio MCP server (spawn a child process in `cwd`) and attach its
/// toolset to the session resources.
async fn start_stdio_mcp_server(
    config: &agent_client_protocol::schema::v1::McpServerStdio,
    cwd: &Path,
    cancellation: &RequestCancellation,
    resources: &mut McpSessionResources,
) -> Result<(), AcpServerError> {
    if !config.command.is_absolute() {
        return Err(AcpServerError::MalformedMessage(format!(
            "MCP server '{}' command must be an absolute path",
            config.name
        )));
    }

    let mut command = tokio::process::Command::new(&config.command);
    command.args(&config.args).current_dir(cwd);
    for variable in &config.env {
        command.env(&variable.name, &variable.value);
    }
    let transport = TokioChildProcess::new(command).map_err(|error| {
        AcpServerError::Execution(format!(
            "failed to start MCP server '{}': {error}",
            config.name
        ))
    })?;
    let startup = tokio::time::timeout(MCP_STARTUP_TIMEOUT, ().serve(transport));
    let client = tokio::select! {
        _ = cancellation.cancelled() => {
            return Err(AcpServerError::Execution(
                "ACP session creation was cancelled while starting MCP servers".into(),
            ));
        }
        result = startup => result,
    }
    .map_err(|_| {
        AcpServerError::Execution(format!(
            "MCP server '{}' did not initialize within {} seconds",
            config.name,
            MCP_STARTUP_TIMEOUT.as_secs()
        ))
    })?
    .map_err(|error| {
        AcpServerError::Execution(format!(
            "failed to initialize MCP server '{}': {error}",
            config.name
        ))
    })?;
    let toolset = McpToolset::new(client).with_name(format!("acp:{}", config.name));
    let cancellation = toolset.cancellation_token().await;
    resources.cancellations.push(Some(cancellation));
    resources.toolsets.push(Arc::new(toolset));
    Ok(())
}

/// Start one streamable-HTTP MCP server (feature `mcp-http`) and attach its
/// toolset to the session resources. Unlike stdio there is no child process;
/// the remote endpoint is contacted with the client-supplied headers.
#[cfg(feature = "mcp-http")]
async fn start_http_mcp_server(
    config: &agent_client_protocol::schema::v1::McpServerHttp,
    cancellation: &RequestCancellation,
    resources: &mut McpSessionResources,
) -> Result<(), AcpServerError> {
    let mut builder = McpHttpClientBuilder::new(&config.url).timeout(MCP_STARTUP_TIMEOUT);
    for header in &config.headers {
        builder = builder.header(&header.name, &header.value);
    }
    let startup = tokio::time::timeout(MCP_STARTUP_TIMEOUT, builder.connect());
    let connect = tokio::select! {
        _ = cancellation.cancelled() => {
            return Err(AcpServerError::Execution(
                "ACP session creation was cancelled while starting MCP servers".into(),
            ));
        }
        result = startup => result,
    }
    .map_err(|_| {
        AcpServerError::Execution(format!(
            "MCP server '{}' did not initialize within {} seconds",
            config.name,
            MCP_STARTUP_TIMEOUT.as_secs()
        ))
    })?
    .map_err(|error| {
        AcpServerError::Execution(format!(
            "failed to initialize MCP server '{}': {error}",
            config.name
        ))
    })?;
    let toolset = connect.with_name(format!("acp:{}", config.name));
    let cancellation = toolset.cancellation_token().await;
    resources.cancellations.push(Some(cancellation));
    resources.toolsets.push(Arc::new(toolset));
    Ok(())
}

fn validate_mcp_servers(servers: &[McpServer]) -> Result<(), AcpServerError> {
    const MAX_MCP_SERVERS: usize = 16;
    if servers.len() > MAX_MCP_SERVERS {
        return Err(AcpServerError::MalformedMessage(format!(
            "at most {MAX_MCP_SERVERS} MCP servers may be attached to one ACP session"
        )));
    }
    let mut names = std::collections::HashSet::new();
    for server in servers {
        let (name, env): (&str, &[_]) = match server {
            McpServer::Stdio(config) => (config.name.as_str(), &config.env[..]),
            #[cfg(feature = "mcp-http")]
            McpServer::Http(config) => {
                if config.url.trim().is_empty() {
                    return Err(AcpServerError::MalformedMessage(format!(
                        "MCP server '{}' has an empty url",
                        config.name
                    )));
                }
                (config.name.as_str(), &[])
            }
            unsupported => {
                return Err(AcpServerError::MalformedMessage(format!(
                    "this ACP agent does not support the {} MCP transport",
                    transport_kind(unsupported)
                )));
            }
        };
        let config_name = name;
        if name.trim().is_empty() {
            return Err(AcpServerError::MalformedMessage(
                "MCP server names cannot be empty".into(),
            ));
        }
        if !names.insert(config_name) {
            return Err(AcpServerError::MalformedMessage(format!(
                "duplicate MCP server name: {}",
                config_name
            )));
        }
        let mut environment_names = std::collections::HashSet::new();
        for variable in env {
            if variable.name.trim().is_empty() {
                return Err(AcpServerError::MalformedMessage(format!(
                    "MCP server '{}' has an empty environment variable name",
                    config_name
                )));
            }
            if !environment_names.insert(variable.name.as_str()) {
                return Err(AcpServerError::MalformedMessage(format!(
                    "MCP server '{}' repeats environment variable '{}'",
                    config_name, variable.name
                )));
            }
        }
    }
    Ok(())
}

fn validate_absolute(path: &Path, field: &str) -> Result<(), AcpServerError> {
    if path.is_absolute() {
        Ok(())
    } else {
        Err(AcpServerError::MalformedMessage(format!("{field} must be an absolute path")))
    }
}

fn prompt_content(blocks: &[ContentBlock]) -> Result<Content, AcpServerError> {
    let mut content = Content::new("user");
    for block in blocks {
        // Text, image, audio, resource-link, and embedded-resource content all
        // flow through the shared mapping. `block_to_part` maps image/audio to
        // `Part::InlineData` (advertised via the `image`/`audio` prompt
        // capabilities) and returns an error for genuinely unsupported content
        // types, which we surface as a descriptive malformed-message error.
        let part = crate::content::block_to_part(block)
            .map_err(|error| AcpServerError::MalformedMessage(error.to_string()))?;
        content.parts.push(part);
    }
    if content.parts.is_empty() {
        return Err(AcpServerError::MalformedMessage("prompt must contain content".into()));
    }
    Ok(content)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use adk_core::{Event, EventStream, InvocationContext, Result as AdkResult};
    use agent_client_protocol::Agent as AcpAgentRole;
    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::schema::v1::{
        AudioContent, CloseSessionRequest, EmbeddedResource as AcpEmbeddedResource,
        EmbeddedResourceResource, EnvVariable, ImageContent, InitializeRequest,
        LoadSessionRequest, McpServerStdio, NewSessionRequest, ResumeSessionRequest, TextContent,
        TextResourceContents as AcpTextResourceContents,
    };
    use agent_client_protocol::Channel;
    use base64::{Engine as _, engine::general_purpose};
    use tokio::sync::Notify;

    use super::super::capabilities::{AgentCapabilities, CapabilitiesBuilder};
    use super::super::config::AcpServerConfigBuilder;
    use super::super::test_helpers::mock_agent_and_session;
    use super::super::transport::stdio::serve_connection;
    use super::*;

    /// The advertised prompt capabilities must correspond exactly to the content
    /// types the prompt handler accepts (Capability_Accuracy). The handler
    /// accepts text, image, audio, resource-link, and embedded-resource content,
    /// so `embedded_context`, `image`, and `audio` are all advertised and all of
    /// those content types are accepted by [`prompt_content`].
    ///
    /// **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 13.1, 13.2**
    #[test]
    fn advertised_prompt_capabilities_match_accepted_content_types() {
        let (agent, session_service) = mock_agent_and_session();
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_service)
            .build()
            .expect("valid config");
        let caps = CapabilitiesBuilder::build(&config);

        // embedded_context advertised <=> handler accepts embedded-resource content.
        assert!(caps.prompt_capabilities.embedded_context, "embedded_context must be advertised");
        let embedded = ContentBlock::Resource(AcpEmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(
                AcpTextResourceContents::new("fn main() {}", "file:///main.rs")
                    .mime_type(Some("text/x-rust".to_string())),
            ),
        ));
        let text = ContentBlock::Text(TextContent::new("hello"));
        prompt_content(&[text, embedded])
            .expect("handler accepts text and embedded-resource content");

        // image advertised <=> handler accepts image content, mapped to InlineData.
        assert!(caps.prompt_capabilities.image, "image must be advertised");
        let raw = vec![0x89, 0x50, 0x4E, 0x47];
        let encoded = general_purpose::STANDARD.encode(&raw);
        let image = ContentBlock::Image(ImageContent::new(encoded, "image/png"));
        let content = prompt_content(&[image]).expect("handler accepts image content");
        assert!(
            matches!(&content.parts[0], adk_core::Part::InlineData { mime_type, data, .. }
                if mime_type == "image/png" && data == &raw),
            "image content must map to Part::InlineData"
        );

        // audio advertised <=> handler accepts audio content, mapped to InlineData.
        assert!(caps.prompt_capabilities.audio, "audio must be advertised");
        let raw_audio = vec![1u8, 2, 3, 4, 5];
        let encoded_audio = general_purpose::STANDARD.encode(&raw_audio);
        let audio = ContentBlock::Audio(AudioContent::new(encoded_audio, "audio/mp3"));
        let content = prompt_content(&[audio]).expect("handler accepts audio content");
        assert!(
            matches!(&content.parts[0], adk_core::Part::InlineData { mime_type, data, .. }
                if mime_type == "audio/mp3" && data == &raw_audio),
            "audio content must map to Part::InlineData"
        );
    }

    /// Content that cannot be mapped to a `Part` (here, a malformed base64 image
    /// payload) is surfaced as a descriptive `MalformedMessage` error rather than
    /// being silently accepted or truncated.
    ///
    /// **Validates: Requirements 6.4, 13.3**
    #[test]
    fn unmappable_prompt_content_is_rejected_with_descriptive_error() {
        let bad_image = ContentBlock::Image(ImageContent::new("not*valid*base64", "image/png"));
        let error = prompt_content(&[bad_image]).expect_err("malformed content must be rejected");
        assert!(
            matches!(error, AcpServerError::MalformedMessage(_)),
            "expected a descriptive MalformedMessage error, got {error:?}"
        );
    }

    /// An empty prompt (no content blocks) is rejected with a descriptive error,
    /// preserving the existing empty-prompt contract.
    #[test]
    fn empty_prompt_is_rejected() {
        let error = prompt_content(&[]).expect_err("empty prompt must be rejected");
        assert!(
            matches!(error, AcpServerError::MalformedMessage(message) if message.contains("content"))
        );
    }

    #[test]
    fn validates_session_mcp_configuration_before_process_start() {
        let duplicate_names = vec![
            McpServer::Stdio(McpServerStdio::new("tools", "/bin/echo")),
            McpServer::Stdio(McpServerStdio::new("tools", "/bin/echo")),
        ];
        assert!(
            validate_mcp_servers(&duplicate_names)
                .expect_err("duplicate names")
                .to_string()
                .contains("duplicate MCP server name")
        );

        let duplicate_environment = vec![McpServer::Stdio(
            McpServerStdio::new("tools", "/bin/echo")
                .env(vec![EnvVariable::new("TOKEN", "one"), EnvVariable::new("TOKEN", "two")]),
        )];
        assert!(
            validate_mcp_servers(&duplicate_environment)
                .expect_err("duplicate environment")
                .to_string()
                .contains("repeats environment variable")
        );
    }

    /// An agent whose turn hangs until released, so a test can hold a prompt
    /// in flight while another connection arrives.
    struct GatedAgent {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl Agent for GatedAgent {
        fn name(&self) -> &str {
            "gated-agent"
        }

        fn description(&self) -> &str {
            "Hangs each turn until released"
        }

        fn sub_agents(&self) -> &[Arc<dyn Agent>] {
            &[]
        }

        async fn run(&self, _ctx: Arc<dyn InvocationContext>) -> AdkResult<EventStream> {
            self.started.notify_one();
            self.release.notified().await;
            let mut event = Event::new("gated-agent");
            event.set_content(Content::new("model").with_text("released"));
            Ok(Box::pin(futures::stream::once(async move { Ok(event) })))
        }
    }

    /// Client-side work for [`spawn_acp_connection`]: receives the client's
    /// connection and returns the future driving it. Boxed so the connection
    /// spawner stays one concrete type regardless of what a test captures.
    type ClientWork = Box<
        dyn FnOnce(
                ConnectionTo<AcpAgentRole>,
            ) -> futures::future::BoxFuture<'static, Result<(), agent_client_protocol::Error>>
            + Send,
    >;

    /// Task handle for one half (server or client side) of a spawned
    /// in-memory ACP connection.
    type ConnectionTask = tokio::task::JoinHandle<Result<(), agent_client_protocol::Error>>;

    /// Spawn one fresh in-memory ACP connection (official client + agent
    /// component pair) sharing `handler` — the per-connection shape the HTTP
    /// transport's factory produces, so "call this twice" means "two clients,
    /// two connections, one process-level handler".
    ///
    /// Returns the server-side and client-side tasks. Awaiting the client task
    /// asserts the whole connection succeeded; aborting the server task
    /// simulates the connection dying (the SDK drops the connection's task
    /// actor, and with it any prompt still executing).
    fn spawn_acp_connection(
        handler: Arc<AcpSessionHandler>,
        capabilities: AgentCapabilities,
        client_work: ClientWork,
    ) -> (ConnectionTask, ConnectionTask) {
        let (server_channel, client_channel) = Channel::duplex();
        let server = tokio::spawn(serve_connection(
            handler,
            capabilities,
            "test-agent".into(),
            "Session rebind test agent".into(),
            server_channel,
        ));
        let client = tokio::spawn(async move {
            Client
                .builder()
                .connect_with(client_channel, move |connection| async move {
                    client_work(connection).await
                })
                .await
        });
        (server, client)
    }

    /// Initialize a session on a fresh connection. Shared by the rebind tests.
    async fn initialize_and_create_session(
        connection: &ConnectionTo<AcpAgentRole>,
    ) -> Result<(SessionId, PathBuf), agent_client_protocol::Error> {
        connection
            .send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;
        let cwd = std::env::current_dir().expect("absolute cwd");
        let session = connection
            .send_request(NewSessionRequest::new(cwd.clone()))
            .block_task()
            .await?;
        Ok((session.session_id, cwd))
    }

    /// **Session activation rebind**: once a session's turn has finished, a
    /// *new* connection can `session/load` it and a third connection can
    /// `session/resume` it — with no `session/close` in between. The idle
    /// activation left by connection 1 is rebound, not rejected. With
    /// `max_sessions = 1`, this also pins that rebinding does not consume a
    /// new session slot.
    #[tokio::test]
    async fn idle_session_rebinds_to_new_connections_without_close() {
        let (agent, session_service) = mock_agent_and_session();
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_service)
            .agent_name("test-agent")
            .max_sessions(1)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        // Connection 1: create the session, run one turn to completion. No
        // session/close is ever sent and the connection is simply dropped.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (server1, client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    let (session_id, cwd) = initialize_and_create_session(&connection).await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("hello"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    let _ = session_tx.send((session_id, cwd));
                    Ok(())
                })
            }),
        );
        // The oneshot fires only after the prompt response arrived, and the
        // in-flight probe is released before that response is sent, so the
        // session is guaranteed idle from here on.
        let (session_id, cwd) = session_rx.await.expect("connection 1 completed its turn");
        server1.abort();
        let _ = server1.await;
        let _ = client1.await;

        // Connection 2 (a fresh connection, like the next per-turn HTTP
        // request): loading the same session must succeed.
        let load_session_id = session_id.clone();
        let load_cwd = cwd.clone();
        let (_server2, client2) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(load_session_id, load_cwd))
                        .block_task()
                        .await?;
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 load completed before timeout")
            .expect("official ACP client completed")
            .expect("connection 2 load succeeded without session/close");

        // Connection 3: the rebound (still idle) entry rebinds again.
        let (_server3, client3) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(ResumeSessionRequest::new(session_id, cwd))
                        .block_task()
                        .await?;
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client3)
            .await
            .expect("connection 3 resume completed before timeout")
            .expect("official ACP client completed")
            .expect("connection 3 resume succeeded without session/close");
    }

    /// **Busy refusal**: while a prompt is executing on one connection,
    /// `session/load` and `session/resume` from another connection are both
    /// refused. Once the turn finishes, the very next load succeeds.
    #[tokio::test]
    async fn load_and_resume_are_refused_only_while_a_prompt_is_running() {
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let agent: Arc<dyn Agent> = Arc::new(GatedAgent {
            started: started.clone(),
            release: release.clone(),
        });
        let session_service: Arc<dyn SessionService> =
            Arc::new(adk_session::InMemorySessionService::new());
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_service)
            .agent_name("gated-agent")
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        // Connection 1: start a prompt that hangs inside the agent.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (_server1, client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    let (session_id, cwd) = initialize_and_create_session(&connection).await?;
                    let _ = session_tx.send((session_id.clone(), cwd));
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new("hang"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        let (session_id, cwd) = session_rx.await.expect("connection 1 created the session");
        // The agent only starts once the prompt's in-flight probe is set, so
        // this notification means the session is genuinely busy.
        started.notified().await;

        // Connection 2: both load and resume are refused while it runs.
        let busy_session_id = session_id.clone();
        let busy_cwd = cwd.clone();
        let (_server2, client2) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let refused_load = connection
                        .send_request(LoadSessionRequest::new(
                            busy_session_id.clone(),
                            busy_cwd.clone(),
                        ))
                        .block_task()
                        .await
                        .expect_err(
                            "load must be refused while the session's prompt is running",
                        );
                    assert!(
                        refused_load.to_string().contains("already active"),
                        "expected an already-active refusal, got: {refused_load}"
                    );
                    let refused_resume = connection
                        .send_request(ResumeSessionRequest::new(busy_session_id, busy_cwd))
                        .block_task()
                        .await
                        .expect_err(
                            "resume must be refused while the session's prompt is running",
                        );
                    assert!(
                        refused_resume.to_string().contains("already active"),
                        "expected an already-active refusal, got: {refused_resume}"
                    );
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 refusals completed before timeout")
            .expect("official ACP client completed")
            .expect("connection 2 saw both refusals");

        // Finish the turn, then the same load from a third connection works.
        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(5), client1)
            .await
            .expect("connection 1 turn completed before timeout")
            .expect("official ACP client completed")
            .expect("connection 1 turn succeeded");

        let (_server3, client3) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(session_id, cwd))
                        .block_task()
                        .await?;
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client3)
            .await
            .expect("connection 3 load completed before timeout")
            .expect("official ACP client completed")
            .expect("load must succeed once the turn finished");
    }

    /// **Explicit close**: `session/close` still releases the activation
    /// without deleting history — a new connection can then load the session
    /// (its stored turn is replayed) and run a full follow-up turn.
    #[tokio::test]
    async fn close_then_load_from_a_new_connection_succeeds() {
        let (agent, session_service) = mock_agent_and_session();
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_service)
            .agent_name("test-agent")
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        // Connection 1: create the session and complete one turn.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (server1, client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    let (session_id, cwd) = initialize_and_create_session(&connection).await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("hello"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    let _ = session_tx.send((session_id, cwd));
                    Ok(())
                })
            }),
        );
        let (session_id, cwd) = session_rx.await.expect("connection 1 completed its turn");
        server1.abort();
        let _ = server1.await;
        let _ = client1.await;

        // Connection 2: close the session, then load it from the same new
        // connection. The load must succeed, replay the stored turn, and
        // leave the session able to run another full turn.
        let updates = Arc::new(Mutex::new(Vec::<SessionUpdate>::new()));
        let updates_for_client = updates.clone();
        let (server2, client2) = {
            let (server_channel, client_channel) = Channel::duplex();
            let server = tokio::spawn(serve_connection(
                handler,
                capabilities,
                "test-agent".into(),
                "Close-then-load test agent".into(),
                server_channel,
            ));
            let client = tokio::spawn(
                Client
                    .builder()
                    .on_receive_notification(
                        async move |notification: SessionNotification,
                                    _connection: ConnectionTo<AcpAgentRole>| {
                            updates_for_client
                                .lock()
                                .expect("updates lock")
                                .push(notification.update);
                            Ok(())
                        },
                        agent_client_protocol::on_receive_notification!(),
                    )
                    .connect_with(client_channel, move |connection: ConnectionTo<AcpAgentRole>| async move {
                        connection
                            .send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await?;
                        connection
                            .send_request(CloseSessionRequest::new(session_id.clone()))
                            .block_task()
                            .await?;
                        connection
                            .send_request(LoadSessionRequest::new(session_id.clone(), cwd))
                            .block_task()
                            .await?;
                        let follow_up = connection
                            .send_request(PromptRequest::new(
                                session_id,
                                vec![ContentBlock::Text(TextContent::new("again"))],
                            ))
                            .block_task()
                            .await?;
                        assert_eq!(follow_up.stop_reason, StopReason::EndTurn);
                        Ok(())
                    }),
            );
            (server, client)
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 close+load completed before timeout")
            .expect("official ACP client completed")
            .expect("connection 2 close+load+prompt succeeded");
        server2.abort();
        let _ = server2.await;

        // One replayed chunk from the load plus one live chunk from the
        // follow-up turn: close deleted neither the history nor the session's
        // ability to keep working.
        let replayed_and_live = updates
            .lock()
            .expect("updates lock")
            .iter()
            .filter(|update| {
                matches!(
                    update,
                    SessionUpdate::AgentMessageChunk(chunk)
                        if matches!(&chunk.content, ContentBlock::Text(text)
                            if text.text == "mock response")
                )
            })
            .count();
        assert!(
            replayed_and_live >= 2,
            "expected the replayed turn and the follow-up turn, got {replayed_and_live} chunks"
        );
    }

    /// **Abandoned connection**: a prompt still executing when its connection
    /// dies (here: the server task is killed mid-turn, which drops the prompt
    /// future exactly like a transport failure does) must stop blocking
    /// `session/load` — the drop-guarded in-flight probe releases the session
    /// without any cleanup code running. The new connection can then load and
    /// keep using the session.
    #[tokio::test]
    async fn abandoned_connection_mid_prompt_releases_the_session() {
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let agent: Arc<dyn Agent> = Arc::new(GatedAgent {
            started: started.clone(),
            release: release.clone(),
        });
        let session_service: Arc<dyn SessionService> =
            Arc::new(adk_session::InMemorySessionService::new());
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_service)
            .agent_name("gated-agent")
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        // Connection 1: start a hanging prompt, then kill the whole
        // connection while it runs.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (server1, _client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    let (session_id, cwd) = initialize_and_create_session(&connection).await?;
                    let _ = session_tx.send((session_id.clone(), cwd));
                    let _prompt = connection
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new("hang"))],
                        ))
                        .block_task()
                        .await;
                    Ok(())
                })
            }),
        );
        let (session_id, cwd) = session_rx.await.expect("connection 1 created the session");
        started.notified().await;
        server1.abort();
        let _ = server1.await;
        // Store the release permit now: the abandoned turn's `notified()` future
        // is already dropped with the connection, so the permit goes to the
        // takeover turn below (Notify permits persist until a waiter arrives).
        release.notify_one();

        // Connection 2: the abandoned prompt must not block the load, and the
        // rebound session must run a full turn.
        let (_server2, client2) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(session_id.clone(), cwd))
                        .block_task()
                        .await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new("take over"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 takeover completed before timeout")
            .expect("official ACP client completed")
            .expect("load and takeover prompt must succeed after the connection died");
    }

    // ── per-session agent factory (`_meta` overrides) ─────────────────────

    /// A factory whose every `build` call records the `_meta` map and cwd it
    /// received and returns an agent that records its factory-call index
    /// when run.
    struct RecordingFactory {
        metas: Arc<Mutex<Vec<String>>>,
        cwds: Arc<Mutex<Vec<String>>>,
        served_by: Arc<Mutex<Vec<String>>>,
        fail: bool,
    }

    impl RecordingFactory {
        fn new() -> Self {
            Self {
                metas: Arc::new(Mutex::new(Vec::new())),
                cwds: Arc::new(Mutex::new(Vec::new())),
                served_by: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            }
        }
    }

    impl super::super::config::AgentFactory for RecordingFactory {
        fn build(
            &self,
            meta: &serde_json::Map<String, serde_json::Value>,
            cwd: &std::path::Path,
        ) -> Result<Arc<dyn Agent>, String> {
            if self.fail {
                return Err("bad override value".to_string());
            }
            let index = self.metas.lock().unwrap().len();
            self.metas
                .lock()
                .unwrap()
                .push(serde_json::to_string(meta).expect("meta serializes"));
            self.cwds
                .lock()
                .unwrap()
                .push(cwd.display().to_string());
            Ok(Arc::new(FactoryAgent {
                marker: format!("call-{index}"),
                served_by: self.served_by.clone(),
            }))
        }
    }

    struct FactoryAgent {
        marker: String,
        served_by: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl Agent for FactoryAgent {
        fn name(&self) -> &str {
            "factory-agent"
        }

        fn description(&self) -> &str {
            "Records which factory call produced it when run"
        }

        fn sub_agents(&self) -> &[Arc<dyn Agent>] {
            &[]
        }

        async fn run(&self, _ctx: Arc<dyn InvocationContext>) -> AdkResult<EventStream> {
            self.served_by.lock().unwrap().push(self.marker.clone());
            let mut event = Event::new("factory-agent");
            event.set_content(adk_core::Content::new("model").with_text("factory response"));
            Ok(Box::pin(futures::stream::once(async move { Ok(event) })))
        }
    }

    fn override_meta(model: &str) -> Meta {
        let mut map = serde_json::Map::new();
        map.insert(
            "agentx/model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
        map
    }

    /// A config-level base agent that records when it runs, so a test can
    /// tell base-served turns from factory-served ones.
    struct RecordingBaseAgent {
        served: Arc<Mutex<bool>>,
    }

    #[async_trait::async_trait]
    impl Agent for RecordingBaseAgent {
        fn name(&self) -> &str {
            "recording-base-agent"
        }

        fn description(&self) -> &str {
            "Records that it ran"
        }

        fn sub_agents(&self) -> &[Arc<dyn Agent>] {
            &[]
        }

        async fn run(&self, _ctx: Arc<dyn InvocationContext>) -> AdkResult<EventStream> {
            *self.served.lock().unwrap() = true;
            let mut event = Event::new("recording-base-agent");
            event.set_content(adk_core::Content::new("model").with_text("base response"));
            Ok(Box::pin(futures::stream::once(async move { Ok(event) })))
        }
    }

    /// `_meta` on `session/new` reaches the factory and the composed agent
    /// serves the session's prompts.
    #[tokio::test]
    async fn factory_meta_composes_the_serving_agent() {
        let (base, session_service) = mock_agent_and_session();
        let factory = RecordingFactory::new();
        let metas = factory.metas.clone();
        let served_by = factory.served_by.clone();
        let config = AcpServerConfigBuilder::new()
            .agent(base)
            .agent_factory(Arc::new(factory))
            .session_service(session_service)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        let (server, client) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    request.meta = Some(override_meta("test-model"));
                    let session = connection.send_request(request).block_task().await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session.session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("hello"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        let _ = server;
        tokio::time::timeout(std::time::Duration::from_secs(5), client)
            .await
            .expect("connection completed before timeout")
            .expect("official ACP client completed")
            .expect("factory-composed session prompt must succeed");

        let metas = metas.lock().unwrap();
        assert_eq!(metas.len(), 1, "factory called exactly once");
        assert!(
            metas[0].contains("agentx/model") && metas[0].contains("test-model"),
            "factory received the _meta map verbatim: {}",
            metas[0]
        );
        assert_eq!(*served_by.lock().unwrap(), vec!["call-0".to_string()]);
    }

    /// `session/new` **without** `_meta` still routes through a configured
    /// factory: the factory receives an empty map plus the session's cwd, and
    /// the composed agent serves the prompt. Per-cwd concerns (file-domain
    /// binding) must cover every session, not only override-carrying ones.
    #[tokio::test]
    async fn factory_without_meta_receives_empty_map_and_cwd() {
        let (base, session_service) = mock_agent_and_session();
        let factory = RecordingFactory::new();
        let metas = factory.metas.clone();
        let cwds = factory.cwds.clone();
        let served_by = factory.served_by.clone();
        let config = AcpServerConfigBuilder::new()
            .agent(base)
            .agent_factory(Arc::new(factory))
            .session_service(session_service)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        let (server, client) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    // No `_meta` on purpose: the plain-client shape.
                    let request =
                        NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    let session = connection.send_request(request).block_task().await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session.session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("hello"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        let _ = server;
        tokio::time::timeout(std::time::Duration::from_secs(5), client)
            .await
            .expect("connection completed before timeout")
            .expect("official ACP client completed")
            .expect("factory-composed session prompt must succeed");

        let metas = metas.lock().unwrap();
        assert_eq!(metas.len(), 1, "factory called even without _meta");
        assert_eq!(metas[0], "{}", "factory received an empty map: {}", metas[0]);
        let cwds = cwds.lock().unwrap();
        assert_eq!(
            cwds.len(),
            1,
            "one build call records one cwd (metas/cwds stay paired)"
        );
        assert_eq!(
            cwds[0],
            std::env::current_dir().expect("absolute cwd").display().to_string(),
            "factory received the session/new cwd verbatim: {}",
            cwds[0]
        );
        assert_eq!(*served_by.lock().unwrap(), vec!["call-0".to_string()]);
    }

    /// `session/load` **without** `_meta` keeps the session's factory-composed
    /// agent: the rebind preserves it and no second factory call happens.
    #[tokio::test]
    async fn load_without_meta_preserves_the_session_agent() {
        let (base, session_service) = mock_agent_and_session();
        let factory = RecordingFactory::new();
        let metas = factory.metas.clone();
        let served_by = factory.served_by.clone();
        let config = AcpServerConfigBuilder::new()
            .agent(base)
            .agent_factory(Arc::new(factory))
            .session_service(session_service)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        // Connection 1: create with overrides, run one turn.
        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (server1, client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    request.meta = Some(override_meta("model-a"));
                    let session = connection.send_request(request).block_task().await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session.session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("first"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    session_tx
                        .send((session.session_id, std::env::current_dir().expect("absolute cwd")))
                        .expect("test awaits the session");
                    Ok(())
                })
            }),
        );
        let _ = server1;
        tokio::time::timeout(std::time::Duration::from_secs(5), client1)
            .await
            .expect("connection 1 completed before timeout")
            .expect("official ACP client completed")
            .expect("first turn must succeed");
        let (session_id, cwd) = session_rx.await.expect("session created");

        // Connection 2: load with no `_meta`, run a turn — same factory agent.
        let (_server2, client2) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(session_id.clone(), cwd))
                        .block_task()
                        .await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new("second"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 completed before timeout")
            .expect("official ACP client completed")
            .expect("load without _meta must succeed");

        assert_eq!(
            metas.lock().unwrap().len(),
            1,
            "no rebuild: load without _meta keeps the existing agent"
        );
        assert_eq!(
            *served_by.lock().unwrap(),
            vec!["call-0".to_string(), "call-0".to_string()],
            "both turns served by the same factory-composed agent"
        );
    }

    /// `session/load` **with** `_meta` rebuilds the session's agent through
    /// the factory with the fresh overrides.
    #[tokio::test]
    async fn load_with_meta_rebuilds_the_session_agent() {
        let (base, session_service) = mock_agent_and_session();
        let factory = RecordingFactory::new();
        let metas = factory.metas.clone();
        let served_by = factory.served_by.clone();
        let config = AcpServerConfigBuilder::new()
            .agent(base)
            .agent_factory(Arc::new(factory))
            .session_service(session_service)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        let (session_tx, session_rx) = tokio::sync::oneshot::channel();
        let (server1, client1) = spawn_acp_connection(
            handler.clone(),
            capabilities.clone(),
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    request.meta = Some(override_meta("model-a"));
                    let session = connection.send_request(request).block_task().await?;
                    session_tx
                        .send((session.session_id, std::env::current_dir().expect("absolute cwd")))
                        .expect("test awaits the session");
                    Ok(())
                })
            }),
        );
        let _ = server1;
        tokio::time::timeout(std::time::Duration::from_secs(5), client1)
            .await
            .expect("connection 1 completed before timeout")
            .expect("official ACP client completed")
            .expect("session creation must succeed");
        let (session_id, cwd) = session_rx.await.expect("session created");

        let (_server2, client2) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = LoadSessionRequest::new(session_id.clone(), cwd);
                    request.meta = Some(override_meta("model-b"));
                    connection.send_request(request).block_task().await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new("second"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), client2)
            .await
            .expect("connection 2 completed before timeout")
            .expect("official ACP client completed")
            .expect("load with _meta must succeed");

        let metas = metas.lock().unwrap();
        assert_eq!(metas.len(), 2, "the load rebuilt through the factory");
        assert!(
            metas[1].contains("model-b"),
            "rebuild saw the fresh overrides: {}",
            metas[1]
        );
        assert_eq!(
            *served_by.lock().unwrap(),
            vec!["call-1".to_string()],
            "the post-load turn was served by the rebuilt agent"
        );
    }

    /// A factory error fails `session/new` loudly — no silent base fallback.
    #[tokio::test]
    async fn factory_error_fails_session_new() {
        let (base, session_service) = mock_agent_and_session();
        let mut factory = RecordingFactory::new();
        factory.fail = true;
        let config = AcpServerConfigBuilder::new()
            .agent(base)
            .agent_factory(Arc::new(factory))
            .session_service(session_service)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        let (server, client) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    request.meta = Some(override_meta("bad"));
                    let outcome = connection.send_request(request).block_task().await;
                    assert!(
                        outcome.is_err(),
                        "session/new must fail when the factory rejects the overrides"
                    );
                    Ok(())
                })
            }),
        );
        let _ = server;
        tokio::time::timeout(std::time::Duration::from_secs(5), client)
            .await
            .expect("connection completed before timeout")
            .expect("official ACP client completed")
            .expect("the rejected session/new must surface as a client error");
    }

    /// Without a factory, `_meta` rides the request harmlessly and the
    /// config-level agent serves every session (legacy behaviour unchanged).
    #[tokio::test]
    async fn no_factory_meta_is_ignored_and_base_agent_serves() {
        let base_served = Arc::new(Mutex::new(false));
        let agent: Arc<dyn Agent> = Arc::new(RecordingBaseAgent { served: base_served.clone() });
        let session_svc: Arc<dyn SessionService> =
            Arc::new(adk_session::InMemorySessionService::new());
        let config = AcpServerConfigBuilder::new()
            .agent(agent)
            .session_service(session_svc)
            .build()
            .expect("valid config");
        let capabilities = CapabilitiesBuilder::build(&config);
        let handler =
            Arc::new(AcpSessionHandler::new(&config, CancellationToken::new()).expect("handler"));

        let (server, client) = spawn_acp_connection(
            handler,
            capabilities,
            Box::new(move |connection| {
                Box::pin(async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let mut request = NewSessionRequest::new(std::env::current_dir().expect("absolute cwd"));
                    request.meta = Some(override_meta("ignored-model"));
                    let session = connection.send_request(request).block_task().await?;
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session.session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("hello"))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(prompt.stop_reason, StopReason::EndTurn);
                    Ok(())
                })
            }),
        );
        let _ = server;
        tokio::time::timeout(std::time::Duration::from_secs(5), client)
            .await
            .expect("connection completed before timeout")
            .expect("official ACP client completed")
            .expect("meta without a factory must not disturb the session");
        assert!(
            *base_served.lock().unwrap(),
            "the config-level agent served the prompt"
        );
    }
}
