//! ACP session lifecycle and ADK-Rust Runner bridge.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
    ContentBlock, ListSessionsRequest, ListSessionsResponse, McpServer, NewSessionRequest,
    PromptRequest, SessionId, SessionInfo, SessionNotification, StopReason,
};
use agent_client_protocol::{Client, ConnectionTo};
use futures::StreamExt;
use rmcp::{ServiceExt, transport::TokioChildProcess};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::config::AcpServerConfig;
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
    execution_token: Option<CancellationToken>,
    mcp: McpSessionResources,
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
    session_service: Arc<dyn SessionService>,
    app_name: String,
    user_id: String,
    max_sessions: usize,
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
                SessionState { execution_token: None, mcp: McpSessionResources::default() },
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
        {
            let mut sessions = self.sessions.lock().await;
            if sessions.contains_key(&id) {
                return Err(AcpServerError::Execution(
                    "the session is already active on this ACP connection".into(),
                ));
            }
            if sessions.len() >= self.max_sessions {
                return Err(AcpServerError::MaxSessionsReached(self.max_sessions));
            }
            sessions.insert(
                id.clone(),
                SessionState { execution_token: None, mcp: McpSessionResources::default() },
            );
        }
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
    /// session is registered on the connection (subject to the max-session and
    /// already-active checks) with its MCP servers started. After reactivation,
    /// every stored event is mapped to its `SessionUpdate` variant through the
    /// shared [`ResponseStreamer`] and sent in chronological order.
    #[allow(clippy::too_many_arguments)]
    pub async fn load_session(
        &self,
        session_id: &SessionId,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: Vec<McpServer>,
        request_cancellation: RequestCancellation,
        connection: ConnectionTo<Client>,
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
        {
            let mut sessions = self.sessions.lock().await;
            if sessions.contains_key(&id) {
                return Err(AcpServerError::Execution(
                    "the session is already active on this ACP connection".into(),
                ));
            }
            if sessions.len() >= self.max_sessions {
                return Err(AcpServerError::MaxSessionsReached(self.max_sessions));
            }
            sessions.insert(
                id.clone(),
                SessionState { execution_token: None, mcp: McpSessionResources::default() },
            );
        }
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
            sessions.insert(
                new_session_id.clone(),
                SessionState { execution_token: None, mcp: McpSessionResources::default() },
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
            && let Some(token) = state.execution_token
        {
            token.cancel();
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
        let (cancellation_token, runtime_toolsets) = {
            let mut sessions = self.sessions.lock().await;
            let state =
                sessions.get_mut(&id).ok_or_else(|| AcpServerError::SessionNotFound(id.clone()))?;
            if state.execution_token.is_some() {
                return Err(AcpServerError::Execution(
                    "a prompt is already running in this session".into(),
                ));
            }
            let token = CancellationToken::new();
            state.execution_token = Some(token.clone());
            (token, state.mcp.toolsets.clone())
        };

        let result = self
            .execute_prompt(&request, connection, cancellation_token.clone(), runtime_toolsets)
            .await;

        let mut sessions = self.sessions.lock().await;
        if let Some(state) = sessions.get_mut(&id) {
            state.execution_token = None;
        }
        drop(sessions);

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
            .agent(self.agent.clone())
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
            .and_then(|state| state.execution_token.clone())
        {
            token.cancel();
        }
    }

    /// Close an active session without deleting its persisted history.
    pub async fn close_session(&self, session_id: &SessionId) -> Result<(), AcpServerError> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions
            .remove(&session_id.to_string())
            .ok_or_else(|| AcpServerError::SessionNotFound(session_id.to_string()))?;
        if let Some(token) = state.execution_token {
            token.cancel();
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
            if let Some(token) = &state.execution_token {
                token.cancel();
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
    use agent_client_protocol::schema::v1::{
        AudioContent, EmbeddedResource as AcpEmbeddedResource, EmbeddedResourceResource,
        EnvVariable, ImageContent, McpServerStdio, TextContent,
        TextResourceContents as AcpTextResourceContents,
    };
    use base64::{Engine as _, engine::general_purpose};

    use super::super::capabilities::CapabilitiesBuilder;
    use super::super::config::AcpServerConfigBuilder;
    use super::super::test_helpers::mock_agent_and_session;
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
}
