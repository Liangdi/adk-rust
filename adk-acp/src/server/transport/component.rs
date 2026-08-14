//! Agent-component builder shared by every transport.
//!
//! [`build_agent_component`] wires the [`AcpSessionHandler`] into the official
//! ACP SDK's typed request/notification callbacks, producing a component that
//! implements [`ConnectTo<Client>`]. Both the stdio and HTTP transports call
//! this function so that protocol wiring is defined in exactly one place.

use std::sync::Arc;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, CloseSessionRequest, CloseSessionResponse, DeleteSessionRequest,
    DeleteSessionResponse, ForkSessionRequest, ForkSessionResponse, Implementation,
    InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
    LoadSessionRequest, LoadSessionResponse, NewSessionRequest, NewSessionResponse, PromptRequest,
    PromptResponse, ResumeSessionRequest, ResumeSessionResponse, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Error, Responder};
use tracing::warn;

use super::super::capabilities::AgentCapabilities;
use super::super::error::AcpServerError;
use super::super::handler::AcpSessionHandler;

/// Build the ACP agent component: the typed request/notification callbacks
/// wired around the session handler, ready to connect to any client component.
///
/// Extracting this from the stdio transport lets the HTTP transport reuse the
/// identical protocol wiring — the official HTTP server accepts a factory that
/// returns a fresh `ConnectTo<Client>` per connection.
///
/// The returned builder is a single fixed monomorphized type, so `impl
/// ConnectTo<Client>` compiles without dynamic dispatch.
#[allow(clippy::too_many_lines)]
pub fn build_agent_component(
    handler: Arc<AcpSessionHandler>,
    initialize_capabilities: AgentCapabilities,
    initialize_name: String,
    initialize_title: String,
) -> impl ConnectTo<Client> {
    let new_handler = handler.clone();
    let prompt_handler = handler.clone();
    let cancel_handler = handler.clone();
    let close_handler = handler.clone();
    let resume_handler = handler.clone();
    let load_handler = handler.clone();
    let fork_handler = handler.clone();
    let set_mode_handler = handler.clone();
    let set_config_handler = handler.clone();
    let list_handler = handler.clone();
    let delete_handler = handler;

    Agent
        .builder()
        .name(initialize_name.clone())
        .on_receive_request(
            move |request: InitializeRequest,
                  responder: Responder<InitializeResponse>,
                  _connection: ConnectionTo<Client>| {
                let capabilities = initialize_capabilities.clone();
                let name = initialize_name.clone();
                let title = initialize_title.clone();
                async move {
                    let version = match request.protocol_version {
                        ProtocolVersion::V1 => ProtocolVersion::V1,
                        _ => ProtocolVersion::V1,
                    };
                    let mut implementation = Implementation::new(name, env!("CARGO_PKG_VERSION"));
                    if !title.is_empty() {
                        implementation = implementation.title(title);
                    }
                    responder.respond(
                        InitializeResponse::new(version)
                            .agent_capabilities(capabilities)
                            .agent_info(implementation),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: NewSessionRequest,
                  responder: Responder<NewSessionResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = new_handler.clone();
                async move {
                    let cancellation = responder.cancellation();
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        let response = match handler.create_session(request, cancellation).await {
                            Ok(session_id) => {
                                emit_activation_updates(&handler, &session_id, &spawned_connection)
                                    .await;
                                let (modes, config_options) =
                                    handler.session_controls_snapshot(&session_id).await;
                                Ok(NewSessionResponse::new(session_id)
                                    .modes(modes)
                                    .config_options(config_options))
                            }
                            Err(error) => Err(to_protocol_error(error)),
                        };
                        responder.respond_with_result(response)
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: PromptRequest,
                  responder: Responder<PromptResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = prompt_handler.clone();
                async move {
                    let cancellation = responder.cancellation();
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        let session_id = request.session_id.clone();
                        let cancellation_handler = handler.clone();
                        let work = handler.handle_prompt(request, spawned_connection);
                        tokio::pin!(work);
                        let result = tokio::select! {
                            result = &mut work => result,
                            _ = cancellation.cancelled() => {
                                cancellation_handler.cancel_session(&session_id).await;
                                work.await
                            }
                        };
                        responder.respond_with_result(
                            result.map(PromptResponse::new).map_err(to_protocol_error),
                        )
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification, _connection: ConnectionTo<Client>| {
                cancel_handler.cancel_session(&notification.session_id).await;
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: CloseSessionRequest,
                        responder: Responder<CloseSessionResponse>,
                        _connection: ConnectionTo<Client>| {
                match close_handler.close_session(&request.session_id).await {
                    Ok(()) => responder.respond(CloseSessionResponse::new()),
                    Err(error) => responder.respond_with_error(to_protocol_error(error)),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: ResumeSessionRequest,
                  responder: Responder<ResumeSessionResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = resume_handler.clone();
                async move {
                    let cancellation = responder.cancellation();
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        let session_id = request.session_id.clone();
                        let result = handler
                            .resume_session(
                                &request.session_id,
                                request.cwd,
                                request.additional_directories,
                                request.mcp_servers,
                                cancellation,
                            )
                            .await;
                        let response = match result {
                            Ok(()) => {
                                emit_activation_updates(&handler, &session_id, &spawned_connection)
                                    .await;
                                let (modes, config_options) =
                                    handler.session_controls_snapshot(&session_id).await;
                                Ok(ResumeSessionResponse::new()
                                    .modes(modes)
                                    .config_options(config_options))
                            }
                            Err(error) => Err(to_protocol_error(error)),
                        };
                        responder.respond_with_result(response)
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: LoadSessionRequest,
                  responder: Responder<LoadSessionResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = load_handler.clone();
                async move {
                    let cancellation = responder.cancellation();
                    let replay_connection = connection.clone();
                    let activation_connection = connection.clone();
                    connection.spawn(async move {
                        let session_id = request.session_id.clone();
                        let result = handler
                            .load_session(
                                &request.session_id,
                                request.cwd,
                                request.additional_directories,
                                request.mcp_servers,
                                cancellation,
                                replay_connection,
                            )
                            .await;
                        let response = match result {
                            Ok(()) => {
                                emit_activation_updates(
                                    &handler,
                                    &session_id,
                                    &activation_connection,
                                )
                                .await;
                                let (modes, config_options) =
                                    handler.session_controls_snapshot(&session_id).await;
                                Ok(LoadSessionResponse::new()
                                    .modes(modes)
                                    .config_options(config_options))
                            }
                            Err(error) => Err(to_protocol_error(error)),
                        };
                        responder.respond_with_result(response)
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: ForkSessionRequest,
                  responder: Responder<ForkSessionResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = fork_handler.clone();
                async move {
                    let cancellation = responder.cancellation();
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        let result = handler
                            .fork_session(
                                &request.session_id,
                                request.cwd,
                                request.additional_directories,
                                request.mcp_servers,
                                cancellation,
                            )
                            .await;
                        let response = match result {
                            Ok(new_session_id) => {
                                emit_activation_updates(
                                    &handler,
                                    &new_session_id,
                                    &spawned_connection,
                                )
                                .await;
                                let (modes, config_options) =
                                    handler.session_controls_snapshot(&new_session_id).await;
                                Ok(ForkSessionResponse::new(new_session_id)
                                    .modes(modes)
                                    .config_options(config_options))
                            }
                            Err(error) => Err(to_protocol_error(error)),
                        };
                        responder.respond_with_result(response)
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: SetSessionModeRequest,
                  responder: Responder<SetSessionModeResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = set_mode_handler.clone();
                async move {
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        responder.respond_with_result(
                            handler
                                .set_mode(&request.session_id, &request.mode_id, spawned_connection)
                                .await
                                .map(|()| SetSessionModeResponse::new())
                                .map_err(to_protocol_error),
                        )
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            move |request: SetSessionConfigOptionRequest,
                  responder: Responder<SetSessionConfigOptionResponse>,
                  connection: ConnectionTo<Client>| {
                let handler = set_config_handler.clone();
                async move {
                    let spawned_connection = connection.clone();
                    connection.spawn(async move {
                        let session_id = request.session_id.clone();
                        let result = handler
                            .set_config_option(
                                &request.session_id,
                                &request.config_id,
                                request.value,
                                spawned_connection,
                            )
                            .await;
                        let response = match result {
                            Ok(()) => {
                                let (_, config_options) =
                                    handler.session_controls_snapshot(&session_id).await;
                                Ok(SetSessionConfigOptionResponse::new(
                                    config_options.unwrap_or_default(),
                                ))
                            }
                            Err(error) => Err(to_protocol_error(error)),
                        };
                        responder.respond_with_result(response)
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: ListSessionsRequest,
                        responder: Responder<ListSessionsResponse>,
                        _connection: ConnectionTo<Client>| {
                match list_handler.list_sessions(request).await {
                    Ok(response) => responder.respond(response),
                    Err(error) => responder.respond_with_error(to_protocol_error(error)),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: DeleteSessionRequest,
                        responder: Responder<DeleteSessionResponse>,
                        _connection: ConnectionTo<Client>| {
                match delete_handler.delete_session(&request.session_id).await {
                    Ok(()) => responder.respond(DeleteSessionResponse::new()),
                    Err(error) => responder.respond_with_error(to_protocol_error(error)),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
}

/// Best-effort emission of the session-activation `session/update`
/// notifications (available commands and session info) for a freshly-active
/// session.
///
/// A transport hiccup while sending these notifications must not fail the
/// surrounding session request, so any error is logged and swallowed rather
/// than propagated.
async fn emit_activation_updates(
    handler: &Arc<AcpSessionHandler>,
    session_id: &agent_client_protocol::schema::v1::SessionId,
    connection: &ConnectionTo<Client>,
) {
    if let Err(error) = handler.emit_session_activation_updates(session_id, connection).await {
        warn!(%error, session_id = %session_id, "failed to emit session activation updates");
    }
}

fn to_protocol_error(error: AcpServerError) -> Error {
    match error {
        AcpServerError::MalformedMessage(message)
        | AcpServerError::SessionNotFound(message)
        | AcpServerError::UnsupportedVersion { requested: message, .. } => {
            Error::invalid_params().data(message)
        }
        AcpServerError::MaxSessionsReached(max) => {
            Error::invalid_params().data(format!("maximum active sessions reached: {max}"))
        }
        other => Error::internal_error().data(other.to_string()),
    }
}
