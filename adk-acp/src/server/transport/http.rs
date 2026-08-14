//! ACP HTTP/WebSocket transport via the official `agent-client-protocol-http` crate.
//!
//! Exposes the same agent component as the stdio transport over an axum router
//! supporting POST (JSON-RPC batch), GET (WebSocket upgrade), and DELETE
//! (connection teardown). A fresh agent component is constructed per
//! connection through the registry's factory closure.

use std::net::SocketAddr;
use std::sync::Arc;

use agent_client_protocol_http::{AcpHttpServer, ServerOptions};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::super::capabilities::{AgentCapabilities, CapabilitiesBuilder};
use super::super::config::AcpServerConfig;
use super::super::error::AcpServerError;
use super::super::handler::AcpSessionHandler;
use super::Transport;
use super::component::build_agent_component;

/// ACP HTTP/WebSocket transport. Binds a TCP listener and serves the official
/// HTTP router, constructing a fresh agent component per inbound connection.
pub struct HttpTransport {
    addr: SocketAddr,
    capabilities: AgentCapabilities,
    agent_name: String,
    agent_title: String,
}

impl HttpTransport {
    /// Create an HTTP transport bound to `addr`, reading capabilities and
    /// identity from the server config.
    pub fn new(addr: SocketAddr, config: &AcpServerConfig) -> Self {
        Self {
            addr,
            capabilities: CapabilitiesBuilder::build(config),
            agent_name: config.agent_name.clone(),
            agent_title: config.agent_description.clone(),
        }
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn serve(
        &self,
        handler: Arc<AcpSessionHandler>,
        shutdown: CancellationToken,
    ) -> Result<(), AcpServerError> {
        info!(agent = %self.agent_name, addr = %self.addr, "ACP HTTP transport started");

        let capabilities = self.capabilities.clone();
        let name = self.agent_name.clone();
        let title = self.agent_title.clone();

        // The factory is called once per inbound connection, so it must be
        // `Fn()` — only `Arc`/`Clone` handles are captured, each cloned per call.
        let factory = move || {
            build_agent_component(
                handler.clone(),
                capabilities.clone(),
                name.clone(),
                title.clone(),
            )
        };

        let router = AcpHttpServer::new(factory)
            .with_options(ServerOptions {
                path: "/".to_string(),
                ..Default::default()
            })
            .into_router();

        let listener = tokio::net::TcpListener::bind(self.addr)
            .await
            .map_err(|error| AcpServerError::Transport(error.to_string()))?;

        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                shutdown.cancelled().await;
            })
            .await
            .map_err(|error| AcpServerError::Transport(error.to_string()))
    }
}
