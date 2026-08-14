//! Transport layer for ACP protocol messages.
//!
//! Defines the [`Transport`] trait and provides [`StdioTransport`] for the
//! official ACP JSON-RPC stream over stdin/stdout. The HTTP transport
//! (`HttpTransport`) is available behind the `http` feature.
//!
//! Both transports share [`build_agent_component`], which wires the session
//! handler into the official ACP SDK's typed callbacks.

pub mod component;
pub mod stdio;

#[cfg(feature = "http")]
pub mod http;

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::error::AcpServerError;
use super::handler::AcpSessionHandler;

pub use component::build_agent_component;

/// A transport layer for ACP protocol messages.
///
/// Implementations handle the wire format and connection management,
/// routing incoming messages to the [`AcpSessionHandler`] and sending
/// responses back to the client.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Start listening for incoming connections/messages.
    ///
    /// Returns when the transport is shut down (via the cancellation token)
    /// or encounters a fatal error.
    async fn serve(
        &self,
        handler: Arc<AcpSessionHandler>,
        shutdown: CancellationToken,
    ) -> Result<(), AcpServerError>;
}

pub use stdio::StdioTransport;

#[cfg(feature = "http")]
pub use http::HttpTransport;
