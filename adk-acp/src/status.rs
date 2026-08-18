//! Agent status state machine for ACP sessions.
//!
//! Tracks the lifecycle of an ACP agent connection, enabling UIs and
//! orchestrators to display real-time status.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// The current status of an ACP agent connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AgentStatus {
    /// Agent process is being spawned and initialized.
    Starting = 0,
    /// Agent is idle, waiting for a prompt.
    Idle = 1,
    /// Agent is processing a prompt (generating code, running tools).
    Running = 2,
    /// Agent is waiting for a permission decision (HITL pause).
    WaitingPermission = 3,
    /// Agent encountered an error.
    Error = 4,
    /// Agent is shutting down.
    Stopping = 5,
    /// Agent process has exited.
    Stopped = 6,
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Idle => write!(f, "idle"),
            Self::Running => write!(f, "running"),
            Self::WaitingPermission => write!(f, "waiting_permission"),
            Self::Error => write!(f, "error"),
            Self::Stopping => write!(f, "stopping"),
            Self::Stopped => write!(f, "stopped"),
        }
    }
}

impl From<u8> for AgentStatus {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Starting,
            1 => Self::Idle,
            2 => Self::Running,
            3 => Self::WaitingPermission,
            4 => Self::Error,
            5 => Self::Stopping,
            6 => Self::Stopped,
            _ => Self::Error,
        }
    }
}

/// Thread-safe, lock-free status tracker.
///
/// Share across the connection task and the main task to observe
/// real-time agent status without blocking.
///
/// # Example
///
/// ```rust
/// use agentx_adk_acp::status::{AgentStatus, StatusTracker};
///
/// let tracker = StatusTracker::new();
/// assert_eq!(tracker.get(), AgentStatus::Starting);
///
/// tracker.set(AgentStatus::Idle);
/// assert_eq!(tracker.get(), AgentStatus::Idle);
/// assert!(tracker.is_idle());
/// ```
#[derive(Debug, Clone)]
pub struct StatusTracker {
    inner: Arc<AtomicU8>,
}

impl StatusTracker {
    /// Create a new tracker in `Starting` state.
    pub fn new() -> Self {
        Self { inner: Arc::new(AtomicU8::new(AgentStatus::Starting as u8)) }
    }

    /// Get the current status.
    pub fn get(&self) -> AgentStatus {
        AgentStatus::from(self.inner.load(Ordering::Relaxed))
    }

    /// Set the status.
    pub fn set(&self, status: AgentStatus) {
        self.inner.store(status as u8, Ordering::Relaxed);
    }

    /// Whether the agent is idle (ready for a prompt).
    pub fn is_idle(&self) -> bool {
        self.get() == AgentStatus::Idle
    }

    /// Whether the agent is currently processing.
    pub fn is_running(&self) -> bool {
        self.get() == AgentStatus::Running
    }

    /// Whether the agent is waiting for permission approval.
    pub fn is_waiting_permission(&self) -> bool {
        self.get() == AgentStatus::WaitingPermission
    }

    /// Whether the agent has stopped (exited or errored).
    pub fn is_done(&self) -> bool {
        matches!(self.get(), AgentStatus::Stopped | AgentStatus::Error)
    }
}

impl Default for StatusTracker {
    fn default() -> Self {
        Self::new()
    }
}
