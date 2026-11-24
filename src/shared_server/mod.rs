//! Shares model sessions across VisionGrep invocations over a local Unix socket.

mod address;
mod client;
mod protocol;
mod serve;

pub(crate) use client::{index, search, stop};
pub(crate) use protocol::Event;
pub(crate) use serve::run;
