mod args;
mod index;
mod socket;
mod stdio;
mod terminal;

pub(crate) use args::{Cli, ServeMode, ServerAction, Subcommands};
pub(crate) use index::index;
pub(crate) use socket::{
    serve as serve_socket, stop as stop_socket, try_search as try_socket_search,
};
pub(crate) use stdio::serve;
pub(crate) use terminal::Terminal;
