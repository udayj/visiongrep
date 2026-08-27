mod ingest;
mod scan;
mod store;

pub(crate) use ingest::{IngestEvent, embed_images, embed_into_index};
pub(crate) use scan::{ImageFile, discover_images};
pub(crate) use store::{ImageIndex, ImageRecord};
