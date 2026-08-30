mod ingest;
mod scan;
mod store;

pub(crate) use ingest::{IngestEvent, embed_images, ingest_into_index};
pub(crate) use scan::{ImageFile, SearchRoot, discover_images};
pub(crate) use store::{ImageIndex, ImageRecord, StagedImageIndex};
