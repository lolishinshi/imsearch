mod index;
mod invlists;
mod types;

pub use index::*;
pub use invlists::*;
pub use types::*;

pub fn get_faiss_stats() -> faiss_sys::FaissIndexIVFStats {
    unsafe {
        let stats = faiss_sys::faiss_get_indexIVF_stats();
        *stats
    }
}

pub fn reset_faiss_stats() {
    unsafe {
        faiss_sys::faiss_IndexIVFStats_reset(faiss_sys::faiss_get_indexIVF_stats());
    }
}
