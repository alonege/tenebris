use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("matmul: inner dimensions do not match: lhs.k = {lhs_k}, rhs.m = {rhs_m}")]
    MatmulDimMismatch { lhs_k: usize, rhs_m: usize },

    #[error("broadcast: dimension {dim}: {lhs} vs {rhs} (only 1 or equal allowed)")]
    BroadcastMismatch { dim: usize, lhs: usize, rhs: usize },

    #[error("{operation}: layout needs {required_len} elements, storage has {storage_len}")]
    LayoutOutOfBounds {
        operation: &'static str,
        required_len: usize,
        storage_len: usize,
    },

    #[error("storage mutex poisoned (a thread panicked while holding the lock)")]
    StorageLockPoisoned,

    #[error("{operation}: not yet implemented for this backend")]
    Unsupported { operation: &'static str },
}
