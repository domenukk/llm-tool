//! Compatibility shims for `no_std` / `std` mode.
//!
//! Re-exports [`HashMap`] and lock helpers from either `std` or
//! `hashbrown`/`spin` depending on the active feature set.

// -- HashMap ------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) use std::collections::HashMap;

#[cfg(not(feature = "std"))]
pub(crate) use hashbrown::HashMap;

// -- RwLock -------------------------------------------------------------------
//
// `std::sync::RwLock::read/write` return `Result` (for poisoning).
// `spin::RwLock::read/write` return the guard directly (no poisoning).
// We normalise the API with thin wrappers.

#[cfg(feature = "std")]
mod lock {
    use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

    /// Acquire a read lock, returning `Ok(guard)` or `Err(message)`.
    pub(crate) fn read_lock<T>(
        lock: &RwLock<T>,
    ) -> Result<RwLockReadGuard<'_, T>, alloc::string::String> {
        lock.read()
            .map_err(|e| alloc::format!("RwLock poisoned: {e}"))
    }

    /// Acquire a write lock, returning `Ok(guard)` or `Err(message)`.
    pub(crate) fn write_lock<T>(
        lock: &RwLock<T>,
    ) -> Result<RwLockWriteGuard<'_, T>, alloc::string::String> {
        lock.write()
            .map_err(|e| alloc::format!("RwLock poisoned: {e}"))
    }
}

#[cfg(not(feature = "std"))]
mod lock {
    use spin::{RwLock, RwLockReadGuard, RwLockWriteGuard};

    /// Acquire a read lock — infallible under `spin`, but returns `Result`
    /// to maintain a unified API with the `std` path.
    pub(crate) fn read_lock<T>(
        lock: &RwLock<T>,
    ) -> Result<RwLockReadGuard<'_, T>, alloc::string::String> {
        // Unified signature with std::sync::RwLock — spin locks never fail.
        Result::<_, core::convert::Infallible>::Ok(lock.read()).map_err(|e| match e {})
    }

    /// Acquire a write lock — infallible under `spin`, but returns `Result`
    /// to maintain a unified API with the `std` path.
    pub(crate) fn write_lock<T>(
        lock: &RwLock<T>,
    ) -> Result<RwLockWriteGuard<'_, T>, alloc::string::String> {
        // Unified signature with std::sync::RwLock — spin locks never fail.
        Result::<_, core::convert::Infallible>::Ok(lock.write()).map_err(|e| match e {})
    }
}

/// Read-write lock — [`std::sync::RwLock`] under `std`,
/// [`spin::RwLock`] under `no_std`.
#[cfg(feature = "std")]
pub(crate) use std::sync::RwLock;

pub(crate) use lock::{read_lock, write_lock};
#[cfg(not(feature = "std"))]
pub(crate) use spin::RwLock;
