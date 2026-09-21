//! Compatibility shims for `no_std` / `std` mode.
//!
//! Re-exports [`HashMap`] and lock helpers from either `std` or
//! `hashbrown`/`spin` depending on the active feature set.

// -- HashMap ------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) use std::collections::HashMap;
/// Borrowing iterator over a [`HashMap`]'s entries — `std` variant.
#[cfg(feature = "std")]
pub(crate) use std::collections::hash_map::Iter as HashMapIter;

#[cfg(not(feature = "std"))]
pub(crate) use hashbrown::HashMap;
/// Borrowing iterator over a [`HashMap`]'s entries — `hashbrown` variant.
#[cfg(not(feature = "std"))]
pub(crate) use hashbrown::hash_map::Iter as HashMapIter;

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

    /// Acquire a read lock, recovering the guard even if the lock was poisoned.
    pub(crate) fn read_lock_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
        match lock.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("RwLock poisoned on read; recovering inner data");
                poisoned.into_inner()
            }
        }
    }

    /// Acquire a write lock, recovering the guard even if the lock was poisoned.
    pub(crate) fn write_lock_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
        match lock.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("RwLock poisoned on write; recovering inner data");
                poisoned.into_inner()
            }
        }
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

    /// Acquire a read lock — infallible under `spin`.
    pub(crate) fn read_lock_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
        lock.read()
    }

    /// Acquire a write lock — infallible under `spin`.
    pub(crate) fn write_lock_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
        lock.write()
    }
}

// -- OnceLock -----------------------------------------------------------------
#[cfg(feature = "std")]
pub(crate) use std::sync::OnceLock;
/// Read-write lock — [`std::sync::RwLock`] under `std`,
/// [`spin::RwLock`] under `no_std`.
#[cfg(feature = "std")]
pub(crate) use std::sync::RwLock;

pub(crate) use lock::{read_lock, read_lock_recover, write_lock, write_lock_recover};
#[cfg(not(feature = "std"))]
pub(crate) use spin::RwLock;

#[cfg(not(feature = "std"))]
#[derive(Debug, Default)]
pub(crate) struct OnceLock<T>(spin::Once<T>);

#[cfg(not(feature = "std"))]
impl<T> OnceLock<T> {
    pub const fn new() -> Self {
        Self(spin::Once::new())
    }

    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.0.call_once(f)
    }

    pub fn get(&self) -> Option<&T> {
        self.0.get()
    }
}
