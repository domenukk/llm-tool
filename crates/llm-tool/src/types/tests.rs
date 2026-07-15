//! Unit tests for core types.
//!
//! These are `std`-gated because they exercise lock poisoning, which only
//! occurs with `std::sync::RwLock` (the `spin` locks used under `no_std`
//! cannot be poisoned).

use std::sync::Arc;

use super::{
    Json, PromptOutput, PromptRole, RegistryItem, ResourceOutput, ResourceOutputContent,
    SharedState, ToolContext, ToolError, ToolOutput,
};
use crate::compat::{RwLock, write_lock};

// ── ToolContext: shared state ────────────────────────────────────────

#[test]
fn with_shared_state_shares_store() {
    let store = SharedState::new();
    let a = ToolContext::new()
        .with_conversation_id("conv")
        .with_shared_state(store.clone());
    let b = ToolContext::new().with_shared_state(store.clone());

    a.set_state("k", serde_json::json!(1)).unwrap();
    // `b` observes `a`'s write because they share the same backing store.
    assert_eq!(b.get_state("k", serde_json::json!(0)), serde_json::json!(1));
    assert_eq!(a.conversation_id(), Some("conv"));
    assert_eq!(b.conversation_id(), None);
}

#[test]
fn shared_state_handle_round_trips_via_accessor() {
    let ctx = ToolContext::new();
    ctx.set_state("k", serde_json::json!(7)).unwrap();
    // A handle obtained from `shared_state()` aliases the same store.
    let shared = ctx.shared_state();
    let other = ToolContext::new().with_shared_state(shared);
    assert_eq!(
        other.get_state("k", serde_json::json!(0)),
        serde_json::json!(7)
    );
}

#[test]
fn get_state_returns_default_when_absent() {
    let ctx = ToolContext::new();
    assert_eq!(
        ctx.get_state("missing", serde_json::json!("d")),
        serde_json::json!("d")
    );
}

// ── ToolContext: typed extensions ────────────────────────────────────

#[test]
fn ext_round_trip() {
    #[derive(Clone, PartialEq, Debug)]
    struct Marker(u32);

    let ctx = ToolContext::new();
    assert!(ctx.get_ext::<Marker>().is_none());
    ctx.set_ext(Marker(7)).unwrap();
    assert_eq!(ctx.get_ext::<Marker>(), Some(Marker(7)));
}

// ── Lock poisoning ───────────────────────────────────────────────────

/// Poison the given lock by panicking on a separate thread while holding its
/// write guard, so the panic does not unwind the test itself.
fn poison<T: Send + Sync + 'static>(lock: &Arc<RwLock<T>>) {
    let clone = Arc::clone(lock);
    let handle = std::thread::spawn(move || {
        let guard = write_lock(&clone).expect("first lock acquisition is not poisoned");
        // Panicking while `guard` is alive poisons the lock during unwinding.
        std::hint::black_box(&guard);
        panic!("intentional poison");
    });
    handle
        .join()
        .expect_err("poisoning thread should have panicked");
}

#[test]
fn set_state_errors_on_poisoned_lock() {
    let ctx = ToolContext::new();
    poison(&ctx.state.0);
    let err = ctx.set_state("k", serde_json::json!(1)).unwrap_err();
    assert!(err.to_string().contains("poisoned"));
}

#[test]
fn get_state_returns_default_on_poisoned_lock() {
    let ctx = ToolContext::new();
    poison(&ctx.state.0);
    assert_eq!(
        ctx.get_state("k", serde_json::json!("d")),
        serde_json::json!("d")
    );
}

#[test]
fn set_ext_errors_on_poisoned_lock() {
    let ctx = ToolContext::new();
    poison(&ctx.extensions);
    let err = ctx.set_ext(5u32).unwrap_err();
    assert!(err.to_string().contains("poisoned"));
}

#[test]
fn get_ext_returns_none_on_poisoned_lock() {
    let ctx = ToolContext::new();
    poison(&ctx.extensions);
    assert!(ctx.get_ext::<u32>().is_none());
}

// ── ToolOutput From impls ────────────────────────────────────────────

#[test]
fn tool_output_from_scalars() {
    assert_eq!(ToolOutput::from(42i64).content(), "42");
    assert_eq!(ToolOutput::from(true).content(), "true");
    let f: ToolOutput = 1.5f64.into();
    assert_eq!(f.content(), "1.5");
    let v: ToolOutput = serde_json::json!({"a": 1}).into();
    assert_eq!(v.content(), r#"{"a":1}"#);
}

// ── Json<T> conversion ───────────────────────────────────────────────

#[test]
fn json_object_populates_metadata() {
    #[derive(serde::Serialize)]
    struct M {
        a: i32,
        b: i32,
    }
    let out: ToolOutput = Json(M { a: 1, b: 2 }).into();
    assert_eq!(out.metadata()["a"], 1);
    assert_eq!(out.metadata()["b"], 2);
}

#[test]
fn json_scalar_has_no_metadata() {
    let out: ToolOutput = Json(5i32).into();
    assert_eq!(out.content(), "5");
    assert!(out.metadata().is_empty());
}

#[test]
#[should_panic(expected = "Serialize impl")]
fn json_panics_on_serialize_failure() {
    struct Bad;
    impl serde::Serialize for Bad {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("boom"))
        }
    }
    let out: ToolOutput = Json(Bad).into();
    // Unreachable — the conversion panics above.
    assert_eq!(out.content(), "");
}

// ── ToolError From impls ─────────────────────────────────────────────

#[test]
fn tool_error_from_io_error() {
    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
    let err: ToolError = io.into();
    assert!(err.to_string().contains("nope"));
    assert_eq!(err.metadata()["error_kind"], serde_json::json!("NotFound"));
}

// ── ToolError::not_found / is_not_found ───────────────────────────────

#[test]
fn registry_item_display() {
    assert_eq!(RegistryItem::Tool.to_string(), "tool");
    assert_eq!(RegistryItem::Prompt.to_string(), "prompt");
    assert_eq!(RegistryItem::Resource.to_string(), "resource");
}

#[test]
fn not_found_sets_message_and_predicate() {
    let err = ToolError::not_found(RegistryItem::Tool, "add_nummbers");
    assert!(err.to_string().contains("add_nummbers"));
    assert!(err.to_string().contains("tool"));
    assert!(err.is_not_found());
    assert_eq!(
        err.metadata()[ToolError::ERROR_KIND_KEY],
        serde_json::json!(ToolError::KIND_NOT_REGISTERED)
    );
}

#[test]
fn is_not_found_false_for_plain_error() {
    // A generic execution error must not be mistaken for a registry miss.
    assert!(!ToolError::new("handler blew up").is_not_found());
}

#[test]
fn is_not_found_does_not_collide_with_io_not_found() {
    // `From<io::Error>` also writes `error_kind`, but with the io kind's debug
    // name ("NotFound"), which must NOT be treated as a registry miss.
    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
    let err: ToolError = io.into();
    assert!(!err.is_not_found());
}

#[test]
fn tool_error_from_serde_json_error() {
    let parse_err = serde_json::from_str::<serde_json::Value>("{bad").unwrap_err();
    let err: ToolError = parse_err.into();
    assert!(err.metadata().contains_key("category"));
}

#[test]
fn tool_error_from_boxed_error() {
    let boxed: Box<dyn core::error::Error + Send + Sync> = "boom".into();
    let err: ToolError = boxed.into();
    assert_eq!(err.to_string(), "boom");
}

#[test]
fn tool_error_from_infallible_via_question_mark() {
    fn inner() -> Result<(), ToolError> {
        let ok: Result<(), core::convert::Infallible> = Ok(());
        // Exercises `From<Infallible>` in the `?` desugaring.
        ok?;
        Ok(())
    }
    assert!(inner().is_ok());
}

// ── PromptOutput / ResourceOutput ────────────────────────────────────

#[test]
fn prompt_role_display_and_as_str() {
    assert_eq!(PromptRole::User.as_str(), "user");
    assert_eq!(PromptRole::Assistant.as_str(), "assistant");
    assert_eq!(PromptRole::System.as_str(), "system");
    assert_eq!(PromptRole::User.to_string(), "user");
}

#[test]
fn prompt_output_constructors() {
    let p: PromptOutput = "hi".into();
    assert_eq!(p.messages.len(), 1);
    assert_eq!(p.messages[0].role, PromptRole::User);
    assert_eq!(p.messages[0].role.as_str(), "user");
    assert_eq!(p.messages[0].content, "hi");

    let p2: PromptOutput = String::from("yo").into();
    assert_eq!(p2.messages[0].content, "yo");

    let a = PromptOutput::assistant("resp");
    assert_eq!(a.messages[0].role, PromptRole::Assistant);
    assert_eq!(a.messages[0].content, "resp");

    let s = PromptOutput::system("instructions");
    assert_eq!(s.messages[0].role, PromptRole::System);
    assert_eq!(s.messages[0].content, "instructions");
}

#[test]
fn resource_output_text_variant() {
    let t = ResourceOutput::text("u://a", Some("text/plain"), "body");
    match &t.contents[0] {
        ResourceOutputContent::Text {
            uri,
            mime_type,
            text,
        } => {
            assert_eq!(uri, "u://a");
            assert_eq!(mime_type.as_deref(), Some("text/plain"));
            assert_eq!(text, "body");
        }
        ResourceOutputContent::Blob { .. } => panic!("expected a text content block"),
    }
}

#[test]
fn resource_output_blob_variant() {
    let b = ResourceOutput::blob("u://b", None, "ZGF0YQ==");
    match &b.contents[0] {
        ResourceOutputContent::Blob {
            uri,
            mime_type,
            blob,
        } => {
            assert_eq!(uri, "u://b");
            assert!(mime_type.is_none());
            assert_eq!(blob, "ZGF0YQ==");
        }
        ResourceOutputContent::Text { .. } => panic!("expected a blob content block"),
    }
}
