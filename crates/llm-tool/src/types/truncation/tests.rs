//! Unit tests for truncation operations.

use alloc::borrow::Cow;

use super::{
    alignment::TruncationAlignment,
    policy::{TruncationPolicy, truncate_middle, truncate_output},
    sink::SpillRef,
};
use crate::types::ToolOutput;

#[test]
fn small_string_unchanged() {
    let policy = TruncationPolicy::new(100);
    let text = "short string within limits";
    let result = truncate_middle(text, &policy);
    assert!(matches!(result, Cow::Borrowed(_)));
    assert_eq!(result, text);
}

#[test]
fn middle_out_truncates_and_inserts_marker() {
    let policy = TruncationPolicy::new(80);
    let text =
        "HEAD_CONTENT_1234567890_MIDDLE_CONTENT_PADDING_TO_EXCEED_LIMIT_TAIL_CONTENT_1234567890";
    let result = truncate_middle(text, &policy);
    assert!(result.len() <= 80);
    assert!(result.starts_with("HEAD_CONTENT"));
    assert!(result.ends_with("TAIL_CONTENT_1234567890"));
    assert!(result.contains("bytes truncated"));
}

#[test]
fn utf8_multi_byte_boundaries_preserved() {
    let crab = "🦀".repeat(50); // 200 bytes
    let policy = TruncationPolicy::new(90);
    let result = truncate_middle(&crab, &policy);
    assert!(result.len() <= 90);
    let count = result.chars().count();
    assert!(count > 0);
    assert!(result.contains("bytes truncated"));
}

#[test]
fn custom_head_tail_ratios() {
    let text = "0123456789".repeat(20); // 200 bytes
    let policy_head = TruncationPolicy::new(80).with_head_ratio(0.9);
    let res_head = truncate_middle(&text, &policy_head);
    assert!(res_head.len() <= 80);

    let policy_tail = TruncationPolicy::new(80).with_head_ratio(0.1);
    let res_tail = truncate_middle(&text, &policy_tail);
    assert!(res_tail.len() <= 80);
}

#[test]
fn custom_marker_format() {
    let policy = TruncationPolicy::new(60).with_marker(" --- cut {count} bytes --- ");
    let text = "A".repeat(200);
    let result = truncate_middle(&text, &policy);
    assert!(result.len() <= 60);
    assert!(result.contains(" --- cut "));
}

#[test]
fn truncate_output_in_place() {
    let mut out = ToolOutput::new("X".repeat(500));
    let policy = TruncationPolicy::new(100);
    truncate_output(&mut out, &policy);
    assert!(out.content().len() <= 100);
    assert!(out.content().contains("bytes truncated"));
}

#[test]
fn test_spill_ref_newtype() {
    let r = SpillRef::new("obs-abc-123");
    assert_eq!(r.as_str(), "obs-abc-123");
    assert_eq!(alloc::format!("{r}"), "obs-abc-123");
    assert_eq!(&*r, "obs-abc-123");
}

#[test]
fn test_line_boundary_default() {
    let policy = TruncationPolicy::new(50);
    assert_eq!(policy.alignment, TruncationAlignment::CharBoundary);
}
