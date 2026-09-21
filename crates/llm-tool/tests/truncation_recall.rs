//! Comprehensive integration tests for middle-out truncation, observation spilling,
//! marker formatting, and boundary alignment.

use std::sync::Arc;

use llm_tool::{
    HeadRatio, MarkerPlaceholder, OutputSpillSink, SpillContext, SpillError, SpillRef,
    TruncationAlignment, TruncationPolicy, truncate_middle, truncate_middle_detailed,
    truncate_output_detailed,
};

#[derive(Debug)]
struct MockSpillSink {
    assigned_ref: SpillRef,
}

impl OutputSpillSink for MockSpillSink {
    fn spill(&self, _ctx: &SpillContext<'_>, _full: &str) -> Result<SpillRef, SpillError> {
        Ok(self.assigned_ref.clone())
    }
}

#[derive(Debug)]
struct FailingSpillSink {
    error_message: String,
}

impl OutputSpillSink for FailingSpillSink {
    fn spill(&self, _ctx: &SpillContext<'_>, _full: &str) -> Result<SpillRef, SpillError> {
        Err(SpillError::StorageFailed(self.error_message.clone()))
    }
}

#[test]
fn test_no_sink_output_byte_identical_to_pre_change() {
    let text =
        "HEAD_CONTENT_1234567890_MIDDLE_CONTENT_PADDING_TO_EXCEED_LIMIT_TAIL_CONTENT_1234567890";
    let policy = TruncationPolicy::new(80);

    let result = truncate_middle(text, &policy);
    assert_eq!(result.len(), 80);
    assert_eq!(
        result,
        "HEAD_CONTENT_1234567890_M\n[... 36 bytes truncated ...]\nT_TAIL_CONTENT_1234567890"
    );

    // Custom marker with deprecated placeholder {count} is also byte-identical
    let custom_policy = TruncationPolicy::new(60).with_marker(" --- cut {count} bytes --- ");
    let a_payload = "A".repeat(200);
    let custom_res = truncate_middle(&a_payload, &custom_policy);
    assert!(custom_res.len() <= 60);
    assert!(custom_res.contains(" --- cut "));
    assert!(custom_res.starts_with("AAAAAAAAAA"));
    assert!(custom_res.ends_with("AAAAAAAAAA"));
}

#[test]
fn test_omitted_range_matches_bytes_removed() {
    let input =
        "PREFIX_START_1234567890_VERY_LONG_MIDDLE_PAYLOAD_WITH_MORE_DATA_SUFFIX_END_9876543210";
    let policy = TruncationPolicy::new(60);

    let outcome = truncate_middle_detailed(input, &policy)
        .expect("detailed truncation without sink should succeed");

    assert!(outcome.is_truncated());
    assert_eq!(outcome.original_len, input.len());

    let range = outcome
        .omitted
        .as_ref()
        .expect("omitted range must be present");
    assert_eq!(range.len(), outcome.omitted_bytes());

    let head = &input[..range.start];
    let tail = &input[range.end..];

    assert!(outcome.text.starts_with(head));
    assert!(outcome.text.ends_with(tail));

    // Verify the omitted slice is indeed the middle part
    let omitted_slice = &input[range.clone()];
    assert!(omitted_slice.contains("MIDDLE_PAYLOAD"));
    assert_eq!(
        range.start + range.len() + (input.len() - range.end),
        input.len()
    );
}

#[test]
fn test_long_spill_ref_still_respects_max_bytes() {
    let long_ref_str = "spill-".to_string() + &"z".repeat(300);
    let sink = Arc::new(MockSpillSink {
        assigned_ref: SpillRef::new(long_ref_str),
    });

    let input = "Important Initial Context\n".to_string()
        + &"Payload Line Content\n".repeat(100)
        + "Final Result Summary\n";

    // Test with various constrained byte budgets
    for limit in [80, 100, 150, 200] {
        let policy = TruncationPolicy::new(limit).with_spill_sink(sink.clone());
        let outcome = policy
            .truncate_detailed(&input)
            .expect("truncation with sink should succeed");

        assert!(
            outcome.text.len() <= limit,
            "text length {} exceeded limit {}",
            outcome.text.len(),
            limit
        );
        assert!(outcome.spill_ref.is_some());
        // UTF-8 code points must be intact
        assert!(outcome.text.chars().count() > 0);
    }
}

#[test]
fn test_line_boundary_never_emits_partial_line() {
    let lines = [
        "line 1: Alpha beta gamma delta",
        "line 2: Epsilon zeta eta theta",
        "line 3: Iota kappa lambda mu",
        "line 4: Nu xi omicron pi",
        "line 5: Rho sigma tau upsilon",
        "line 6: Phi chi psi omega",
    ];
    let input = lines.join("\n") + "\n";

    let policy = TruncationPolicy::new(130).with_alignment(TruncationAlignment::LineBoundary);
    let outcome = policy
        .truncate_detailed(&input)
        .expect("line boundary truncation should succeed");

    assert!(outcome.text.len() <= 130);
    assert!(outcome.is_truncated());

    // Split outcome into lines, filtering out marker lines that begin with "[" or whitespace
    for line in outcome.text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[...") || trimmed.ends_with("...]") || trimmed.is_empty() {
            continue;
        }
        assert!(
            lines.contains(&line),
            "Emitted partial or corrupted line: '{line}'"
        );
    }
}

#[test]
fn test_line_boundary_single_line_degrades_gracefully() {
    let single_line = "A".repeat(5000);
    let policy = TruncationPolicy::new(100).with_alignment(TruncationAlignment::LineBoundary);

    let outcome = policy
        .truncate_detailed(&single_line)
        .expect("truncation on single line should succeed");

    assert!(outcome.text.len() <= 100);
    assert!(outcome.is_truncated());
    assert!(outcome.text.contains("bytes truncated"));
}

#[test]
fn test_spill_failure_surfaced_not_swallowed() {
    let sink = Arc::new(FailingSpillSink {
        error_message: "storage partition is read-only".into(),
    });

    let policy = TruncationPolicy::new(400).with_spill_sink(sink);
    let input = "X".repeat(1000);

    // truncate_detailed must return Err, never silently fall back
    match policy.truncate_detailed(&input) {
        Err(SpillError::StorageFailed(msg)) => {
            assert_eq!(msg, "storage partition is read-only");
        }
        Ok(_) => panic!("expected spill failure to be surfaced, but got Ok"),
        Err(other) => panic!("unexpected error: {other:?}"),
    }

    // truncate() must not panic, and must not pretend the bytes were retained:
    // the failure is rendered into the marker itself.
    let payload = "X".repeat(1000);
    let text = policy.truncate(&payload);
    assert!(
        text.contains("could NOT be retained"),
        "spill failure must be visible in the marker, got: {text}"
    );
    assert!(
        text.contains("storage partition is read-only"),
        "marker must name the underlying cause, got: {text}"
    );
    assert!(
        text.contains("unrecoverable"),
        "marker must state the omitted bytes are unrecoverable, got: {text}"
    );
    assert!(
        !text.contains("read_observation"),
        "marker must not advertise recall when nothing was retained, got: {text}"
    );
    assert!(
        text.len() <= 400,
        "truncate() must still respect max_bytes, got {} bytes",
        text.len()
    );

    // A custom marker template that never mentions the failure must still carry the warning.
    let templated = TruncationPolicy::new(300)
        .with_marker("[cut {omitted_bytes} of {original_bytes}, ref={ref}]")
        .with_spill_sink(Arc::new(FailingSpillSink {
            error_message: "storage partition is read-only".into(),
        }));
    let templated_text = templated.truncate(&payload);
    assert!(
        templated_text.contains("could NOT be retained"),
        "custom templates must not be able to hide spill failure, got: {templated_text}"
    );
}

#[test]
fn test_tight_budget_never_truncates_silently() {
    let payload = "X".repeat(1000);

    // Budget too small for the informative marker: content is sacrificed, not the notice.
    let policy = TruncationPolicy::new(40);
    let text = policy.truncate(&payload);
    assert!(
        text.contains("truncated"),
        "a tight budget must still announce truncation, got: {text}"
    );
    assert!(text.len() <= 40, "got {} bytes", text.len());

    // Budget too small even for the minimal marker: emit a partial marker, never a
    // bare slice the model would read as complete output.
    let tiny = TruncationPolicy::new(8);
    let tiny_text = tiny.truncate(&payload);
    assert!(
        !tiny_text.starts_with('X'),
        "must not emit a bare, complete-looking slice, got: {tiny_text}"
    );
    assert!(tiny_text.len() <= 8, "got {} bytes", tiny_text.len());

    // A tight budget with a failed sink must still flag that nothing was retained.
    let failing = TruncationPolicy::new(40).with_spill_sink(Arc::new(FailingSpillSink {
        error_message: "storage partition is read-only".into(),
    }));
    let failed_text = failing.truncate(&payload);
    assert!(
        failed_text.contains("NOT retained"),
        "tight budget must still flag retention loss, got: {failed_text}"
    );
}

#[test]
fn test_spill_failure_marker_omitted_when_no_sink_configured() {
    let policy = TruncationPolicy::new(80);
    let payload = "Y".repeat(300);
    let text = policy.truncate(&payload);

    assert!(text.contains("bytes truncated"));
    assert!(
        !text.contains("could NOT be retained"),
        "no sink configured means nothing was promised; no failure warning belongs here"
    );
}

#[test]
fn test_deprecated_and_new_placeholders_substitution() {
    let input = "0123456789".repeat(30); // 300 bytes

    // Legacy {count}
    let p_count = TruncationPolicy::new(80).with_marker("[cut {count}B]");
    let res_count = truncate_middle(&input, &p_count);
    assert!(res_count.contains("[cut "));
    assert!(res_count.contains("B]"));

    // Legacy {bytes}
    let p_bytes = TruncationPolicy::new(80).with_marker("[cut {bytes}B]");
    let res_bytes = truncate_middle(&input, &p_bytes);
    assert!(res_bytes.contains("[cut "));

    // Legacy <N>
    let p_n = TruncationPolicy::new(80).with_marker("[cut <N>B]");
    let res_n = truncate_middle(&input, &p_n);
    assert!(res_n.contains("[cut "));

    // Canonical new placeholders: {omitted_bytes}, {original_bytes}, {offset_start}, {offset_end}
    let p_canonical = TruncationPolicy::new(120).with_marker(
        "[{omitted_bytes} of {original_bytes} cut between {offset_start}..{offset_end}]",
    );
    let res_canonical = truncate_middle(&input, &p_canonical);
    assert!(res_canonical.contains("of 300 cut between"));

    // Verify token methods
    assert_eq!(
        MarkerPlaceholder::OmittedBytes.token(),
        MarkerPlaceholder::TOKEN_OMITTED_BYTES
    );
    assert_eq!(
        MarkerPlaceholder::SpillRef.token(),
        MarkerPlaceholder::TOKEN_SPILL_REF
    );
}

#[test]
fn test_sink_configured_default_marker_format() {
    let sink = Arc::new(MockSpillSink {
        assigned_ref: SpillRef::new("obs-test-42"),
    });

    let input = "HEADER_LINE_0123456789\n".to_string()
        + &"INTERMEDIATE_DATA_BLOCK\n".repeat(20)
        + "FOOTER_LINE_9876543210\n";

    let policy = TruncationPolicy::new(180).with_spill_sink(sink);
    let outcome = policy
        .truncate_detailed(&input)
        .expect("detailed truncation should succeed");

    assert!(
        outcome.text.contains(
            "Full output retained as ref=\"obs-test-42\" — recall with read_observation."
        )
    );
    assert!(outcome.text.contains("bytes omitted"));
    assert_eq!(
        outcome.spill_ref.as_ref().map(SpillRef::as_str),
        Some("obs-test-42")
    );
}

#[test]
fn test_tool_output_detailed_truncation() {
    let mut output = llm_tool::ToolOutput::new("Z".repeat(400));
    let policy = TruncationPolicy::new(80);

    let outcome = truncate_output_detailed(&mut output, &policy)
        .expect("truncate_output_detailed should succeed");

    assert!(outcome.is_truncated());
    assert_eq!(output.content(), outcome.text);
    assert!(output.content().len() <= 80);
}

#[test]
fn test_head_ratio_helpers() {
    let hr_balanced = HeadRatio::BALANCED;
    assert_eq!(hr_balanced.compute_head_budget(100), 50);

    let hr_75 = HeadRatio::from_percent(75);
    assert_eq!(hr_75.compute_head_budget(100), 75);

    let hr_permille = HeadRatio::from_permille(300);
    assert_eq!(hr_permille.compute_head_budget(100), 30);

    let hr_f64 = HeadRatio::from_f64(0.2);
    assert_eq!(hr_f64.compute_head_budget(100), 20);
}
