//! Head/tail budget allocation ratio.

use alloc::format;

/// Ratio of available content budget assigned to the head portion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadRatio {
    numerator: u32,
    denominator: u32,
}

impl HeadRatio {
    /// 50/50 balance between head and tail.
    pub const BALANCED: Self = Self {
        numerator: 1,
        denominator: 2,
    };
    /// All content budget allocated to the head.
    pub const HEAD_ONLY: Self = Self {
        numerator: 1,
        denominator: 1,
    };
    /// All content budget allocated to the tail.
    pub const TAIL_ONLY: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// Construct a ratio from a numerator and denominator.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        if denominator == 0 {
            Self::BALANCED
        } else {
            Self {
                numerator: if numerator > denominator {
                    denominator
                } else {
                    numerator
                },
                denominator,
            }
        }
    }

    /// Construct a ratio from a percentage (0..=100).
    #[must_use]
    pub const fn from_percent(percent: u32) -> Self {
        Self::new(percent, 100)
    }

    /// Construct a ratio from parts per thousand (0..=1000).
    #[must_use]
    pub const fn from_permille(permille: u32) -> Self {
        Self::new(permille, 1000)
    }

    /// Construct a ratio from a floating point value in `[0.0, 1.0]`.
    #[must_use]
    pub fn from_f64(ratio: f64) -> Self {
        if ratio.is_nan() || ratio <= 0.0 {
            Self::TAIL_ONLY
        } else if ratio >= 1.0 {
            Self::HEAD_ONLY
        } else {
            let clamped = ratio.clamp(0.0, 1.0);
            let s = format!("{clamped:.4}");
            match s.strip_prefix("0.") {
                Some(dec) => match dec.parse::<u32>() {
                    Ok(val) => Self::new(val, 10_000),
                    Err(e) => {
                        tracing::debug!(error = %e, "failed to parse head ratio digits");
                        Self::BALANCED
                    }
                },
                None => Self::BALANCED,
            }
        }
    }

    /// Compute the portion of `total_budget` assigned to the head.
    #[must_use]
    pub fn compute_head_budget(&self, total_budget: usize) -> usize {
        if self.denominator == 0 {
            return total_budget / 2;
        }
        let total = u128::try_from(total_budget).unwrap_or(u128::MAX);
        let num = u128::from(self.numerator);
        let den = u128::from(self.denominator);
        let head = (total.saturating_mul(num)) / den;
        usize::try_from(head)
            .unwrap_or(total_budget)
            .min(total_budget)
    }
}

impl Default for HeadRatio {
    fn default() -> Self {
        Self::BALANCED
    }
}

impl From<f64> for HeadRatio {
    fn from(r: f64) -> Self {
        Self::from_f64(r)
    }
}
