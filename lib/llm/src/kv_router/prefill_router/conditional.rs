// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional prefill/decode routing: bypass remote prefill when local decode is cheaper.
//!
//! Ported from the `conditional-pd` Python prototype in the parent workspace.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::router as env_router;

use crate::protocols::common::{llm_backend::PreprocessedRequest, timing::RequestTracker};

static BYPASS_TOTAL: AtomicU64 = AtomicU64::new(0);

pub static CONDITIONAL_PD_CONFIG: OnceLock<ConditionalPdConfig> = OnceLock::new();

pub fn conditional_pd_config() -> &'static ConditionalPdConfig {
    CONDITIONAL_PD_CONFIG.get_or_init(ConditionalPdConfig::from_env)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BypassReason {
    Disabled,
    ShortPrompt,
    LongPromptRemote,
    HighKvHit,
    BusyPrefillQueue,
    LocalCostLeRemote,
    RemoteCostLtLocal,
}

impl BypassReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ShortPrompt => "uncached_tokens_below_min_remote_threshold",
            Self::LongPromptRemote => "uncached_tokens_above_force_remote_threshold",
            Self::HighKvHit => "kv_hit_ratio_above_local_threshold",
            Self::BusyPrefillQueue => "prefill_queue_above_remote_threshold",
            Self::LocalCostLeRemote => "local_cost_le_remote_cost",
            Self::RemoteCostLtLocal => "remote_cost_lt_local_cost",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConditionalPdConfig {
    pub enabled: bool,
    pub min_uncached_for_remote: usize,
    pub force_remote_above_uncached: Option<usize>,
    pub min_kv_hit_ratio_for_local: Option<f64>,
    pub max_prefill_queue_tokens_for_remote: Option<f64>,
    pub local_prefill_ms_per_token: f64,
    pub remote_prefill_ms_per_token: f64,
    pub kv_transfer_ms: f64,
    pub decode_queue_ms_per_token: f64,
    pub prefill_queue_ms_per_token: f64,
}

impl Default for ConditionalPdConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_uncached_for_remote: 64,
            force_remote_above_uncached: Some(512),
            min_kv_hit_ratio_for_local: Some(0.75),
            max_prefill_queue_tokens_for_remote: Some(256.0),
            local_prefill_ms_per_token: 0.08,
            remote_prefill_ms_per_token: 0.05,
            kv_transfer_ms: 12.0,
            decode_queue_ms_per_token: 0.002,
            prefill_queue_ms_per_token: 0.004,
        }
    }
}

impl ConditionalPdConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.enabled = parse_bool_env(env_router::DYN_CONDITIONAL_PD, false);
        config.min_uncached_for_remote =
            parse_usize_env(env_router::DYN_CONDITIONAL_PD_MIN_UNCACHED_FOR_REMOTE, 64);
        config.force_remote_above_uncached = parse_optional_usize_env(
            env_router::DYN_CONDITIONAL_PD_FORCE_REMOTE_ABOVE_UNCACHED,
            Some(512),
        );
        config.min_kv_hit_ratio_for_local = parse_optional_f64_env(
            env_router::DYN_CONDITIONAL_PD_MIN_KV_HIT_RATIO_FOR_LOCAL,
            Some(0.75),
        );
        config.max_prefill_queue_tokens_for_remote = parse_optional_f64_env(
            env_router::DYN_CONDITIONAL_PD_MAX_PREFILL_QUEUE_FOR_REMOTE,
            Some(256.0),
        );
        config
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RequestSignals {
    pub uncached_tokens: usize,
    pub cached_tokens: usize,
    pub decode_queue_tokens: f64,
    pub prefill_queue_tokens: f64,
}

impl RequestSignals {
    pub fn kv_hit_ratio(self) -> f64 {
        let total = self.uncached_tokens + self.cached_tokens;
        if total == 0 {
            0.0
        } else {
            self.cached_tokens as f64 / total as f64
        }
    }
}

pub fn extract_request_signals(req: &PreprocessedRequest) -> RequestSignals {
    let total_tokens = req.token_ids.len();
    let cached_tokens = req
        .tracker
        .as_ref()
        .and_then(|tracker| tracker.cached_tokens())
        .or_else(|| {
            req.tracker.as_ref().and_then(|tracker| {
                tracker
                    .kv_hit_rate()
                    .map(|rate| (rate * total_tokens as f64).round() as usize)
            })
        })
        .unwrap_or(0)
        .min(total_tokens);
    let uncached_tokens = total_tokens.saturating_sub(cached_tokens);

    RequestSignals {
        uncached_tokens,
        cached_tokens,
        decode_queue_tokens: 0.0,
        prefill_queue_tokens: 0.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostBreakdown {
    pub local_cost: f64,
    pub remote_cost: f64,
}

impl CostBreakdown {
    pub fn chosen_is_local(self) -> bool {
        self.local_cost <= self.remote_cost
    }
}

pub fn estimate_costs(config: &ConditionalPdConfig, signals: RequestSignals) -> CostBreakdown {
    let uncached = signals.uncached_tokens as f64;
    let decode_wait = signals.decode_queue_tokens * config.decode_queue_ms_per_token;
    let prefill_wait = signals.prefill_queue_tokens * config.prefill_queue_ms_per_token;
    let local_prefill = uncached * config.local_prefill_ms_per_token;
    let remote_prefill = uncached * config.remote_prefill_ms_per_token;
    let cache_discount = signals.kv_hit_ratio() * 0.6 * remote_prefill;

    let local_cost = local_prefill + 0.35 * decode_wait - cache_discount;
    let remote_cost =
        remote_prefill + 0.45 * prefill_wait + config.kv_transfer_ms + 0.35 * decode_wait
            - cache_discount;

    CostBreakdown {
        local_cost,
        remote_cost,
    }
}

pub fn should_bypass_local_prefill(
    config: &ConditionalPdConfig,
    signals: RequestSignals,
) -> Option<BypassReason> {
    if !config.enabled {
        return None;
    }

    if signals.uncached_tokens <= config.min_uncached_for_remote {
        return Some(BypassReason::ShortPrompt);
    }

    if config
        .force_remote_above_uncached
        .is_some_and(|threshold| signals.uncached_tokens >= threshold)
    {
        return None;
    }

    if config
        .min_kv_hit_ratio_for_local
        .is_some_and(|threshold| signals.kv_hit_ratio() >= threshold)
    {
        return Some(BypassReason::HighKvHit);
    }

    if config
        .max_prefill_queue_tokens_for_remote
        .is_some_and(|threshold| signals.prefill_queue_tokens > threshold)
    {
        return Some(BypassReason::BusyPrefillQueue);
    }

    let costs = estimate_costs(config, signals);
    if costs.chosen_is_local() {
        Some(BypassReason::LocalCostLeRemote)
    } else {
        None
    }
}

pub fn record_bypass(reason: BypassReason) {
    BYPASS_TOTAL.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(
        reason = reason.as_str(),
        bypass_total = BYPASS_TOTAL.load(Ordering::Relaxed),
        "conditional PD bypass: routing directly to decode"
    );
}

pub fn bypass_total() -> u64 {
    BYPASS_TOTAL.load(Ordering::Relaxed)
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_optional_usize_env(name: &str, default: Option<usize>) -> Option<usize> {
    match std::env::var(name) {
        Ok(value) if value.eq_ignore_ascii_case("none") || value.is_empty() => None,
        Ok(value) => value.parse().ok().or(default),
        Err(_) => default,
    }
}

fn parse_optional_f64_env(name: &str, default: Option<f64>) -> Option<f64> {
    match std::env::var(name) {
        Ok(value) if value.eq_ignore_ascii_case("none") || value.is_empty() => None,
        Ok(value) => value.parse().ok().or(default),
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompt_bypasses_when_enabled() {
        let config = ConditionalPdConfig {
            enabled: true,
            ..Default::default()
        };
        let signals = RequestSignals {
            uncached_tokens: 32,
            cached_tokens: 512,
            decode_queue_tokens: 0.0,
            prefill_queue_tokens: 0.0,
        };
        assert_eq!(
            should_bypass_local_prefill(&config, signals),
            Some(BypassReason::ShortPrompt)
        );
    }

    #[test]
    fn disabled_never_bypasses() {
        let config = ConditionalPdConfig::default();
        let signals = RequestSignals {
            uncached_tokens: 16,
            cached_tokens: 0,
            decode_queue_tokens: 0.0,
            prefill_queue_tokens: 0.0,
        };
        assert_eq!(should_bypass_local_prefill(&config, signals), None);
    }

    #[test]
    fn long_prompt_forces_remote_prefill() {
        let config = ConditionalPdConfig {
            enabled: true,
            min_uncached_for_remote: 0,
            force_remote_above_uncached: Some(512),
            min_kv_hit_ratio_for_local: None,
            ..Default::default()
        };
        let signals = RequestSignals {
            uncached_tokens: 1024,
            cached_tokens: 0,
            decode_queue_tokens: 0.0,
            prefill_queue_tokens: 0.0,
        };
        assert_eq!(should_bypass_local_prefill(&config, signals), None);
    }
}
