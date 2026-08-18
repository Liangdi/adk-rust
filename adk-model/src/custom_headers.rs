//! Shared extra-headers plumbing for HTTP model providers.
//!
//! Providers fronted by a gateway or router often need identity or routing
//! headers stamped on every LLM request (e.g. an `X-Agent-Id` header a
//! supervisor's router reads for attribution). Each provider config carries
//! them as ordered `(name, value)` pairs — the multi-line `Name: Value`
//! text itself is parsed by the embedding application (agentx maps
//! `AGENTX_CUSTOM_HEADERS` onto this) — and clients parse the pairs into a
//! `HeaderMap` once at construction, so an invalid header fails the build
//! loudly instead of panicking per request (`RequestBuilder::header` with a
//! raw string panics on invalid names).

use adk_core::{AdkError, ErrorCategory, ErrorComponent};

/// Parse ordered `(name, value)` pairs into a [`reqwest::header::HeaderMap`].
///
/// Surrounding whitespace is trimmed; duplicate names keep the last value.
/// Invalid header names (e.g. containing spaces) or values (e.g. containing
/// control characters) are construction errors — callers surface them before
/// any request is sent.
pub fn parse_extra_headers(
    pairs: &[(String, String)],
) -> Result<reqwest::header::HeaderMap, AdkError> {
    let mut map = reqwest::header::HeaderMap::new();
    for (raw_name, value) in pairs {
        // Keep the user-written spelling for errors; `HeaderName` itself
        // lowercases on parse.
        let raw_name = raw_name.trim();
        let name = reqwest::header::HeaderName::try_from(raw_name).map_err(|e| {
            AdkError::new(
                ErrorComponent::Model,
                ErrorCategory::InvalidInput,
                "model.headers.name",
                format!("invalid custom header name '{raw_name}': {e}"),
            )
        })?;
        let value = reqwest::header::HeaderValue::try_from(value.trim()).map_err(|e| {
            AdkError::new(
                ErrorComponent::Model,
                ErrorCategory::InvalidInput,
                "model.headers.value",
                format!("invalid custom header value for '{raw_name}': {e}"),
            )
        })?;
        map.insert(name, value);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_parse_with_last_wins_and_trimming() {
        let map = parse_extra_headers(&[
            ("  X-Agent-Id ".to_string(), " agent-7 ".to_string()),
            ("X-Group-Id".to_string(), "g1".to_string()),
            ("X-Group-Id".to_string(), "g2".to_string()),
        ])
        .expect("valid pairs parse");
        assert_eq!(map.get("x-agent-id").unwrap(), "agent-7");
        assert_eq!(map.get("x-group-id").unwrap(), "g2");
    }

    #[test]
    fn invalid_name_is_a_construction_error() {
        let err = parse_extra_headers(&[("Bad Header".to_string(), "v".to_string())])
            .expect_err("space in header name is invalid");
        assert!(err.to_string().contains("Bad Header"));
    }

    #[test]
    fn invalid_value_is_a_construction_error() {
        let err = parse_extra_headers(&[("X-K".to_string(), "bad\u{7}value".to_string())])
            .expect_err("control character in value is invalid");
        assert!(err.to_string().contains("X-K"));
    }
}
