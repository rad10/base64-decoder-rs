//! This module handles all functions used to analyze data and determine
//! validity and accuracy in terms of what the user desires

/// Provides a confidence on a given string using the whatlang library
pub fn validate_with_whatlang(text: String) -> f64 {
    match whatlang::detect(text.as_str()) {
        Some(info) => info.confidence(),
        None => 0.0,
    }
}
