//! This module handles all functions used to analyze data and determine
//! validity and accuracy in terms of what the user desires

use std::fmt::Display;

#[cfg(feature = "whatlang")]
use crate::phrase::schema::variation::Variation;

#[cfg(feature = "ollama")]
pub mod ollama;

/// Provides a confidence on a given string using the whatlang library
#[cfg(feature = "whatlang")]
pub fn validate_with_whatlang<T>(text: &Variation<T>) -> f64
where
    Variation<T>: Display,
{
    match whatlang::detect(text.to_string().as_str()) {
        Some(info) => info.confidence(),
        None => 0.0,
    }
}
