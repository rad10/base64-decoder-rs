//! This module handles all functions used to analyze data and determine
//! validity and accuracy in terms of what the user desires

use std::fmt::Display;

use crate::phrase::schema::Variation;

/// Provides a confidence on a given string using the whatlang library
pub fn validate_with_whatlang<T>(text: &Variation<T>) -> f64
where
    Variation<T>: Display,
{
    match whatlang::detect(text.to_string().as_str()) {
        Some(info) => info.confidence(),
        None => 0.0,
    }
}
