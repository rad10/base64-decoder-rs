use base64_bruteforcer_rs::phrase::schema::Phrase;
use clap::{Parser, ValueEnum};
use patharg::InputArg;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub(crate) struct ToolArgs {
    /// Sets the string to brute-force
    #[arg(default_value_t)]
    pub(crate) input: InputArg,

    /// Takes an already pulled out schema to bruteforce
    #[arg(short = 's', long)]
    pub(crate) use_schema: bool,

    /// Tells the tools that the string is UTF16
    #[arg(short, long)]
    pub(crate) use_utf16: bool,

    /// Skips printing out all possible values and instead prints out the
    /// combination framework as well as other options such as the number of
    /// combinations the schema provides
    #[arg(short, long)]
    pub(crate) info: bool,

    /// Sets the form of reduction used on the string
    #[arg(short, long, value_enum, default_value_t = ReductionMethod::Halves)]
    pub(crate) reduction_method: ReductionMethod,

    /// Sets which validator is used on the strings
    #[arg(short = 'a', long, value_enum, default_value_t = StringValidator::WhatLang)]
    pub(crate) validation_method: StringValidator,

    #[command(flatten)]
    pub(crate) verbose: clap_verbosity_flag::Verbosity,
}

// Added implementation for fromstring to allow as argument
pub(crate) fn parse_json_to_schema(s: &str) -> Result<Phrase<String>, String> {
    let raw_schema: Result<Vec<Vec<String>>, String> =
        serde_json::from_str(s).map_err(|e| format!("Failed to collect schema: {e}"));
    raw_schema.map(Phrase::from)
}

/// Provides all possible ways to reduce the phrases permutations
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum ReductionMethod {
    /// Uses Reduction by pairs
    Pairs,
    /// Uses reduction by halves
    Halves,
}

/// Determines what function to use when validating part of a phrase
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum StringValidator {
    /// Sets no validation method and skips reduction before printing
    /// possibilities
    None,
    /// Uses the whatlang library to validate strings during reduction
    #[cfg(feature = "whatlang")]
    WhatLang,
}
