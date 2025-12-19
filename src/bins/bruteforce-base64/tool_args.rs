use base64_bruteforcer_rs::phrase::schema::Phrase;
#[cfg(feature = "ollama")]
use clap::Args;
use clap::{Parser, Subcommand, ValueEnum};
use patharg::InputArg;
#[cfg(feature = "ollama")]
use url::Url;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub(crate) struct ToolArgs {
    /// Sets the string to brute-force
    #[arg(default_value_t)]
    pub(crate) input: InputArg,

    /// Takes an already pulled out schema to bruteforce
    #[arg(short = 's', long)]
    pub(crate) use_schema: bool,

    /// Tells the tool what format the underlying string is encoded in
    #[arg(short, long, value_enum, default_value_t = UtfFormat::UTF8)]
    pub(crate) format: UtfFormat,

    /// Skips printing out all possible values and instead prints out the
    /// combination framework as well as other options such as the number of
    /// combinations the schema provides
    #[arg(short, long)]
    pub(crate) info: bool,

    /// Sets the form of reduction used on the string
    #[arg(short, long, value_enum, default_value_t = ReductionMethod::Halves)]
    pub(crate) reduction_method: ReductionMethod,

    /// Sets which validator is used on the strings
    #[command(subcommand)]
    pub(crate) validation_method: StringValidator,

    #[command(flatten)]
    pub(crate) verbose: clap_verbosity_flag::Verbosity,
}

// Added implementation for fromstring to allow as argument
pub(crate) fn parse_json_to_schema(s: &str) -> Result<Phrase<String>, String> {
    let raw_schema: Result<Vec<Vec<String>>, String> =
        serde_json::from_str(s).map_err(|e| format!("Failed to collect schema: {e}"));
    raw_schema.map(Phrase::from_iter)
}

/// Provides all possible ways to reduce the phrases permutations
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum ReductionMethod {
    /// Uses Reduction by pairs
    Pairs,
    /// Uses reduction by halves
    Halves,
}

/// Specifies the UTF format the given text is in
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum UtfFormat {
    /// Expected the underlying text to be UTF8
    UTF8,
    /// Expected the underlying text to be UTF16LE
    ///
    /// This is common on strings encoded on windows systems
    UTF16LE,
    /// Expected the underlying text to be UTF32LE
    UTF32LE,
}

/// Determines what function to use when validating part of a phrase
#[derive(Clone, PartialEq, Subcommand)]
pub(crate) enum StringValidator {
    /// Sets no validation method and skips reduction before printing
    /// possibilities
    None,
    /// Uses the whatlang library to validate strings during reduction
    #[cfg(feature = "whatlang")]
    WhatLang,
    /// Use Ollama to give a confidence value on each string
    #[cfg(feature = "ollama")]
    OllamaGroup(OllamaArgs),
}

/// Contains the arguments used to configure and access ollama
#[cfg(feature = "ollama")]
#[derive(Args, Clone, PartialEq)]
pub(crate) struct OllamaArgs {
    /// The address of the ollama instance to connect to
    pub(crate) address: Url,
    /// The model to use when doing validation checks
    pub(crate) model: String,
}
