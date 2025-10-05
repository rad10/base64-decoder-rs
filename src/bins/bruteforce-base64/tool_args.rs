use base64_bruteforcer_rs::phrase_reduction::Phrase;
use clap::{Args, Parser};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub(crate) struct ToolArgs {
    /// Determines the input type the tool will take
    #[command(flatten)]
    pub(crate) input: ActionType,

    /// Tells the tools that the string is UTF16
    #[arg(short, long)]
    pub(crate) use_utf16: bool,

    /// Skips printing out all possible values and instead prints out the
    /// combination framework as well as other options such as the number of
    /// combinations the schema provides
    #[arg(short, long)]
    pub(crate) info: bool,

    /// Skips human inferences to reduce possible combinations
    #[arg(short, long)]
    pub(crate) no_prune: bool,

    #[command(flatten)]
    pub(crate) verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
pub(crate) struct ActionType {
    /// Sets the string to brute-force
    pub(crate) b64_string: Option<String>,

    /// Takes an already pulled out schema to bruteforce
    #[arg(short = 's', long, value_parser = parse_json_to_schema)]
    pub(crate) use_schema: Option<Phrase<String>>,
}

// Added implementation for fromstring to allow as argument

fn parse_json_to_schema(s: &str) -> Result<Phrase<String>, String> {
    let raw_schema: Result<Vec<Vec<String>>, String> =
            serde_json::from_str(s).map_err(|e| format!("Failed to collect schema: {e}"));
        raw_schema.map(Phrase::from)
}
