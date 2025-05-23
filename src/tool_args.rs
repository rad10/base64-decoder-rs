use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub(crate) struct ToolArgs {
    /// Sets the string to brute-force
    pub(crate) b64_string: String,

    /// Tells the tools that the string is UTF16
    #[arg(short, long)]
    pub(crate) use_utf16: bool,

    #[command(flatten)]
    pub(crate) verbose: clap_verbosity_flag::Verbosity,
}
