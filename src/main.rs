use clap::Parser;
use decoding_guesser::get_valid_combinations;
use itertools::Itertools;
use tool_args::ToolArgs;

mod decoding_guesser;
mod tool_args;

fn main() {
    let parser = ToolArgs::parse();

    env_logger::builder()
        .filter_level(parser.verbose.log_level_filter())
        .init();

    // set base64 string as bytes
    let example_string = parser.b64_string.as_bytes();

    // TODO: utf-16 fails in this process. Need an argument that can determine the chunks to get 3 characters each
    let _ = &example_string
        .iter()
        .chunks(4)
        .into_iter()
        .map(|piece| {
            let combinations = get_valid_combinations(piece.map(|c| c.to_owned()).collect_vec().as_slice())
                .filter(|b64_output| b64_output.is_ascii()) // Checking that all characters are ascii notation
                // .filter(|b64_output| !b64_output.escape_ascii().to_string().contains("\\x")) // Fails if the escaped string contains any escape characters
                .collect_vec();

            // Replacing with a single ??? if options ends up empty
            if combinations.is_empty() {
                vec![b"???".to_vec()]
            } else {
                combinations
            }
        })
        .multi_cartesian_product()
        .map(|combinations| combinations.concat())
        .for_each(|flatten| {
            println!("{}", flatten.escape_ascii().to_string());
        });
}
