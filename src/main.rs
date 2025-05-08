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
            let piece_vec = piece.map(|c| c.to_owned()).collect_vec();
            let combinations = get_valid_combinations(piece_vec.as_slice())
                .filter(|b64_output| b64_output.is_ascii()) // Checking that all characters are ascii notation
                .filter(|char_set| {
                    !parser.use_utf16 // Sets this filter if tool is set to check for utf8
                            || char_set.len() == 2
                            || (char_set.len() == 3 && ((char_set[0] == b'\0'
                                && char_set[1] != b'\0'
                                && char_set[2] == b'\0')
                            || (char_set[0] != b'\0'
                                && char_set[1] == b'\0'
                                && char_set[2] != b'\0')))
                })
                // .filter(|b64_output| !b64_output.escape_ascii().to_string().contains("\\x")) // Fails if the escaped string contains any escape characters
                .collect_vec();
            // filters for utf16 if argument is set

            log::debug!(
                "{} -> {:?}",
                piece_vec.escape_ascii().to_string(),
                combinations
                    .iter()
                    .map(|combo| combo.escape_ascii().to_string())
                    .collect_vec()
            );
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
