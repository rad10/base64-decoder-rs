use clap::Parser;
use decoding_guesser::{DISALLOWED_ASCII};
use itertools::Itertools;
use tool_args::ToolArgs;

use crate::decoding_guesser::Base64Bruteforcer;

mod decoding_guesser;
mod tool_args;

fn main() {
    let parser = ToolArgs::parse();

    env_logger::builder()
        .filter_level(parser.verbose.log_level_filter())
        .init();

    // set base64 string as bytes
    let example_string = parser.b64_string.as_bytes();

    match parser.use_utf16 {
        false => {
            let bruteforcer = Base64Bruteforcer::<u8>::collect_combinations(example_string);
            log::debug!("Combinations: {:?}", bruteforcer);

            // Creating distinct lines to see results
            bruteforcer
                .into_iter()
                .multi_cartesian_product()
                .map(|sections| sections.concat())
                .for_each(|line| println!("{}", line.escape_ascii().to_string()));
        }
        true => todo!(),
    }
    return;

    let _ = &example_string
        .iter()
        .chunks(4)
        .into_iter()
        .map(|piece| {
            let piece_vec = piece.map(|c| c.to_owned()).collect_vec();
            let combinations = get_valid_combinations(piece_vec.as_slice())
                .filter(|b64_output| b64_output.is_ascii()) // Checking that all characters are ascii notation
                .filter(|skip_control_codes| skip_control_codes.iter().all(|checked_char| DISALLOWED_ASCII.binary_search(checked_char).is_err())) // Filter out combinations if they contain control characters
                .filter(|char_set| {
                    !parser.use_utf16 // Sets this filter if tool is set to check for utf8
                            || char_set.len() < 3
                            || ((char_set[0] == b'\0'
                                && char_set[1] != b'\0'
                                && char_set[2] == b'\0')
                            || (char_set[0] != b'\0'
                                && char_set[1] == b'\0'
                                && char_set[2] != b'\0'))
                })
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
        .map(|format_utf16| {
            if parser.use_utf16 {
                // If its UTF16, then trim all \0 characters for better readability
                match String::from_utf8(format_utf16.clone()) {
                    Ok(formatted) => {
                        // Successfully converted to UTF8. Trimming null characters
                        formatted.replace("\0", "").into_bytes()
                    }
                    Err(_) => format_utf16,
                }
            } else {
                format_utf16
            }
        })
        .for_each(|flatten| {
            println!("{}", flatten.escape_ascii().to_string());
        });
}
