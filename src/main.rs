use clap::Parser;
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
        true => {
            let bruteforcer = Base64Bruteforcer::<u16>::collect_combinations(example_string);
            log::debug!("Combinations: {:?}", bruteforcer);

            // Creating distinct lines to see results
            bruteforcer
                .into_iter()
                .multi_cartesian_product()
                .map(|sections| sections.concat())
                .map(|line| {
                    String::from_utf16(line.as_slice())
                        .expect("Line did not make a proper utf16 string")
                })
                .for_each(|line| println!("{}", line.escape_default().to_string()));
        }
    }
}
