use clap::Parser;
use itertools::Itertools;
use tool_args::ToolArgs;

use crate::base64_parser::Base64Bruteforcer;

mod base64_parser;
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
            let mut bruteforcer = Base64Bruteforcer::<u8>::default();
            bruteforcer.collect_combinations(example_string);
            log::debug!(
                "Combinations: {:?}",
                bruteforcer
                    .combinations
                    .iter()
                    .map(|section| section
                        .iter()
                        .map(|variation| variation.escape_ascii().to_string())
                        .collect_vec())
                    .collect_vec()
            );

            // Creating distinct lines to see results
            bruteforcer
                .combinations
                .into_iter()
                .multi_cartesian_product()
                .map(|sections| sections.concat())
                .for_each(|line| println!("{}", line.escape_ascii().to_string()));
        }
        true => {
            let mut bruteforcer = Base64Bruteforcer::<u16>::default();
            bruteforcer.collect_combinations(example_string);
            log::debug!(
                "Combinations: {:?}",
                bruteforcer
                    .combinations
                    .iter()
                    .map(|section| section
                        .iter()
                        .map(|variation| String::from_utf16(variation.as_slice())
                            .expect("Section is not utf16")
                            .escape_default()
                            .to_string())
                        .collect_vec())
                    .collect_vec()
            );

            // Creating distinct lines to see results
            bruteforcer
                .combinations
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
