use clap::Parser;
use tool_args::ToolArgs;

use crate::{base64_parser::{Base64Bruteforcer, BruteforcerTraits}, phrase_solving::SchemaReduce};

mod base64_parser;
mod phrase_solving;
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

            if !parser.no_prune {
                log::info!(
                    "schema: {:?}\n# of permutations: {}",
                    bruteforcer.convert_to_string(),
                    bruteforcer.permutations()
                );

                log::info!("Reducing permutations to logical choices");
                bruteforcer.reduce_to_end();
            }

            if parser.info {
                println!(
                    "schema: {:?}\n# of permutations: {}",
                    bruteforcer.convert_to_string(),
                    bruteforcer.permutations()
                );
                return;
            }

            // Creating distinct lines to see results
            bruteforcer
                .produce_lines()
                .for_each(|line| println!("{}", line.escape_ascii().to_string()));
        }
        true => {
            let mut bruteforcer = Base64Bruteforcer::<u16>::default();
            bruteforcer.collect_combinations(example_string);

            if !parser.no_prune {
                log::info!(
                    "schema: {:?}\n# of permutations: {}",
                    bruteforcer.convert_to_string(),
                    bruteforcer.permutations()
                );

                log::info!("Reducing permutations to logical choices");
                bruteforcer.reduce_to_end();
            }

            if parser.info {
                println!(
                    "schema: {:?}\n# of permutations: {}",
                    bruteforcer.convert_to_string(),
                    bruteforcer.permutations()
                );
                return;
            }

            // Creating distinct lines to see results
            bruteforcer
                .produce_lines()
                .map(|line| String::from_utf16(line.as_slice()).unwrap())
                .for_each(|line| println!("{}", line.escape_default().to_string()));
        }
    }
}
