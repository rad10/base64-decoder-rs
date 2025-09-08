use base64_bruteforcer_rs::{
    base64_parser::{ConvertString, Permutation},
    phrase_reduction::{Phrase, Variation},
    phrase_solving::SchemaReduce,
};
use clap::Parser;
use tool_args::ToolArgs;

use crate::{
    base64_parser::{Base64Bruteforcer, BruteforcerTraits, DisplayLines},
    phrase_solving::StringBruteforcer,
};

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

    let string_permutation = match parser.use_utf16 {
        false => {
            let mut bruteforcer = Base64Bruteforcer::<u8>::default();
            bruteforcer.collect_combinations(example_string);
            StringBruteforcer::from(bruteforcer)
        }
        true => {
            let mut bruteforcer = Base64Bruteforcer::<u16>::default();
            bruteforcer.collect_combinations(example_string);
            StringBruteforcer::from(bruteforcer)
        }
    };

    let mut phrase: Phrase<String> = Phrase::new(
        string_permutation
            .schema
            .iter()
            .map(|section| {
                section
                    .iter()
                    .map(|variation| Variation::new(variation.to_owned()))
                    .collect()
            })
            .collect(),
    );

    if !parser.no_prune {
        log::info!(
            "schema: {:?}\n# of permutations: {}",
            phrase.convert_to_string(),
            phrase.permutations(),
        );

        log::info!("Reducing permutations to logical choices");
        phrase.reduce_to_end();
    }

    if parser.info {
        println!(
            "schema: {:?}\n# of permutations: {}",
            phrase.convert_to_string(),
            phrase.permutations(),
        );
        return;
    }

    // Creating distinct lines to see results
    string_permutation
        .produce_lines()
        .for_each(|line| println!("{line}"));
}
