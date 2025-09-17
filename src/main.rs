use base64_bruteforcer_rs::{
    base64_parser::{
        Base64Bruteforcer, BruteforcerTraits, ConvertString, DisplayLines, Permutation,
    },
    phrase_reduction::Phrase,
    phrase_solving::SchemaReduce,
};
use clap::Parser;
use tool_args::ToolArgs;

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

    let string_permutation: Phrase<String> = match parser.use_utf16 {
        false => {
            let mut bruteforcer = Base64Bruteforcer::<u8>::default();
            bruteforcer.collect_combinations(example_string);
            Phrase::from(bruteforcer)
        }
        true => {
            let mut bruteforcer = Base64Bruteforcer::<u16>::default();
            bruteforcer.collect_combinations(example_string);
            Phrase::from(bruteforcer)
        }
    };

    let mut string_permutation = string_permutation;

    if !parser.no_prune {
        log::info!(
            "schema: {:?}\n# of permutations: {}",
            string_permutation.convert_to_string(),
            string_permutation.permutations(),
        );

        log::info!("Reducing permutations to logical choices");
        string_permutation.reduce_to_end();
    }

    if parser.info {
        println!(
            "schema: {:?}\n# of permutations: {}",
            string_permutation.convert_to_string(),
            string_permutation.permutations(),
        );
        return;
    }

    // Creating distinct lines to see results
    string_permutation
        .produce_lines()
        .for_each(|line| println!("{line}"));
}
