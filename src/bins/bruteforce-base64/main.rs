use base64_bruteforcer_rs::{
    base64_parser::{Base64Bruteforcer, BruteforcerTraits},
    phrase::{
        reduction::{by_halves::ReduceHalves, by_pairs::ReducePairs},
        schema::{ConvertString, Permutation, Phrase},
        validation::validate_with_whatlang,
    },
};
use clap::Parser;
use tool_args::ToolArgs;

mod tool_args;

fn main() {
    let parser = ToolArgs::parse();

    env_logger::builder()
        .filter_level(parser.verbose.log_level_filter())
        .init();

    // set base64 string as bytes
    let mut string_permutation: Phrase<String> = if let Some(b64_string) = parser.input.b64_string {
        match parser.use_utf16 {
            false => {
                let mut bruteforcer = Base64Bruteforcer::<u8>::default();
                bruteforcer.collect_combinations(b64_string.as_bytes());
                Phrase::from(bruteforcer).into()
            }
            true => {
                let mut bruteforcer = Base64Bruteforcer::<u16>::default();
                bruteforcer.collect_combinations(b64_string.as_bytes());
                Phrase::from(bruteforcer).into()
            }
        }
    } else if let Some(schema) = parser.input.use_schema {
        schema
    } else {
        unreachable!();
    };

    if !parser.no_prune {
        if parser.info {
            println!(
                "schema: {:?}\n# of permutations: {}",
                string_permutation.convert_to_string(),
                string_permutation.permutations(),
            );
        }

        log::info!("Reducing permutations to logical choices");
        match parser.reduction_method {
            tool_args::ReductionMethod::Pairs => {
                string_permutation.pairs_to_end(validate_with_whatlang)
            }
            tool_args::ReductionMethod::Halves => {
                string_permutation.halves_to_end(10_000_f64, validate_with_whatlang)
            }
        };
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
        .iter()
        .for_each(|line| println!("{line}"));
}
