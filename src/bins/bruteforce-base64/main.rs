use base64_bruteforcer_rs::{
    base64_parser::{Base64Bruteforcer, BruteforcerTraits},
    phrase::{
        schema::{ConvertString, Permutation, Phrase},
        validation::validate_with_whatlang,
    },
};
use clap::Parser;
use tool_args::ToolArgs;

mod tool_args;

#[cfg(not(feature = "rayon"))]
use base64_bruteforcer_rs::phrase::reduction::{by_halves::ReduceHalves, by_pairs::ReducePairs};
#[cfg(feature = "rayon")]
use base64_bruteforcer_rs::phrase::reduction::{
    by_halves::rayon::ParReduceHalves, by_pairs::rayon::ParReducePairs,
};

use crate::tool_args::{ReductionMethod, StringValidator, parse_json_to_schema};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> () {
    let parser = ToolArgs::parse();

    env_logger::builder()
        .filter_level(parser.verbose.log_level_filter())
        .init();

    // Attempt to read contents of file. Fail tool if it fails
    let b64_string = if let Ok(input_content) = parser.input.async_read_to_string().await {
        input_content
    } else {
        log::error!("Failed to get file content. Failing early.");
        return;
    };

    // Determine if input is a schema that has already been processed in past
    // or is a base64 string
    let mut string_permutation = if parser.use_schema {
        if let Ok(string_schema) = parse_json_to_schema(&b64_string) {
            string_schema
        } else {
            log::error!(
                "Input does not match expected outputs. Please reobtain bruteforce progress in correct format."
            );
            return;
        }
    } else if parser.use_utf16 {
        let mut bruteforcer = Base64Bruteforcer::<u16>::default();
        bruteforcer.collect_combinations(b64_string.as_bytes());
        Phrase::from(bruteforcer).into()
    } else {
        let mut bruteforcer = Base64Bruteforcer::<u8>::default();
        bruteforcer.collect_combinations(b64_string.as_bytes());
        Phrase::from(bruteforcer).into()
    };

    if parser.validation_method != StringValidator::None {
        if parser.info {
            println!(
                "schema: {:?}\n# of permutations: {}",
                string_permutation.convert_to_string(),
                string_permutation.permutations(),
            );
        }

        log::info!("Reducing permutations to logical choices");
        string_permutation = match (parser.reduction_method, parser.validation_method) {
            (_, StringValidator::None) => unreachable!(),
            (ReductionMethod::Pairs, StringValidator::WhatLang) => {
                #[cfg(not(feature = "rayon"))]
                {
                    string_permutation.pairs_to_end(validate_with_whatlang)
                }
                #[cfg(feature = "rayon")]
                tokio_rayon::spawn(move || string_permutation.pairs_to_end(validate_with_whatlang))
                    .await
            }
            (ReductionMethod::Halves, StringValidator::WhatLang) => {
                #[cfg(not(feature = "rayon"))]
                {
                    string_permutation.halves_to_end(
                        |snip| snip.permutations() <= 100_000_f64,
                        validate_with_whatlang,
                    )
                }
                #[cfg(feature = "rayon")]
                tokio_rayon::spawn(move || {
                    string_permutation.halves_to_end(
                        |snip| snip.permutations() <= 100_000_f64,
                        validate_with_whatlang,
                    )
                })
                .await
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
