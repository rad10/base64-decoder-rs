#[cfg(not(feature = "rayon"))]
use base64_bruteforcer_rs::base64_parser::FromBase64;
#[cfg(feature = "ollama")]
use base64_bruteforcer_rs::phrase::schema::Variation;
#[cfg(feature = "ollama")]
use base64_bruteforcer_rs::phrase::validation::ollama::OllamaHandler;
use base64_bruteforcer_rs::phrase::{
    schema::{ConvertString, Permutation, Phrase, Snippet, SnippetExt},
    validation::validate_with_whatlang,
};
#[cfg(feature = "rayon")]
use base64_bruteforcer_rs::{
    base64_parser::rayon::FromParBase64, phrase::schema::ThreadedSnippetExt,
};
use clap::Parser;
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use tool_args::ToolArgs;

mod tool_args;

use crate::tool_args::{ReductionMethod, StringValidator, parse_json_to_schema};

/// This is a simple helper function for the reduction by halves
fn halves_size_check<I: Permutation>(item: I) -> bool {
    item.permutations() <= 100_000_f64
}

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
        #[cfg(not(feature = "rayon"))]
        {
            Phrase::<Vec<u16>>::parse_base64(b64_string.into_bytes(), None).into()
        }
        #[cfg(feature = "rayon")]
        tokio_rayon::spawn(move || {
            Phrase::<Vec<u16>>::par_parse_base64(b64_string.into_bytes().into_par_iter(), None)
        })
        .await
        .into()
    } else {
        #[cfg(not(feature = "rayon"))]
        {
            Phrase::<Vec<u8>>::parse_base64(b64_string.into_bytes(), None).into()
        }
        #[cfg(feature = "rayon")]
        tokio_rayon::spawn(move || {
            Phrase::<Vec<u8>>::par_parse_base64(b64_string.into_bytes().into_par_iter(), None)
        })
        .await
        .into()
    };

    if parser.validation_method != StringValidator::None {
        if parser.info {
            println!(
                "schema: {:?}\n# of permutations: {:e}",
                string_permutation.convert_to_string(),
                string_permutation.permutations(),
            );
        }

        log::info!("Flattening single choice variations");
        string_permutation = string_permutation.flatten_sections();

        if parser.info {
            println!(
                "schema: {:?}\n# of permutations: {:e}",
                string_permutation.convert_to_string(),
                string_permutation.permutations(),
            );
        }

        // Preparing ollama engine
        #[cfg(feature = "ollama")]
        let mut ollama_engine = if let StringValidator::OllamaGroup(c) = &parser.validation_method {
            Some(OllamaHandler::new(c.address.clone(), c.model.clone()))
        } else {
            None
        };

        // Setting a temp flag if pairs is selected
        log::info!("Reducing permutations to logical choices");
        let mut pair_size = 2;
        'pair_loop: loop {
            let mut last_size = usize::MAX;
            while last_size > string_permutation.len_sections() {
                last_size = string_permutation.len_sections();
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            string_permutation.convert_to_string(),
                            string_permutation.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            string_permutation.convert_to_string(),
                            string_permutation.len_sections(),
                            string_permutation.num_of_references(),
                            string_permutation.permutations()
                        );
                    }
                    _ => (),
                };

                // Update permutation depending on which loop
                string_permutation = match (parser.reduction_method, &parser.validation_method) {
                    (_, StringValidator::None) => unreachable!(),
                    (ReductionMethod::Pairs, StringValidator::WhatLang) => {
                        #[cfg(not(feature = "rayon"))]
                        {
                            use base64_bruteforcer_rs::phrase::reduction::by_pairs::ReducePairs;
                            string_permutation.reduce_pairs(pair_size, validate_with_whatlang)
                        }
                        #[cfg(feature = "rayon")]
                        {
                            use base64_bruteforcer_rs::phrase::reduction::by_pairs::rayon::ParReducePairs;

                            tokio_rayon::spawn(move || {
                                string_permutation.reduce_pairs(pair_size, validate_with_whatlang)
                            })
                            .await
                        }
                    }
                    (ReductionMethod::Halves, StringValidator::WhatLang) => {
                        #[cfg(not(feature = "rayon"))]
                        {
                            use base64_bruteforcer_rs::phrase::reduction::by_halves::ReduceHalves;

                            string_permutation
                                .reduce_halves(halves_size_check, validate_with_whatlang)
                        }
                        #[cfg(feature = "rayon")]
                        {
                            use base64_bruteforcer_rs::phrase::reduction::by_halves::rayon::ParReduceHalves;

                            tokio_rayon::spawn(move || {
                                string_permutation.reduce_halves(
                                    |snip| halves_size_check(snip),
                                    validate_with_whatlang,
                                )
                            })
                            .await
                        }
                    }
                    #[cfg(feature = "ollama")]
                    (ReductionMethod::Pairs, StringValidator::OllamaGroup(c)) => {
                        use base64_bruteforcer_rs::phrase::reduction::by_pairs::r#async::AsyncReducePairsBulk;
                        use base64_bruteforcer_rs::phrase::validation::ollama::AsyncOllama;
                        use futures::stream::StreamExt;

                        let tmp_ollama_engine = ollama_engine
                            .get_or_insert(OllamaHandler::new(c.address.clone(), c.model.clone()));

                        string_permutation
                            .bulk_reduce_pairs(pair_size, async |phr: Snippet<'_, String>| {
                                tmp_ollama_engine
                                    .validate_group(phr)
                                    .await
                                    .collect::<Vec<(f64, Variation<String>)>>()
                                    .await
                            })
                            .await
                    }
                    #[cfg(feature = "ollama")]
                    (ReductionMethod::Halves, StringValidator::OllamaGroup(c)) => {
                        use base64_bruteforcer_rs::phrase::reduction::by_halves::r#async::AsyncReduceHalvesBulk;
                        use base64_bruteforcer_rs::phrase::validation::ollama::AsyncOllama;
                        use futures::stream::StreamExt;

                        let tmp_ollama_engine = ollama_engine
                            .get_or_insert(OllamaHandler::new(c.address.clone(), c.model.clone()));
                        string_permutation
                            .bulk_reduce_halves(
                                async move |snip| halves_size_check(snip),
                                async |phr| {
                                    tmp_ollama_engine
                                        .validate_group(phr)
                                        .await
                                        .collect::<Vec<(f64, Variation<String>)>>()
                                        .await
                                },
                            )
                            .await
                    }
                };
            }
            pair_size += 1;

            // Leaving early if not pairs or pair is now bigger than size
            if parser.reduction_method != ReductionMethod::Pairs
                || pair_size > string_permutation.len_sections()
            {
                break 'pair_loop;
            }
            log::debug!("Increasing pair size to {pair_size}");
        }
    }

    if parser.info {
        println!(
            "schema: {:?}\n# of permutations: {:e}",
            string_permutation.convert_to_string(),
            string_permutation.permutations(),
        );
        return;
    }

    // Creating distinct lines to see results
    #[cfg(not(feature = "rayon"))]
    string_permutation
        .into_iter_str()
        .for_each(|line| println!("{line}"));
    #[cfg(feature = "rayon")]
    string_permutation
        .par_into_iter_str()
        .par_bridge()
        .for_each(move |line| println!("{line}"));
}
