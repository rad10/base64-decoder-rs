#![forbid(unsafe_code)]
use std::io::{Error, ErrorKind};

#[cfg(not(feature = "rayon"))]
use base64_bruteforcer_rs::base64_parser::FromBase64ToAscii;
#[cfg(feature = "ollama")]
use base64_bruteforcer_rs::phrase::{
    schema::variation::Variation, validation::ollama::OllamaHandler,
};
use base64_bruteforcer_rs::phrase::{
    schema::{
        snippet::{ConvertString, Permutation, Phrase, SnippetExt},
        variation::VariationDebug,
    },
    validation::validate_with_whatlang,
};
use clap::Parser;
use futures::StreamExt;
#[cfg(feature = "rayon")]
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    str::ParallelString,
};

#[cfg(feature = "rayon")]
use base64_bruteforcer_rs::base64_parser::rayon::ParallelBruteforceBase64;

use tokio::io::{AsyncWriteExt, Stdout};
use tool_args::ToolArgs;

mod tool_args;

use crate::tool_args::{ReductionMethod, StringValidator, UtfFormat, parse_json_to_schema};

/// This is a simple helper function for the reduction by halves
fn halves_size_check<I: Permutation>(item: I) -> bool {
    item.permutations() <= 100_000_f64
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), String> {
    let parser = ToolArgs::parse();

    env_logger::builder()
        .filter_level(parser.verbose.log_level_filter())
        .init();

    // Attempt to read contents of file. Fail tool if it fails
    let b64_string = parser
        .input
        .async_read_to_string()
        .await
        .map_err(move |_| "Failed to get file content. Failing early.".to_string())?
        .trim()
        .to_owned();

    // Determine if input is a schema that has already been processed in past
    // or is a base64 string
    let mut string_permutation = if parser.use_schema {
        if let Ok(string_schema) = parse_json_to_schema(&b64_string) {
            string_schema
        } else {
            return Err(
                "Input does not match expected outputs. Please reobtain bruteforce progress in correct format.".to_string()
            );
        }
    } else {
        #[cfg(not(feature = "rayon"))]
        let b64_phrase = b64_string.into_bytes().parse_base64();
        #[cfg(feature = "rayon")]
        let b64_phrase = b64_string.into_bytes().par_parse_base64();

        match parser.format {
            UtfFormat::UTF8 => {
                #[cfg(not(feature = "rayon"))]
                {
                    b64_phrase
                        .filter_or([b'?'; 3], |piece| {
                            piece
                                .iter()
                                // Checking each character to ensure they're readable characters
                                .all(|c| {
                                    c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == b'\0'
                                })
                        })
                        .collect::<Phrase<[u8; 3]>>()
                        .into()
                }
                #[cfg(feature = "rayon")]
                tokio_rayon::spawn(move || {
                    b64_phrase
                        .filter_or([b'?'; 3], |piece| {
                            piece
                                .par_iter()
                                // Checking each character to ensure they're readable characters
                                .all(|c| {
                                    c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == b'\0'
                                })
                        })
                        .collect::<Phrase<[u8; 3]>>()
                })
                .await
                .into()
            },
            UtfFormat::UTF16LE => {
                #[cfg(not(feature = "rayon"))]
                {
                    b64_phrase
                        .convert_to_type::<u16>()
                        .filter_or([b'?'.into(); 3], |piece| {
                            // Need to check if it can convert into a string
                            String::from_utf16(piece).is_ok_and(move |s| {
                                s.chars()
                                    // Checking each character to ensure they're readable characters
                                    .all(move |c| {
                                        c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                                    })
                            })
                        })
                        .collect::<Phrase<[u16; 3]>>()
                        .into()
                }
                #[cfg(feature = "rayon")]
                tokio_rayon::spawn(move || {
                    b64_phrase
                        .convert_to_type::<u16>()
                        .filter_or([b'?'.into(); 3], |piece| {
                            // Need to check if it can convert into a string
                            String::from_utf16(piece).is_ok_and(move |s| {
                                s.par_chars()
                                    // Checking each character to ensure they're readable characters
                                    .all(move |c| {
                                        c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                                    })
                            })
                        })
                        .collect::<Phrase<[u16; 3]>>()
                })
                .await
                .into()
            },
            UtfFormat::UTF32LE => {
                #[cfg(not(feature = "rayon"))]
                {
                    b64_phrase
                        .convert_to_type::<u32>()
                        .filter_or([b'?'.into(); 3], |piece| {
                            // Check to see if this can be converted into a string
                            piece.iter().all(|u| {
                                // Fail if any of the u32 are invalid unicode
                                char::from_u32(*u).is_some_and(move |c| {
                                    c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                                })
                            })
                        })
                        .collect::<Phrase<[u32; 3]>>()
                        .into()
                }
                #[cfg(feature = "rayon")]
                tokio_rayon::spawn(move || {
                    b64_phrase
                        .convert_to_type::<u32>()
                        .filter_or([b'?'.into(); 3], |piece| {
                            // Check to see if this can be converted into a string
                            piece.par_iter().all(|u| {
                                // Fail if any of the u32 are invalid unicode
                                char::from_u32(*u).is_some_and(move |c| {
                                    c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                                })
                            })
                        })
                        .collect::<Phrase<[u32; 3]>>()
                })
                .await
                .into()
            },
        }
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
                    },
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            string_permutation.convert_to_string(),
                            string_permutation.len_sections(),
                            string_permutation.num_of_references(),
                            string_permutation.permutations()
                        );
                    },
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
                    },
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
                    },
                    (ReductionMethod::Stream, StringValidator::WhatLang) => {
                        use base64_bruteforcer_rs::phrase::reduction::by_stream::ReduceReading;

                        string_permutation.reduce_reading(
                            move |base| halves_size_check(base),
                            validate_with_whatlang,
                        )
                    },
                    #[cfg(feature = "ollama")]
                    (ReductionMethod::Pairs, StringValidator::OllamaGroup(c)) => {
                        use base64_bruteforcer_rs::phrase::{
                            reduction::by_pairs::r#async::AsyncReducePairsBulk,
                            schema::snippet::Snippet, validation::ollama::AsyncOllama,
                        };
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
                    },
                    #[cfg(feature = "ollama")]
                    (ReductionMethod::Halves, StringValidator::OllamaGroup(c)) => {
                        use base64_bruteforcer_rs::phrase::{
                            reduction::by_halves::r#async::AsyncReduceHalvesBulk,
                            validation::ollama::AsyncOllama,
                        };
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
                    },
                    #[cfg(feature = "ollama")]
                    (ReductionMethod::Stream, StringValidator::OllamaGroup(c)) => {
                        use base64_bruteforcer_rs::phrase::{
                            reduction::by_stream::r#async::AsyncReduceReadings,
                            validation::ollama::AsyncOllama,
                        };

                        let tmp_ollama_engine = ollama_engine
                            .get_or_insert(OllamaHandler::new(c.address.clone(), c.model.clone()));
                        string_permutation
                            .reduce_reading(
                                async move |snip| halves_size_check(snip),
                                async |phr| tmp_ollama_engine.validate_line(&phr).await,
                            )
                            .await
                    },
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
        return Ok(());
    }

    // Creating distinct lines to see results
    let mut stdio = futures::stream::iter(string_permutation.iter_var())
        .map(move |v| {
            if log::max_level() == log::Level::Info {
                v.debug_string()
            } else {
                Ok(v.to_string())
            }
        })
        .fold(
            Result::<Stdout, Error>::Ok(tokio::io::stdout()),
            async move |stdout, line| {
                if let Ok(mut out) = stdout {
                    if let Ok(safe_line) = line {
                        // Writing out line contents
                        out.write_all(safe_line.as_bytes()).await?;
                        // Writing new line to separate values
                        _ = out.write(b"\n").await?;
                        out.flush().await?;
                        Ok(out)
                    } else {
                        Err(std::io::Error::from(ErrorKind::Other))
                    }
                } else {
                    stdout
                }
            },
        )
        .await
        .map_err(|e| e.to_string())?;

    stdio.flush().await.map_err(|e| e.to_string())?;
    Ok(())
}
