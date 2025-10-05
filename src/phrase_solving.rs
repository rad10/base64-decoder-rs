//! this module contains functions used to either reduce a vec of variations or
//! to test human validity
#![allow(dead_code)]

use itertools::Itertools;
use rayon::{
    iter::{ParallelBridge, ParallelIterator},
    slice::ParallelSlice,
};

use crate::base64_parser::{Base64Bruteforcer, ConvertString, DisplayLines, Permutation};

/// Decides the method to determine if a pair makes human readable text
pub enum SolverMethod {
    /// Uses the whatlang NLP library
    WhatLang,
    /// Sends the string to chatgpt to determine if string is valid
    ChatGPT { api_key: String },
}

/// Determines a strings validity using the whatlang NLP library. Uses confidence
/// to determine if string passes or not
pub fn determine_accuracy_whatlang(text: &str, confidence: f64) -> bool {
    match whatlang::detect(text) {
        Some(info) if info.confidence() > confidence => true,
        Some(_) | None => false,
    }
}
pub trait SchemaReduce {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once. Default is 2
    fn reduce_schema(&mut self, number_of_pairs: Option<usize>);

    /// Runs the reduce function until the it will not reduce anymore
    fn reduce_to_end(&mut self);
}

impl SchemaReduce for Base64Bruteforcer<u8> {
    fn reduce_to_end(&mut self) {
        let mut pair_size = 2;
        while pair_size <= self.schema.len() {
            let mut last_size = usize::MAX;
            while last_size > self.schema.len() {
                last_size = self.schema.len();
                self.reduce_schema(Some(pair_size));
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.convert_to_string(),
                            self.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.schema,
                            self.permutations()
                        );
                    }
                    _ => (),
                };
            }
            pair_size += 1;
            log::debug!("Increasing pair size to {pair_size}");
        }
    }

    fn reduce_schema(&mut self, number_of_pairs: Option<usize>) {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.schema.len() => n,
            Some(_) => self.schema.len(), // If the number is bigger than the
                                          // source itself, just use the length of the source. Its not
                                          // recommended to ever do this since its no different than checking
                                          // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        self.schema = self
            .schema
            .par_chunks(pair_size)
            .map(|v| v.to_vec())
            .map(|pairs| {
                log::debug!("Visible pair: {:?}", pairs);
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    return pairs;
                }

                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                if pairs.iter().all(|v| v.len() == 1) {
                    return vec![vec![pairs.concat().concat()]];
                }

                // permuting values and collecting only viable options
                let combined: Vec<Vec<Vec<u8>>> = vec![
                    pairs
                        .clone()
                        .into_iter()
                        .multi_cartesian_product()
                        .par_bridge()
                        .map(|join| join.concat())
                        .filter(|line| {
                            let string_rep = str::from_utf8(line.as_slice()).unwrap();
                            log::debug!("detect.string: {string_rep}");
                            determine_accuracy_whatlang(string_rep, 0.10) // if confidence if over 10%, it moves on to the next round
                        })
                        .collect(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs
                } else {
                    combined
                }
            })
            .collect::<Vec<Vec<Vec<Vec<u8>>>>>()
            .concat();
    }
}

impl SchemaReduce for Base64Bruteforcer<u16> {
    fn reduce_to_end(&mut self) {
        let mut pair_size = 2;
        while pair_size <= self.schema.len() {
            let mut last_size = usize::MAX;
            while last_size > self.schema.len() {
                last_size = self.schema.len();
                self.reduce_schema(Some(pair_size));
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.convert_to_string(),
                            self.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.schema,
                            self.permutations()
                        );
                    }
                    _ => (),
                };
            }
            pair_size += 1;
            log::debug!("Increasing pair size to {pair_size}");
        }
    }

    fn reduce_schema(&mut self, number_of_pairs: Option<usize>) {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.schema.len() => n,
            Some(_) => self.schema.len(), // If the number is bigger than the
                                          // source itself, just use the length of the source. Its not
                                          // recommended to ever do this since its no different than checking
                                          // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        self.schema = self
            .schema
            .par_chunks(pair_size)
            // Need to ensure that everything is owned by the end so that the
            // new vector can replace the old
            .map(|v| v.to_vec())
            .map(|pairs| {
                log::debug!("Visible pair: {:?}", pairs);
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    return pairs;
                }

                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                if pairs.iter().all(|v| v.len() == 1) {
                    return vec![vec![pairs.concat().concat()]];
                }

                // permuting values and collecting only viable options
                let combined: Vec<Vec<Vec<u16>>> = vec![
                    pairs
                        .clone()
                        .into_iter()
                        .multi_cartesian_product()
                        .par_bridge()
                        .map(|join| join.concat())
                        .filter(|line| {
                            let string_rep = String::from_utf16_lossy(line.as_slice());
                            log::debug!("detect.string: {string_rep}");
                            determine_accuracy_whatlang(string_rep.as_str(), 0.10) // if confidence if over 10%, it moves on to the next round
                        })
                        .collect(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs
                } else {
                    combined
                }
            })
            .collect::<Vec<Vec<Vec<Vec<u16>>>>>()
            .concat();
    }
}

/// This struct converts resulting bytes from whatever encoding into strings.
/// This is used for ease of implementation over performance/memory management.
#[derive(Default)]
pub struct StringBruteforcer {
    /// Contains the permutation schema generated from collecting combinations
    /// from the bruteforcer
    pub schema: Vec<Vec<String>>,
}

impl StringBruteforcer {
    pub fn new(value: Vec<Vec<String>>) -> Self {
        Self { schema: value }
    }
}

impl Permutation for StringBruteforcer {
    fn permutations(&self) -> f64 {
        self.schema
            .iter()
            .map(|section| section.len())
            .map(|size| size as f64)
            .product()
    }
}

impl ConvertString for StringBruteforcer {
    /// Returns a copy of the internal schema since nothing needs to be converted
    fn convert_to_string(&self) -> Vec<Vec<String>> {
        self.schema.clone()
    }
}

impl DisplayLines<String> for StringBruteforcer {
    /// Turns the schema into an iterator of every possible combination
    fn produce_lines(&self) -> impl Iterator<Item = String> {
        self.schema
            .iter()
            .multi_cartesian_product()
            .map(|s| s.into_iter().join(""))
    }
}

impl<T> From<Base64Bruteforcer<T>> for StringBruteforcer
where
    Base64Bruteforcer<T>: ConvertString,
{
    fn from(value: Base64Bruteforcer<T>) -> Self {
        Self {
            schema: value.convert_to_string(),
        }
    }
}

impl SchemaReduce for StringBruteforcer {
    fn reduce_to_end(&mut self) {
        let mut pair_size = 2;
        while pair_size <= self.schema.len() {
            let mut last_size = usize::MAX;
            while last_size > self.schema.len() {
                last_size = self.schema.len();
                self.reduce_schema(Some(pair_size));
                log::info!(
                    "Schema: {:?}\n# of permutations: {:e}",
                    self.schema,
                    self.permutations()
                );
            }
            pair_size += 1;
            log::debug!("Increasing pair size to {pair_size}");
        }
    }

    fn reduce_schema(&mut self, number_of_pairs: Option<usize>) {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.schema.len() => n,
            Some(_) => self.schema.len(), // If the number is bigger than the
                                          // source itself, just use the length of the source. Its not
                                          // recommended to ever do this since its no different than checking
                                          // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        self.schema = self
            .schema
            .par_chunks(pair_size)
            .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
            .map(|pairs| {
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    return pairs.to_vec();
                }

                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                if pairs.iter().all(|v| v.len() == 1) {
                    return vec![vec![pairs.iter().flatten().join("")]];
                }

                // permuting values and collecting only viable options
                let combined: Vec<Vec<String>> = vec![
                    pairs
                        .iter()
                        .multi_cartesian_product()
                        .par_bridge()
                        .map(|join| join.into_iter().join(""))
                        .inspect(|line| log::debug!("detect.string: {line}"))
                        // if confidence if over 10%, it moves on to the next round
                        .filter(|line| determine_accuracy_whatlang(line.as_str(), 0.10))
                        .collect(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs.to_vec()
                } else {
                    combined
                }
            })
            .collect::<Vec<Vec<Vec<String>>>>()
            .concat();
    }
}
