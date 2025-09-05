//! this module contains functions used to either reduce a vec of variations or
//! to test human validity

use std::usize;

use itertools::Itertools;

use crate::base64_parser::Base64Bruteforcer;

/// Decides the method to determine if a pair makes human readable text
pub enum SolverMethod {
    /// Uses the whatlang NLP library
    WhatLang,
    /// Sends the string to chatgpt to determine if string is valid
    ChatGPT {
        api_key: String,
    },
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
            .iter()
            .chunks(pair_size)
            .into_iter()
            // Need to ensure that everything is owned by the end so that the
            // new vector can replace the old
            .map(|pair| pair.map(|v| v.to_owned()).collect_vec())
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
                let combined = vec![
                    pairs
                        .clone()
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|join| join.concat())
                        .filter(|line| {
                            let string_rep = str::from_utf8(line.as_slice()).unwrap();
                            log::debug!("detect.string: {string_rep}");
                            determine_accuracy_whatlang(string_rep, 0.10) // if confidence if over 10%, it moves on to the next round
                        })
                        .collect_vec(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs
                } else {
                    combined
                }
            })
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
            .iter()
            .chunks(pair_size)
            .into_iter()
            // Need to ensure that everything is owned by the end so that the
            // new vector can replace the old
            .map(|pair| pair.map(|v| v.to_owned()).collect_vec())
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
                let combined = vec![
                    pairs
                        .clone()
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|join| join.concat())
                        .filter(|line| {
                            let string_rep = String::from_utf16_lossy(line.as_slice());
                            log::debug!("detect.string: {string_rep}");
                            determine_accuracy_whatlang(string_rep.as_str(), 0.10) // if confidence if over 10%, it moves on to the next round
                        })
                        .collect_vec(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs
                } else {
                    combined
                }
            })
            .concat();
    }
}
