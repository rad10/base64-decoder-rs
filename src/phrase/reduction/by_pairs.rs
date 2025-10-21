//! This module implements the reduce by pairs algorithm. This is meant to reduce
//! permutations by checking snippets by pairs rather than checking the entire
//! phrase at once

use std::fmt::{Debug, Display};

use itertools::Itertools;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Variation};

pub trait ReducePairs {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once.
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    ///
    /// Default is 2
    fn reduce_pairs<U: Fn(String) -> f64>(
        &mut self,
        number_of_pairs: Option<usize>,
        confidence_interpreter: U,
    ) where
        U: Sync + Send;

    /// Runs the reduce function until the it will not reduce anymore
    fn pairs_to_end<U: Fn(String) -> f64>(&mut self, confidence_interpreter: U)
    where
        U: Sync + Send;
}

impl<T> ReducePairs for Phrase<T>
where
    Section<T>: Send + Sync,
    T: Debug,
    Variation<T>: Clone + Display,
    Vec<Variation<String>>: FromIterator<Variation<T>>,
{
    fn reduce_pairs<U: Fn(String) -> f64>(
        &mut self,
        number_of_pairs: Option<usize>,
        confidence_interpreter: U,
    ) where
        U: Sync + Send,
    {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.sections.len() => n,
            Some(_) => self.sections.len(), // If the number is bigger than the
                                            // source itself, just use the length of the source. Its not
                                            // recommended to ever do this since its no different than checking
                                            // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        self.sections = self
            .sections
            .par_chunks(pair_size)
            // .chunks(pair_size)
            .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
            .flat_map(|pairs| {
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    pairs.to_vec()
                }
                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                else if pairs.iter().all(|v| v.len() == 1) {
                    vec![vec![Variation::join(
                        pairs
                            .iter()
                            .map(|s| &s[0])
                            .collect::<Vec<&Variation<T>>>()
                            .as_slice(),
                    )]]
                } else {
                    // permuting values and collecting only viable options
                    let combined: Vec<Section<T>> = vec![
                        pairs
                            .iter()
                            // Get all combinations of the variations
                            .multi_cartesian_product()
                            // Join them together to get the string to test against
                            .map(|v| Variation::join(v.as_slice()))
                            // Use detector to gain a confidence on each line
                            .map(|line| (confidence_interpreter(line.to_string()), line))
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            // Keeping only half the values to make actual leeway
                            .k_largest_relaxed_by_key(
                                (pairs.permutations() / 2_f64).ceil() as usize,
                                |best_var| (best_var.0 * 100_000_f64) as usize,
                            )
                            .inspect(|(confidence, line)| {
                                log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                            })
                            .map(|collapse| collapse.1)
                            .collect(),
                    ];

                    // Go with originals if new choices aren't preferred
                    // aka if its empty or the permutations is the same as it originally was
                    if !combined[0].is_empty() && (combined[0].len() as f64) < pairs.permutations()
                    {
                        combined
                    } else {
                        pairs.to_vec()
                    }
                }
            })
            .collect::<Vec<Section<T>>>()
    }

    fn pairs_to_end<U: Fn(String) -> f64>(&mut self, confidence_interpreter: U)
    where
        U: Sync + Send,
    {
        // Begin by flattening single variation items
        self.flatten_sections();
        let mut pair_size = 2;
        while pair_size <= self.sections.len() {
            let mut last_size = usize::MAX;
            while last_size > self.sections.len() {
                last_size = self.sections.len();
                self.reduce_pairs(Some(pair_size), &confidence_interpreter);
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
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            self.sections,
                            self.sections.len(),
                            self.sections
                                .iter()
                                .flat_map(|s| s.iter().map(|v| v.links.len()))
                                .sum::<usize>(),
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
}
