//! This module implements the reduce by pairs algorithm. This is meant to reduce
//! permutations by checking snippets by pairs rather than checking the entire
//! phrase at once.
//!
//! The math behind this:
//!
//! Interpret the Cartesian Product as the function
//!
//! ```math
//! C\left(n_b\right)=\prod_{i=1}^{n}b=b^n
//! ```
//!
//! To assume all variables, we can interpret [`confidence_interpreter`] as
//!
//! ```math
//! F(n)=n/2
//! F\left(n\right)=\frac{n}{2}
//! ```
//!
//! The function [`reduce_pairs`] can be represented with `f` and a resulting permutations `P`
//!
//! ```math
//! f(n_b,m)=\prod_{i=1}^{\sfrac{n}{m}}{F\left(C\left(\left[n_{\left(i-1\right)m}\cdotsn_{im}\right]\right)\right)}
//!
//! \Theta\left(n_b,m\right)=\frac{n}{m}b^m
//!
//! P\left(n_b,m\right)=2^{\sfrac{-n}{m}}b^n
//! ```
//!

use std::fmt::{Debug, Display};

use itertools::Itertools;

use crate::phrase::schema::{
    ConvertString, Permutation, Phrase, Section, Snippet, Variation, VariationValue,
};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
pub trait ReducePairs<U> {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once.
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    ///
    /// Default is 2
    fn reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U);

    /// Runs the reduce function until the it will not reduce anymore
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    fn pairs_to_end(&mut self, confidence_interpreter: U);
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
///
/// While this is similar in function to [`ReducePairs`], the reduction functions
/// take a validator that takes values in bulk
pub trait ReducePairsBulk<U, V> {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once.
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    ///
    /// Default is 2
    fn bulk_reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U);

    /// Runs the reduce function until the it will not reduce anymore
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    fn bulk_pairs_to_end(&mut self, confidence_interpreter: U);
}

impl<T, U> ReducePairs<U> for Phrase<T>
where
    T: Debug,
    U: Fn(String) -> f64,
    Variation<T>: Clone + Display + VariationValue,
{
    fn reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U) {
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
            .chunks(pair_size)
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

    fn pairs_to_end(&mut self, confidence_interpreter: U) {
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

impl<T, U, V> ReducePairsBulk<U, V> for Phrase<T>
where
    T: Clone + Debug,
    U: Fn(Snippet<'_, T>) -> V,
    V: Iterator<Item = f64>,
    Variation<T>: Display + VariationValue,
{
    fn bulk_reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U)
    where
        T: Clone + Debug,
        Variation<T>: Display + VariationValue,
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
            .chunks(pair_size)
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
                    let combined: Vec<Section<T>> = vec![{
                        let combos_snippet = Snippet::new(pairs);

                        let inspection_set = confidence_interpreter(combos_snippet.clone());

                        inspection_set
                            .zip(combos_snippet.iter_var())
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            // Keeping only half the values to make actual leeway
                            .k_largest_relaxed_by_key(
                                (combos_snippet.permutations() / 2_f64).ceil() as usize,
                                |(confidence, _)| (confidence * 100_000_f64) as usize,
                            )
                            .inspect(|(confidence, line)| {
                                log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                            })
                            .map(|(_, line)| line)
                            .collect()
                    }];

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

    fn bulk_pairs_to_end(&mut self, confidence_interpreter: U) {
        // Begin by flattening single variation items
        self.flatten_sections();
        let mut pair_size = 2;
        while pair_size <= self.sections.len() {
            let mut last_size = usize::MAX;
            while last_size > self.sections.len() {
                last_size = self.sections.len();
                self.bulk_reduce_pairs(Some(pair_size), &confidence_interpreter);
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

/// Provides and implements the reduction trait using the rayon library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::fmt::{Debug, Display};

    use itertools::Itertools;
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReducePairs<U> {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string is
        /// closer to your objective than another.
        ///
        /// Default is 2
        fn reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U);

        /// Runs the reduce function until the it will not reduce anymore
        fn pairs_to_end(&mut self, confidence_interpreter: U);
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction functions
    /// take a validator that takes values in bulk
    pub trait ParReducePairsBulk<U, V> {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        ///
        /// Default is 2
        fn bulk_reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U);

        /// Runs the reduce function until the it will not reduce anymore
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        fn bulk_pairs_to_end(&mut self, confidence_interpreter: U);
    }

    impl<T, U> ParReducePairs<U> for Phrase<T>
    where
        T: Debug + Send + Sync,
        U: Fn(String) -> f64 + Sync,
        Variation<T>: Clone + Display,
    {
        fn reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U) {
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
            self.sections =
                self.sections
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
                            let combined: Vec<Section<T>> =
                                vec![
                        pairs
                            .iter()
                            // Get all combinations of the variations
                            .multi_cartesian_product()
                            .par_bridge()
                            // Join them together to get the string to test against
                            .map(|v| Variation::join(v.as_slice()))
                            // Use detector to gain a confidence on each line
                            .map(|line| (confidence_interpreter(line.to_string()), line))
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            // Collecting here to drop to a regular iterator
                            .collect::<Vec<(f64, Variation<T>)>>()
                            .iter()
                            // Keeping only half the values to make actual leeway
                            .k_largest_relaxed_by_key(
                                (pairs.permutations() / 2_f64).ceil() as usize,
                                |best_var| (best_var.0 * 100_000_f64) as usize,
                            )
                            .inspect(|(confidence, line)| {
                                log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                            })
                            .map(|collapse| collapse.1.clone())
                            .collect(),
                    ];

                            // Go with originals if new choices aren't preferred
                            // aka if its empty or the permutations is the same as it originally was
                            if !combined[0].is_empty()
                                && (combined[0].len() as f64) < pairs.permutations()
                            {
                                combined
                            } else {
                                pairs.to_vec()
                            }
                        }
                    })
                    .collect::<Vec<Section<T>>>()
        }

        fn pairs_to_end(&mut self, confidence_interpreter: U) {
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

    impl<T, U, V> ParReducePairsBulk<U, V> for Phrase<T>
    where
        T: Clone + Debug + Send + Sync,
        U: Fn(Snippet<'_, T>) -> V + Sync,
        V: Iterator<Item = f64>,
        Variation<T>: Display,
    {
        fn bulk_reduce_pairs(&mut self, number_of_pairs: Option<usize>, confidence_interpreter: U)
        where
            T: Clone + Send + Sync,
            Variation<T>: Display,
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
                        let combined: Vec<Section<T>> = vec![{
                            let combos_snippet = Snippet::new(pairs);

                            let inspection_set = confidence_interpreter(combos_snippet.clone());

                            inspection_set
                                .zip(combos_snippet.iter_var())
                                .inspect(|(confidence, line)| {
                                    log::debug!("confidence, string: {confidence}, {line:?}")
                                })
                                // Keeping only half the values to make actual leeway
                                .k_largest_relaxed_by_key(
                                    (combos_snippet.permutations() / 2_f64).ceil() as usize,
                                    |(confidence, _)| (confidence * 100_000_f64) as usize,
                                )
                                .inspect(|(confidence, line)| {
                                    log::debug!(
                                        "Accepted: confidence, string: {confidence}, {line:?}"
                                    )
                                })
                                .map(|(_, line)| line)
                                .collect()
                        }];

                        // Go with originals if new choices aren't preferred
                        // aka if its empty or the permutations is the same as it originally was
                        if !combined[0].is_empty()
                            && (combined[0].len() as f64) < pairs.permutations()
                        {
                            combined
                        } else {
                            pairs.to_vec()
                        }
                    }
                })
                .collect::<Vec<Section<T>>>()
        }

        fn bulk_pairs_to_end(&mut self, confidence_interpreter: U)
        where
            Variation<T>: Display,
        {
            // Begin by flattening single variation items
            self.flatten_sections();
            let mut pair_size = 2;
            while pair_size <= self.sections.len() {
                let mut last_size = usize::MAX;
                while last_size > self.sections.len() {
                    last_size = self.sections.len();
                    self.bulk_reduce_pairs(Some(pair_size), &confidence_interpreter);
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
}
