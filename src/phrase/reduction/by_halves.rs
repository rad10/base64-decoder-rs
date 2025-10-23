//! Uses the binary reduction algorithm to reduce up to a reasonable size
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
//! F\left(n\right)=\sqrt{n}
//! ```
//!
//! The function [`reduce_halves`] can be represented with `f` and a resulting permutations `P`
//!
//! ```math
//! f\left(S,P\right)=\left\{\begin{matrix}C\left(S\right)\leP=F\left(C\left(S\right)\right)\\C\left(S\right)>P=f\left(S\left[0\cdots\frac{S}{2}\right],P\right)\timesf\left(S\left[\frac{S}{2}\cdotsS\right],P\right)\\\end{matrix}\right.
//!
//! O\left(n_b,P\right)=\left\{\array{b^n\leP=b^n\\b^n>P=2O\left(\frac{n_b}{2},P\right)}\right.=\prod_{i=1}^{m}{\left\{\array{i\geqm=b^{\frac{n}{2^{i-1}}}\\i<m=2}\right.}=2^{m-1}b^{\frac{n}{2^{i-1}}}
//!
//! P\left(n_b,P\right)=2^{m-2}b^{\frac{n}{2^{m-1}}}
//!
//! m=\left\lceil1+\log_2{\frac{n}{\log_b{P}}}\right\rceil
//! ```
//!

use std::fmt::{Debug, Display};

use itertools::Itertools;

use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<U>
where
    U: Fn(String) -> f64,
{
    /// Defines the type of item that is collected from the phrase
    type Item;

    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn reduce_halves(&mut self, permutation_limit: f64, confidence_interpreter: U);

    /// A helper function to `reduce_halves`. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    fn reduce_schema_binary(
        permutation_limit: f64,
        sections: &[Section<Self::Item>],
        confidence_interpreter: &U,
    ) -> Vec<Section<Self::Item>>;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn halves_to_end(&mut self, max_permutations: f64, confidence_interpreter: U);
}

impl<T, U> ReduceHalves<U> for Phrase<T>
where
    Variation<T>: Display + Clone,
    T: Debug,
    U: Fn(String) -> f64,
{
    type Item = T;

    fn reduce_halves(&mut self, permutation_limit: f64, confidence_interpreter: U) {
        self.sections = Self::reduce_schema_binary(
            permutation_limit,
            self.sections.as_slice(),
            &confidence_interpreter,
        );
    }

    fn reduce_schema_binary(
        permutation_limit: f64,
        sections: &[Section<Self::Item>],
        confidence_interpreter: &U,
    ) -> Vec<Section<Self::Item>> {
        // Leave early if section is empty or just one
        if sections.len() < 2 {
            sections.to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if sections.permutations() <= permutation_limit {
            vec![
                sections
                    .iter()
                    .multi_cartesian_product()
                    .map(|v| Variation::join(v.as_slice()))
                    .map(|line| (confidence_interpreter(line.to_string()), line))
                    .inspect(|(confidence, line)| {
                        log::debug!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only square root of permitted permutations to allow rerunning the reduction
                    .k_largest_relaxed_by_key(
                        sections.permutations().sqrt().floor() as usize,
                        |best_var| (best_var.0 * 100_000_f64) as usize,
                    )
                    .inspect(|(confidence, line)| {
                        log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                    })
                    .map(|collapse| collapse.1)
                    .collect::<Section<T>>(),
            ]
        }
        // If permutations are still too big, split it again
        else {
            sections
                .chunks(sections.len() / 2)
                .flat_map(|c| {
                    Self::reduce_schema_binary(permutation_limit, c, confidence_interpreter)
                })
                .collect()
        }
    }

    fn halves_to_end(&mut self, max_permutations: f64, confidence_interpreter: U) {
        // Begin by flattening single variation items
        self.flatten_sections();
        // Set the last permutation to last size. Stop if permutation doesnt
        // shrink in any given instance
        let mut last_size = usize::MAX;
        while last_size > self.sections.len() {
            last_size = self.sections.len();
            self.reduce_halves(max_permutations, &confidence_interpreter);
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

    use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReduceHalves<U: Fn(String) -> f64> {
        /// Defines the type of item that is collected from the phrase
        type Item;

        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn reduce_halves(&mut self, permutation_limit: f64, confidence_interpreter: U);

        /// A helper function to `reduce_halves`. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        fn reduce_schema_binary(
            permutation_limit: f64,
            sections: &[Section<Self::Item>],
            confidence_interpreter: &U,
        ) -> Vec<Section<Self::Item>>;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn halves_to_end(&mut self, max_permutations: f64, confidence_interpreter: U);
    }

    impl<T, U> ParReduceHalves<U> for Phrase<T>
    where
        Variation<T>: Display + Clone,
        T: Debug + Send + Sync,
        U: Fn(String) -> f64 + Send + Sync,
    {
        type Item = T;

        fn reduce_halves(&mut self, permutation_limit: f64, confidence_interpreter: U) {
            self.sections = Self::reduce_schema_binary(
                permutation_limit,
                self.sections.as_slice(),
                &confidence_interpreter,
            );
        }

        fn reduce_schema_binary(
            permutation_limit: f64,
            sections: &[Section<Self::Item>],
            confidence_interpreter: &U,
        ) -> Vec<Section<Self::Item>> {
            // Leave early if section is empty or just one
            if sections.len() < 2 {
                sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if sections.permutations() <= permutation_limit {
                vec![
                    sections
                        .iter()
                        .multi_cartesian_product()
                        // Sending to parallel so multiple splits can happen at once
                        .par_bridge()
                        .map(|v| Variation::join(v.as_slice()))
                        .map(|line| (confidence_interpreter(line.to_string()), line))
                        .inspect(|(confidence, line)| {
                            log::debug!("confidence, string: {confidence}, {line:?}")
                        })
                        // Collecting to get out of parallel iterator
                        .collect::<Vec<(f64, Variation<T>)>>()
                        .iter()
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            sections.permutations().sqrt().floor() as usize,
                            |best_var| (best_var.0 * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                        })
                        .map(|collapse| collapse.1.clone())
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                sections
                    .par_chunks(sections.len() / 2)
                    .flat_map(|c| {
                        Self::reduce_schema_binary(permutation_limit, c, confidence_interpreter)
                    })
                    .collect()
            }
        }

        fn halves_to_end(&mut self, max_permutations: f64, confidence_interpreter: U) {
            // Begin by flattening single variation items
            self.flatten_sections();
            // Set the last permutation to last size. Stop if permutation doesnt
            // shrink in any given instance
            let mut last_size = usize::MAX;
            while last_size > self.sections.len() {
                last_size = self.sections.len();
                self.reduce_halves(max_permutations, &confidence_interpreter);
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
        }
    }
}
