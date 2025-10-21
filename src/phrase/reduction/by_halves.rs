//! Uses the binary reduction algorithm to reduce up to a reasonable size

use std::fmt::{Debug, Display};

use itertools::Itertools;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Variation};

pub trait ReduceHalves<T> {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn reduce_halves<U: Fn(String) -> f64>(
        &mut self,
        permutation_limit: f64,
        confidence_interpreter: U,
    ) where
        U: Sync + Send;

    /// A helper function to `reduce_halves`. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    fn reduce_schema_binary<U: Fn(String) -> f64>(
        permutation_limit: f64,
        sections: &[Section<T>],
        confidence_interpreter: &U,
    ) -> Vec<Section<T>>
    where
        U: Sync + Send;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn halves_to_end<U: Fn(String) -> f64>(
        &mut self,
        max_permutations: f64,
        confidence_interpreter: U,
    ) where
        U: Sync + Send;
}

impl<T> ReduceHalves<T> for Phrase<T>
where
    Section<T>: Send + Sync,
    Variation<T>: Display + Clone,
    T: Debug,
{
    fn reduce_halves<U: Fn(String) -> f64>(
        &mut self,
        permutation_limit: f64,
        confidence_interpreter: U,
    ) where
        U: Sync + Send,
    {
        self.sections = Self::reduce_schema_binary(
            permutation_limit,
            self.sections.as_slice(),
            &confidence_interpreter,
        );
    }

    fn reduce_schema_binary<U: Fn(String) -> f64>(
        permutation_limit: f64,
        sections: &[Section<T>],
        confidence_interpreter: &U,
    ) -> Vec<Section<T>>
    where
        U: Sync + Send,
    {
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
                .par_chunks(sections.len() / 2)
                .flat_map(|c| {
                    Self::reduce_schema_binary(permutation_limit, c, confidence_interpreter)
                })
                .collect()
        }
    }

    fn halves_to_end<U: Fn(String) -> f64>(
        &mut self,
        max_permutations: f64,
        confidence_interpreter: U,
    ) where
        U: Sync + Send,
    {
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
