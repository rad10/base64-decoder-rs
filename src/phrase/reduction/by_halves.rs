//! Uses the binary reduction algorithm to reduce up to a reasonable size

use itertools::Itertools;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::phrase::schema::{Permutation, Phrase, Section, Variation};

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
        U: Sync + Send + Clone;

    /// A helper function to `reduce_halves`. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    fn reduce_schema_binary<U: Fn(String) -> f64>(
        permutation_limit: f64,
        sections: &[Section<String>],
        confidence_interpreter: U,
    ) -> Vec<Section<T>>
    where
        U: Sync + Send + Clone;
}

impl ReduceHalves<String> for Phrase<String> {
    fn reduce_halves<U: Fn(String) -> f64>(
        &mut self,
        permutation_limit: f64,
        confidence_interpreter: U,
    ) where
        U: Sync + Send + Clone,
    {
        self.sections = Self::reduce_schema_binary(
            permutation_limit,
            self.sections.as_slice(),
            confidence_interpreter,
        );
    }

    fn reduce_schema_binary<U: Fn(String) -> f64>(
        permutation_limit: f64,
        sections: &[Section<String>],
        confidence_interpreter: U,
    ) -> Vec<Section<String>>
    where
        U: Sync + Send + Clone,
    {
        // Leave early if section is empty or just one
        if sections.len() < 2 {
            sections.to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if sections.iter().map(|v| v.len() as f64).product::<f64>() <= permutation_limit {
            vec![
                sections
                    .iter()
                    .multi_cartesian_product()
                    .map(|v| Variation::join(v.as_slice()))
                    .map(|line| (confidence_interpreter(line.to_string()), line))
                    .inspect(|(confidence, line)| {
                        log::debug!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only half the values to make actual leeway
                    .k_largest_relaxed_by_key(
                        (sections.permutations() / 2_f64).ceil() as usize,
                        |best_var| (best_var.0 * 100_000_f64) as usize,
                    )
                    .inspect(|(confidence, line)| {
                        log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                    })
                    .map(|collapse| collapse.1)
                    .collect(),
            ]
        }
        // If permutations are still too big, split it again
        else {
            sections
                .par_chunks(sections.len() / 2)
                .flat_map(|c| {
                    Self::reduce_schema_binary(
                        permutation_limit,
                        c,
                        confidence_interpreter.to_owned(),
                    )
                })
                .collect()
        }
    }
}
