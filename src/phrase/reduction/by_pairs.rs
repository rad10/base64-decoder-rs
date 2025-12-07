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
//! To assume all variables, we can interpret `confidence_interpreter` as
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
//! [`reduce_pairs`]: ReducePairs::reduce_pairs

use std::fmt::Debug;

use itertools::Itertools;

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
pub trait ReducePairs: SnippetExt {
    /// Takes a given schema and attempts to reduce valid choices by
    /// matching pairs. Select how many pairs will be compared at once.
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    ///
    /// Default is 2
    fn reduce_pairs(
        &self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64,
    ) -> Self;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
///
/// While this is similar in function to [`ReducePairs`], the reduction functions
/// take a validator that takes values in bulk
pub trait ReducePairsBulk<'a, U: SnippetExt<Item = Self::Item>, V: ?Sized>: SnippetExt {
    /// Takes a given schema and attempts to reduce valid choices by
    /// matching pairs. Select how many pairs will be compared at once.
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    ///
    /// Default is 2
    fn bulk_reduce_pairs(
        &'a self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: impl FnMut(U) -> V,
    ) -> Self
    where
        Variation<Self::Item>: Clone;
}

impl<T> ReducePairs for Phrase<T>
where
    T: Debug,
    Variation<T>: Clone,
{
    fn reduce_pairs(
        &self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: impl Fn(&Variation<T>) -> f64,
    ) -> Self
    where
        T: Debug,
        Variation<T>: Clone,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_pairs(number_of_pairs, move |snip: Snippet<'_, T>| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<'a, 'b, T, U, V> ReducePairsBulk<'a, U, V> for Phrase<T>
where
    T: Debug + 'b,
    U: From<&'b [Vec<Variation<T>>]> + SnippetExt<Item = Self::Item>,
    V: IntoIterator<Item = (f64, Variation<T>)>,
    'a: 'b,
{
    fn bulk_reduce_pairs(
        &'a self,
        number_of_pairs: impl Into<Option<usize>>,
        mut confidence_interpreter: impl FnMut(U) -> V,
    ) -> Self
    where
        Variation<Self::Item>: Clone,
    {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs.into() {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.len_sections() => n,
            Some(_) => self.len_sections(), // If the number is bigger than the
                                            // source itself, just use the length of the source. Its not
                                            // recommended to ever do this since its no different than checking
                                            // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        let new_sections = self
            .sections
            .chunks(pair_size)
            .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
            .flat_map(move |pairs| {
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    pairs.to_vec()
                }
                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                else if pairs.iter().all(|s| s.len() == 1) {
                    vec![vec![Variation::from_iter(pairs.iter().flatten())]]
                } else {
                    // permuting values and collecting only viable options
                    let combined: Vec<Section<T>> = vec![{
                        let snippet_permutations = pairs.permutations();
                        confidence_interpreter(pairs.into())
                            .into_iter()
                            .inspect(|(confidence, line)| {
                                log::trace!("confidence, string: {confidence}, {line:?}")
                            })
                            // Keeping only half the values to make actual leeway
                            .k_largest_relaxed_by_key(
                                (snippet_permutations / 2_f64).ceil() as usize,
                                |(confidence, _)| (confidence * 100_000_f64) as usize,
                            )
                            .inspect(|(confidence, line)| {
                                log::trace!("Accepted: confidence, string: {confidence}, {line:?}")
                            })
                            .map(move |(_, line)| line)
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
            .collect::<Vec<Section<T>>>();
        Self::from_iter(new_sections)
    }
}

/// Provides and implements the reduction trait using the [`rayon`] library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::{fmt::Debug, sync::Arc};

    use itertools::Itertools;
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReducePairs: ThreadedSnippetExt where Arc<Self::Item>: Sync {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string is
        /// closer to your objective than another.
        ///
        /// Default is 2
        fn reduce_pairs(
            &self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
        ) -> Self
        where
            Variation<Self::Item>: Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction functions
    /// take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    pub trait ParReducePairsBulk<'a, U: ThreadedSnippetExt<Item = Self::Item>, V: ?Sized>:
        ThreadedSnippetExt where Arc<Self::Item>: Sync
    {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        ///
        /// Default is 2
        fn bulk_reduce_pairs(
            &'a self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: impl Fn(U) -> V + Send + Sync,
        ) -> Self
        where
            Variation<Self::Item>: Clone + Send;
    }

    impl<T> ParReducePairs for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        fn reduce_pairs(
            &self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: impl Fn(&Variation<T>) -> f64 + Send + Sync,
        ) -> Self
        where
            T: Debug,
        {
            self.bulk_reduce_pairs(number_of_pairs, move |snip: Snippet<'_, T>| {
                snip
                    // Get all combinations of the variations
                    // Join them together to get the string to test against
                    .par_iter_var()
                    .par_bridge()
                    // Use detector to gain a confidence on each line
                    .map(|line| (confidence_interpreter(&line), line))
                    // Collecting here to drop to a regular iterator
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }
    }

    impl<'a, 'b, T, U, V> ParReducePairsBulk<'a, U, V> for Phrase<T>
    where
        T: Debug + 'b,
        Arc<T>: Sync,
        U: From<&'b [Vec<Variation<T>>]> + ThreadedSnippetExt<Item = Self::Item>,
        V: IntoIterator<Item = (f64, Variation<T>)> + Sync,
        'a: 'b,
    {
        fn bulk_reduce_pairs(
            &'a self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: impl Fn(U) -> V + Send + Sync,
        ) -> Self
        where
            Variation<T>: Clone + Send,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs.into() {
                Some(0..2) | None => 2, // Overwrite any stupid options with the
                // default
                Some(n) if n < self.len_sections() => n,
                Some(_) => self.len_sections(), // If the number is bigger than the
                                                // source itself, just use the length of the source. Its not
                                                // recommended to ever do this since its no different than checking
                                                // line by line.
            };

            // Take and operate on each pair in the schema. Will either combine a
            // pair into one section or (worst case scenario) leave the pairs as is
            let new_sections = self
                .sections
                .par_chunks(pair_size)
                .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
                .flat_map(move |pairs| {
                    // If its only 1 pair, we can skip this process
                    if pairs.len() == 1 {
                        pairs.to_vec()
                    }
                    // If there is more than one pair, but each pair only has one
                    // value, then just return a single combined form. It will give
                    // future runs more information and clarity
                    else if pairs.iter().all(|s| s.len() == 1) {
                        vec![vec![Variation::from_iter(pairs.iter().flatten())]]
                    } else {
                        // permuting values and collecting only viable options
                        let combined: Vec<Section<T>> = vec![{
                            let pair_permutation = pairs.permutations();
                            confidence_interpreter(pairs.into())
                                .into_iter()
                                .inspect(|(confidence, line)| {
                                    log::trace!("confidence, string: {confidence}, {line:?}")
                                })
                                // Keeping only half the values to make actual leeway
                                .k_largest_relaxed_by_key(
                                    (pair_permutation / 2_f64).ceil() as usize,
                                    |(confidence, _)| (confidence * 100_000_f64) as usize,
                                )
                                .inspect(|(confidence, line)| {
                                    log::trace!(
                                        "Accepted: confidence, string: {confidence}, {line:?}"
                                    )
                                })
                                .map(move |(_, line)| line)
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
                .collect::<Vec<Section<T>>>();
            Self::from_iter(new_sections)
        }
    }
}

/// Provides and implements the reduction trait using the [`futures`] library to speed up processes
#[cfg(feature = "async")]
pub mod r#async {
    use std::{fmt::Debug, sync::Arc};

    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    #[async_trait]
    pub trait AsyncReducePairs: ThreadedSnippetExt where Arc<Self::Item>: Sync {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string
        /// is closer to your objective than another.
        ///
        /// Default is 2
        async fn reduce_pairs<Fut>(
            &self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: impl for<'a> Fn(&'a Variation<Self::Item>) -> Fut + Send + Sync,
        ) -> Self
        where
            Fut: Future<Output = f64> + Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction functions
    /// take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    #[async_trait]
    pub trait AsyncReducePairsBulk<'a, U: ThreadedSnippetExt<Item = Self::Item>, V: ?Sized>:
        ThreadedSnippetExt where Arc<Self::Item>: Sync
    {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible
        /// permutations and produces an iterator of equal size with the
        /// confidence values of each string
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// Default is 2
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_reduce_pairs<Fut>(
            &'a self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: impl Fn(U) -> Fut + Send + Sync,
        ) -> Self
        where
            Variation<Self::Item>: Clone + Send,
            Fut: Future<Output = V> + Send;
    }

    #[async_trait]
    impl<T> AsyncReducePairs for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        async fn reduce_pairs<Fut>(
            &self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: impl for<'a> Fn(&'a Variation<Self::Item>) -> Fut + Send + Sync,
        ) -> Self
        where
            Fut: Future<Output = f64> + Send,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_pairs(number_of_pairs, async |snip: Snippet<'_, T>| {
                stream::iter(snip.iter_var())
                    .then(async move |line| (conf_link(&line).await, line))
                    .collect::<Vec<(f64, Variation<T>)>>()
                    .await
            })
            .await
        }
    }

    #[async_trait]
    impl<'a, 'b, T, U, V> AsyncReducePairsBulk<'a, U, V> for Phrase<T>
    where
        T: Debug + 'b,
        Arc<T>: Sync,
        U: From<&'b [Vec<Variation<T>>]> + ThreadedSnippetExt<Item = Self::Item>,
        V: IntoIterator<Item = (f64, Variation<T>)>,
        'a: 'b,
    {
        async fn bulk_reduce_pairs<Fut>(
            &'a self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: impl Fn(U) -> Fut + Send + Sync,
        ) -> Self
        where
            Variation<T>: Clone + Send,
            Fut: Future<Output = V> + Send,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs.into() {
                Some(0..2) | None => 2, // Overwrite any stupid options with the
                // default
                Some(n) if n < self.len_sections() => n,
                Some(_) => self.len_sections(), // If the number is bigger than the
                                                // source itself, just use the length of the source. Its not
                                                // recommended to ever do this since its no different than checking
                                                // line by line.
            };

            let conf_link = &confidence_interpreter;
            // Take and operate on each pair in the schema. Will either combine a
            // pair into one section or (worst case scenario) leave the pairs as is
            let new_sections = stream::iter(self.sections.chunks(pair_size))
                .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
                .then(async move |pairs| {
                    // If its only 1 pair, we can skip this process
                    if pairs.len() == 1 {
                        stream::iter(pairs.to_vec())
                    }
                    // If there is more than one pair, but each pair only has one
                    // value, then just return a single combined form. It will give
                    // future runs more information and clarity
                    else if pairs.iter().all(|s| s.len() == 1) {
                        stream::iter(vec![vec![Variation::from_iter(pairs.iter().flatten())]])
                    } else {
                        // permuting values and collecting only viable options
                        let pair_permutation = pairs.permutations();
                        let combined: Vec<Section<T>> = vec![
                            conf_link(pairs.into())
                                .await
                                .into_iter()
                                .inspect(|(confidence, line)| {
                                    log::trace!("confidence, string: {confidence}, {line:?}")
                                })
                                // Keeping only half the values to make actual leeway
                                .k_largest_relaxed_by_key(
                                    (pair_permutation / 2_f64).ceil() as usize,
                                    |(confidence, _)| (confidence * 100_000_f64) as usize,
                                )
                                .inspect(|(confidence, line)| {
                                    log::trace!(
                                        "Accepted: confidence, string: {confidence}, {line:?}"
                                    )
                                })
                                .map(move |(_, line)| line)
                                .collect(),
                        ];

                        // Go with originals if new choices aren't preferred
                        // aka if its empty or the permutations is the same as it originally was
                        if !combined[0].is_empty() && (combined[0].len() as f64) < pair_permutation
                        {
                            stream::iter(combined)
                        } else {
                            stream::iter(pairs.to_vec())
                        }
                    }
                })
                .boxed()
                .flatten()
                .collect::<Vec<Section<T>>>()
                .await;
            Self::from_iter(new_sections)
        }
    }
}
