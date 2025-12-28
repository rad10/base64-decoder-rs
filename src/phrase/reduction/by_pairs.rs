//! This module implements the reduce by pairs algorithm. This is meant to
//! reduce permutations by checking snippets by pairs rather than checking the
//! entire phrase at once.
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
//! The function [`reduce_pairs`] can be represented with `f` and a resulting
//! permutations `P`
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

use crate::phrase::schema::{
    snippet::{BorrowedSnippet, Permutation, Phrase, Section, Snippet, SnippetExt},
    variation::Variation,
};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
pub trait ReducePairs<S: SnippetExt<Item = Self::Item>>: SnippetExt {
    /// Takes a given schema and attempts to reduce valid choices by
    /// matching pairs. Select how many pairs will be compared at once.
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    ///
    /// Default is 2
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::{
    ///     reduction::by_pairs::ReducePairs,
    ///     schema::{
    ///         snippet::{BorrowedSnippet, Phrase, SnippetExt},
    ///         variation::Variation,
    ///     },
    ///     validation::validate_with_whatlang,
    /// };
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec![vec!["Hel"], vec!["HeR"]],
    ///     vec![vec!["lo "]],
    ///     vec![vec!["Wor"], vec!["WoX"]],
    ///     vec![vec!["ld!"]],
    ///     vec![vec!["Thi"], vec!["ThR"]],
    ///     vec![vec!["is "]],
    ///     vec![vec!["is "]],
    ///     vec![vec!["my "]],
    ///     vec![vec!["str"], vec!["stX"]],
    ///     vec![vec!["ing"], vec!["Mng"]],
    ///     vec![vec!["!"]],
    /// ]
    /// .into_iter()
    /// .map(|section| {
    ///     section.into_iter().map(|variation| {
    ///         variation
    ///             .into_iter()
    ///             .map(ToOwned::to_owned)
    ///             .map(Arc::new)
    ///             .collect::<Variation<String>>()
    ///     })
    /// })
    /// .collect();
    ///
    /// let reduced_phrase: Phrase<String> = [
    ///     vec![vec!["Hel", "lo "]],
    ///     vec![vec!["WoX", "ld!"]],
    ///     vec![vec!["Thi", "is "]],
    ///     vec![vec!["is ", "my "]],
    ///     vec![vec!["str", "ing"], vec!["stX", "Mng"]],
    ///     vec![vec!["!"]],
    /// ]
    /// .into_iter()
    /// .map(|section| {
    ///     section.into_iter().map(|variation| {
    ///         variation
    ///             .into_iter()
    ///             .map(ToOwned::to_owned)
    ///             .map(Arc::new)
    ///             .collect::<Variation<String>>()
    ///     })
    /// })
    /// .collect();
    ///
    /// let reduced_result = phrase_string.reduce_pairs(2, validate_with_whatlang);
    /// assert!(reduced_result == reduced_phrase);
    /// ```
    fn reduce_pairs<C>(
        &self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: C,
    ) -> Self
    where
        C: Fn(&Variation<Self::Item>) -> f64;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
///
/// While this is similar in function to [`ReducePairs`], the reduction
/// functions take a validator that takes values in bulk
pub trait ReducePairsBulk<'s, S: SnippetExt<Item = Self::Item>, I: ?Sized>: SnippetExt {
    /// Takes a given schema and attempts to reduce valid choices by
    /// matching pairs. Select how many pairs will be compared at once.
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    ///
    /// Default is 2
    fn bulk_reduce_pairs<C>(
        &'s self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: C,
    ) -> Self
    where
        C: FnMut(S) -> I;
}

impl<T> ReducePairs<Snippet<'_, T>> for Phrase<T>
where
    T: Debug,
    Variation<T>: Clone,
{
    fn reduce_pairs<C>(
        &self,
        number_of_pairs: impl Into<Option<usize>>,
        confidence_interpreter: C,
    ) -> Self
    where
        C: Fn(&Variation<Self::Item>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_pairs(number_of_pairs, move |snip: Snippet<'_, T>| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<'s, 'b, T, S, I> ReducePairsBulk<'s, S, I> for Phrase<T>
where
    T: Debug + 'b,
    S: From<&'b BorrowedSnippet<T>> + SnippetExt<Item = Self::Item>,
    I: IntoIterator<Item = (f64, Variation<T>)>,
    Variation<T>: Clone,
    's: 'b,
{
    fn bulk_reduce_pairs<C>(
        &'s self,
        number_of_pairs: impl Into<Option<usize>>,
        mut confidence_interpreter: C,
    ) -> Self
    where
        C: FnMut(S) -> I,
    {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs.into() {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.len_sections() => n,
            Some(_) => self.len_sections(), /* If the number is bigger than the
                                             * source itself, just use the length of the source.
                                             * Its not
                                             * recommended to ever do this since its no different
                                             * than checking
                                             * line by line. */
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
            });
        Self::from_iter(new_sections)
    }
}

/// Provides and implements the reduction trait using the [`rayon`] library to
/// speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::{fmt::Debug, sync::Arc};

    use itertools::Itertools;
    use rayon::{
        iter::{FromParallelIterator, ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{
        snippet::{
            BorrowedSnippet, Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt,
        },
        variation::Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReducePairs<S: ThreadedSnippetExt<Item = Self::Item>>: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string
        /// is closer to your objective than another.
        ///
        /// Default is 2
        ///
        /// ```rust
        /// use std::sync::Arc;
        ///
        /// use base64_bruteforcer_rs::phrase::{
        ///     reduction::by_pairs::rayon::ParReducePairs,
        ///     schema::{
        ///         snippet::{BorrowedSnippet, Phrase, SnippetExt},
        ///         variation::Variation,
        ///     },
        ///     validation::validate_with_whatlang,
        /// };
        ///
        /// let phrase_string: Phrase<String> = [
        ///     vec![vec!["Hel"], vec!["HeR"]],
        ///     vec![vec!["lo "]],
        ///     vec![vec!["Wor"], vec!["WoX"]],
        ///     vec![vec!["ld!"]],
        ///     vec![vec!["Thi"], vec!["ThR"]],
        ///     vec![vec!["is "]],
        ///     vec![vec!["is "]],
        ///     vec![vec!["my "]],
        ///     vec![vec!["str"], vec!["stX"]],
        ///     vec![vec!["ing"], vec!["Mng"]],
        ///     vec![vec!["!"]],
        /// ]
        /// .into_iter()
        /// .map(|section| {
        ///     section.into_iter().map(|variation| {
        ///         variation
        ///             .into_iter()
        ///             .map(ToOwned::to_owned)
        ///             .map(Arc::new)
        ///             .collect::<Variation<String>>()
        ///     })
        /// })
        /// .collect();
        ///
        /// let reduced_phrase: Phrase<String> = [
        ///     vec![vec!["Hel", "lo "]],
        ///     vec![vec!["WoX", "ld!"]],
        ///     vec![vec!["Thi", "is "]],
        ///     vec![vec!["is ", "my "]],
        ///     vec![vec!["str", "ing"], vec!["stX", "Mng"]],
        ///     vec![vec!["!"]],
        /// ]
        /// .into_iter()
        /// .map(|section| {
        ///     section.into_iter().map(|variation| {
        ///         variation
        ///             .into_iter()
        ///             .map(ToOwned::to_owned)
        ///             .map(Arc::new)
        ///             .collect::<Variation<String>>()
        ///     })
        /// })
        /// .collect();
        ///
        /// let reduced_result = phrase_string.reduce_pairs(2, validate_with_whatlang);
        /// assert!(reduced_result == reduced_phrase);
        /// ```
        fn reduce_pairs<C>(
            &self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
            Variation<Self::Item>: Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction
    /// functions take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    pub trait ParReducePairsBulk<'s, S: ThreadedSnippetExt<Item = Self::Item>, I: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible
        /// permutations and produces an iterator of equal size with the
        /// confidence values of each string
        ///
        /// Default is 2
        fn bulk_reduce_pairs<C>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(S) -> I + Send + Sync,
            Variation<Self::Item>: Send;
    }

    impl<T> ParReducePairs<Snippet<'_, T>> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Send + Sync,
        Variation<T>: Clone,
    {
        fn reduce_pairs<C>(
            &self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(&Variation<T>) -> f64 + Send + Sync,
            Variation<T>: Send,
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

    impl<'s, 'b, T, S, I> ParReducePairsBulk<'s, S, I> for Phrase<T>
    where
        T: Debug + 'b,
        Arc<T>: Send + Sync,
        S: From<&'b BorrowedSnippet<T>> + ThreadedSnippetExt<Item = Self::Item>,
        I: IntoIterator<Item = (f64, Variation<T>)> + Sync,
        Variation<T>: Clone,
        's: 'b,
    {
        fn bulk_reduce_pairs<C>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>>,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(S) -> I + Send + Sync,
            Variation<T>: Send,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs.into() {
                Some(0..2) | None => 2, // Overwrite any stupid options with the
                // default
                Some(n) if n < self.len_sections() => n,
                Some(_) => self.len_sections(), /* If the number is bigger than the
                                                 * source itself, just use the length of the
                                                 * source. Its not
                                                 * recommended to ever do this since its no
                                                 * different than checking
                                                 * line by line. */
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
                        if !combined[0].is_empty() &&
                            (combined[0].len() as f64) < pairs.permutations()
                        {
                            combined
                        } else {
                            pairs.to_vec()
                        }
                    }
                });
            Self::from_par_iter(new_sections)
        }
    }
}

/// Provides and implements the reduction trait using the [`futures`] library to
/// speed up processes
#[cfg(feature = "async")]
pub mod r#async {
    use std::{fmt::Debug, sync::Arc};

    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{
        snippet::{
            BorrowedSnippet, Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt,
        },
        variation::Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    #[async_trait]
    pub trait AsyncReducePairs<'s, S: ThreadedSnippetExt<Item = Self::Item>>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// Takes a given schema and attempts to reduce valid choices by
        /// matching pairs. Select how many pairs will be compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string
        /// is closer to your objective than another.
        ///
        /// Default is 2
        ///
        /// ```rust
        /// # futures::executor::block_on(async {
        /// use std::sync::Arc;
        ///
        /// use base64_bruteforcer_rs::phrase::{
        ///     reduction::by_pairs::r#async::AsyncReducePairs,
        ///     schema::{
        ///         snippet::{BorrowedSnippet, Phrase, SnippetExt},
        ///         variation::Variation,
        ///     },
        ///     validation::validate_with_whatlang,
        /// };
        ///
        /// let phrase_string: Phrase<String> = [
        ///     vec![vec!["Hel"], vec!["HeR"]],
        ///     vec![vec!["lo "]],
        ///     vec![vec!["Wor"], vec!["WoX"]],
        ///     vec![vec!["ld!"]],
        ///     vec![vec!["Thi"], vec!["ThR"]],
        ///     vec![vec!["is "]],
        ///     vec![vec!["is "]],
        ///     vec![vec!["my "]],
        ///     vec![vec!["str"], vec!["stX"]],
        ///     vec![vec!["ing"], vec!["Mng"]],
        ///     vec![vec!["!"]],
        /// ]
        /// .into_iter()
        /// .map(|section| {
        ///     section.into_iter().map(|variation| {
        ///         variation
        ///             .into_iter()
        ///             .map(ToOwned::to_owned)
        ///             .map(Arc::new)
        ///             .collect::<Variation<String>>()
        ///     })
        /// })
        /// .collect();
        ///
        /// let reduced_phrase: Phrase<String> = [
        ///     vec![vec!["Hel", "lo "]],
        ///     vec![vec!["WoX", "ld!"]],
        ///     vec![vec!["Thi", "is "]],
        ///     vec![vec!["is ", "my "]],
        ///     vec![vec!["str", "ing"], vec!["stX", "Mng"]],
        ///     vec![vec!["!"]],
        /// ]
        /// .into_iter()
        /// .map(|section| {
        ///     section.into_iter().map(|variation| {
        ///         variation
        ///             .into_iter()
        ///             .map(ToOwned::to_owned)
        ///             .map(Arc::new)
        ///             .collect::<Variation<String>>()
        ///     })
        /// })
        /// .collect();
        ///
        /// let reduced_result = phrase_string
        ///     .reduce_pairs(2, async move |line| validate_with_whatlang(&line))
        ///     .await;
        /// assert!(reduced_result == reduced_phrase);
        /// # });
        /// ```
        async fn reduce_pairs<C, Fut>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(Variation<Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction
    /// functions take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    #[async_trait]
    pub trait AsyncReducePairsBulk<'s, S: ThreadedSnippetExt<Item = Self::Item>, I: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
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
        /// [`Phrase`]: crate::phrase::schema::snippet::Phrase
        /// [`Snippet`]: crate::phrase::schema::snippet::Snippet
        async fn bulk_reduce_pairs<C, Fut>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(S) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send;
    }

    #[async_trait]
    impl<'s, T> AsyncReducePairs<'s, Snippet<'_, T>> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        async fn reduce_pairs<C, Fut>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(Variation<Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_pairs(number_of_pairs, async |snip: Snippet<'_, T>| {
                stream::iter(snip.par_iter_var())
                    .then(async move |line| (conf_link(line.clone()).await, line))
                    .collect::<Vec<(f64, Variation<T>)>>()
                    .await
            })
            .await
        }
    }

    #[async_trait]
    impl<'s, 'b, T, S, I> AsyncReducePairsBulk<'s, S, I> for Phrase<T>
    where
        T: Debug + 'b,
        Arc<T>: Sync,
        S: From<&'b BorrowedSnippet<T>> + ThreadedSnippetExt<Item = Self::Item>,
        I: IntoIterator<Item = (f64, Variation<T>)>,
        Variation<T>: Clone + Send,
        's: 'b,
    {
        async fn bulk_reduce_pairs<C, Fut>(
            &'s self,
            number_of_pairs: impl Into<Option<usize>> + Send,
            confidence_interpreter: C,
        ) -> Self
        where
            C: Fn(S) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs.into() {
                Some(0..2) | None => 2, // Overwrite any stupid options with the
                // default
                Some(n) if n < self.len_sections() => n,
                Some(_) => self.len_sections(), /* If the number is bigger than the
                                                 * source itself, just use the length of the
                                                 * source. Its not
                                                 * recommended to ever do this since its no
                                                 * different than checking
                                                 * line by line. */
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
