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

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
pub trait ReducePairs<'a, T> {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once.
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    ///
    /// Default is 2
    fn reduce_pairs<'b, U>(
        &'a self,
        number_of_pairs: Option<usize>,
        confidence_interpreter: &'b U,
    ) -> Self
    where
        Self: ReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        U: Fn(&Variation<T>) -> f64,
        'a: 'b;

    /// Runs the reduce function until the it will not reduce anymore
    ///
    /// `confidence_interpreter` is used to determine if a combined string is
    /// closer to your objective than another.
    fn pairs_to_end<'b, U>(
        &'a self,
        recursive_val: Option<(usize, usize)>,
        confidence_interpreter: &'b U,
    ) -> Self
    where
        Self: ReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        U: Fn(&Variation<T>) -> f64,
        'a: 'b;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing pairs
///
/// While this is similar in function to [`ReducePairs`], the reduction functions
/// take a validator that takes values in bulk
pub trait ReducePairsBulk<'a, T, U> {
    /// Takes a given schema and attempts to. Select how many pairs will be
    /// compared at once.
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    ///
    /// Default is 2
    fn bulk_reduce_pairs<V>(
        &'a self,
        number_of_pairs: Option<usize>,
        confidence_interpreter: V,
    ) -> Self
    where
        T: 'a,
        V: FnMut(Snippet<'a, T>) -> U;

    /// Runs the reduce function until the it will not reduce anymore
    ///
    /// `confidence_interpreter` Takes an iterator of all possible permutations
    /// and produces an iterator of equal size with the confidence values of
    /// each string
    fn bulk_pairs_to_end<V>(
        &'a self,
        recursive_val: Option<(usize, usize)>,
        confidence_interpreter: V,
    ) -> Self
    where
        T: 'a,
        V: FnMut(Snippet<'a, T>) -> U;
}

impl<'a, T> ReducePairs<'a, T> for Phrase<T> {
    fn reduce_pairs<'b, U>(
        &'a self,
        number_of_pairs: Option<usize>,
        confidence_interpreter: &'b U,
    ) -> Self
    where
        Self: ReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        U: Fn(&Variation<T>) -> f64,
        'a: 'b,
    {
        self.bulk_reduce_pairs(number_of_pairs, |snip| {
            Box::new(
                snip.into_iter_var()
                    .map(|line| (confidence_interpreter(&line), line)),
            )
        })
    }

    fn pairs_to_end<'b, U>(
        &'a self,
        recursive_val: Option<(usize, usize)>,
        confidence_interpreter: &'b U,
    ) -> Self
    where
        Self: ReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        U: Fn(&Variation<T>) -> f64,
        'a: 'b,
    {
        self.bulk_pairs_to_end(recursive_val, |snip| {
            Box::new(
                snip.into_iter_var()
                    .into_iter()
                    .map(|line| (confidence_interpreter(&line), line)),
            )
        })
    }
}

impl<'a, T, U> ReducePairsBulk<'a, T, U> for Phrase<T>
where
    T: Clone + Debug,
    U: Iterator<Item = (f64, Variation<T>)>,
{
    fn bulk_reduce_pairs<V>(
        &'a self,
        number_of_pairs: Option<usize>,
        mut confidence_interpreter: V,
    ) -> Self
    where
        T: 'a,
        V: FnMut(Snippet<'a, T>) -> U,
    {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
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
            .flat_map(|pairs| {
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    pairs.to_vec()
                }
                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                else if pairs.iter().all(|v| v.len() == 1) {
                    vec![vec![Variation::join_var_vec(
                        pairs.iter().map(|s| &s[0]).collect::<Vec<&Variation<T>>>(),
                    )]]
                } else {
                    // permuting values and collecting only viable options
                    let combined: Vec<Section<T>> = vec![{
                        let pair_snippet = Snippet::new(pairs);
                        let snippet_permutations = pair_snippet.permutations();
                        confidence_interpreter(pair_snippet)
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            // Keeping only half the values to make actual leeway
                            .k_largest_relaxed_by_key(
                                (snippet_permutations / 2_f64).ceil() as usize,
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
            .collect::<Vec<Section<T>>>();
        Self::new(new_sections)
    }

    fn bulk_pairs_to_end<V>(
        &'a self,
        recursive_val: Option<(usize, usize)>,
        confidence_interpreter: V,
    ) -> Self
    where
        T: 'a,
        V: FnMut(Snippet<'a, T>) -> U,
        Self: 'a,
    {
        if let Some((pair_size, last_size)) = recursive_val {
            // Currently in recursive loop
            // Collecting section len to determine if ending or not
            if pair_size < self.len_sections() {
                return self.clone();
            } else if last_size <= self.len_sections() {
                self.bulk_pairs_to_end(Some((pair_size + 1, usize::MAX)), confidence_interpreter)
            } else {
                self.bulk_pairs_to_end(
                    Some((pair_size, self.len_sections())),
                    confidence_interpreter,
                )
            }
        } else {
            // Setting up initial recursion
            self.bulk_pairs_to_end(Some((2, usize::MAX)), confidence_interpreter)
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

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReducePairs<'a, T> {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string is
        /// closer to your objective than another.
        ///
        /// Default is 2
        fn reduce_pairs<'b, U>(
            &'a self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: &'b U,
        ) -> Self
        where
            T: 'a + Send + Sync,
            Self: ParReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
            U: Fn(&Variation<T>) -> f64 + Send + Sync,
            'a: 'b;

        /// Runs the reduce function until the it will not reduce anymore
        fn pairs_to_end<'b, U>(
            &'a self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: &'b U,
        ) -> Self
        where
            T: 'a + Send + Sync,
            Self: ParReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
            U: Fn(&Variation<T>) -> f64 + Send + Sync,
            'a: 'b;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction functions
    /// take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    pub trait ParReducePairsBulk<'a, T, U> {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        ///
        /// Default is 2
        fn bulk_reduce_pairs<V>(
            &'a self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: V,
        ) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(Snippet<'a, T>) -> U + Send + Sync;

        /// Runs the reduce function until the it will not reduce anymore
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        fn bulk_pairs_to_end<V>(
            &'a self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: V,
        ) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(Snippet<'a, T>) -> U + Send + Sync;
    }

    impl<'a, T> ParReducePairs<'a, T> for Phrase<T>
    where
        T: Send + Sync,
    {
        fn reduce_pairs<'b, U>(
            &'a self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: &'b U,
        ) -> Self
        where
            T: Send + Sync,
            Self: ParReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
            U: Fn(&Variation<T>) -> f64 + Send + Sync,
            'a: 'b,
        {
            self.bulk_reduce_pairs(number_of_pairs, |snip| {
                Box::new(
                    snip
                        // Get all combinations of the variations
                        // Join them together to get the string to test against
                        .into_iter_var()
                        .par_bridge()
                        // Use detector to gain a confidence on each line
                        .map(|line| (confidence_interpreter(&line), line))
                        // Collecting here to drop to a regular iterator
                        .collect::<Vec<(f64, Variation<T>)>>()
                        .into_iter(),
                )
            })
        }

        fn pairs_to_end<'b, U>(
            &'a self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: &'b U,
        ) -> Self
        where
            T: Send + Sync,
            Self: ParReducePairsBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
            U: Fn(&Variation<T>) -> f64 + Send + Sync,
            'a: 'b,
        {
            self.bulk_pairs_to_end(recursive_val, |snip| {
                Box::new(
                    snip
                        // Get all combinations of the variations
                        // Join them together to get the string to test against
                        .into_iter_var()
                        .par_bridge()
                        // Use detector to gain a confidence on each line
                        .map(|line| (confidence_interpreter(&line), line))
                        // Collecting here to drop to a regular iterator
                        .collect::<Vec<(f64, Variation<T>)>>()
                        .into_iter(),
                )
            })
        }
    }

    impl<'a, T, U> ParReducePairsBulk<'a, T, U> for Phrase<T>
    where
        T: 'a + Clone + Debug + Send + Sync,
        U: Iterator<Item = (f64, Variation<T>)> + Sync,
        Variation<T>: Display,
    {
        fn bulk_reduce_pairs<V>(
            &'a self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: V,
        ) -> Self
        where
            T: 'a + Clone + Debug + Send + Sync,
            V: Fn(Snippet<'a, T>) -> U + Send + Sync,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs {
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
                .flat_map(|pairs| {
                    // If its only 1 pair, we can skip this process
                    if pairs.len() == 1 {
                        pairs.to_vec()
                    }
                    // If there is more than one pair, but each pair only has one
                    // value, then just return a single combined form. It will give
                    // future runs more information and clarity
                    else if pairs.iter().all(|v| v.len() == 1) {
                        vec![vec![Variation::join_var_vec(
                            pairs.iter().map(|s| &s[0]).collect::<Vec<&Variation<T>>>(),
                        )]]
                    } else {
                        // permuting values and collecting only viable options
                        let combined: Vec<Section<T>> = vec![{
                            let pair_snippet = Snippet::new(pairs);
                            let pair_permutation = pair_snippet.permutations();
                            confidence_interpreter(pair_snippet)
                                .inspect(|(confidence, line)| {
                                    log::debug!("confidence, string: {confidence}, {line:?}")
                                })
                                // Keeping only half the values to make actual leeway
                                .k_largest_relaxed_by_key(
                                    (pair_permutation / 2_f64).ceil() as usize,
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
                .collect::<Vec<Section<T>>>();
            Self::new(new_sections)
        }

        fn bulk_pairs_to_end<V>(
            &'a self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: V,
        ) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(Snippet<'a, T>) -> U + Send + Sync,
            Variation<T>: Display,
        {
            if let Some((pair_size, last_size)) = recursive_val {
                // Currently in recursive loop
                // Collecting section len to determine if ending or not
                if pair_size < self.len_sections() {
                    return self.clone();
                } else if last_size <= self.len_sections() {
                    self.bulk_pairs_to_end(
                        Some((pair_size + 1, usize::MAX)),
                        confidence_interpreter,
                    )
                } else {
                    self.bulk_pairs_to_end(
                        Some((pair_size, self.len_sections())),
                        confidence_interpreter,
                    )
                }
            } else {
                // Setting up initial recursion
                self.bulk_pairs_to_end(Some((2, usize::MAX)), confidence_interpreter)
            }
        }
    }
}

/// Provides and implements the reduction trait using the rayon library to speed up processes
#[cfg(feature = "async")]
pub mod r#async {
    use std::fmt::Debug;

    use futures::{StreamExt, future::BoxFuture, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, Section, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait AsyncReducePairs<'a, T>
    {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` is used to determine if a combined string is
        /// closer to your objective than another.
        ///
        /// Default is 2
        fn reduce_pairs<'b, U, Fut>(
            &'a self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: &'b U,
        ) -> impl Future<Output = Self> + Send
        where
            T: 'a + Send + Sync,
            Self: AsyncReducePairsBulk<T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>, BoxFuture<'b, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>>,
            U: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
            'a: 'b;

        /// Runs the reduce function until the it will not reduce anymore
        fn pairs_to_end<'b, U, Fut>(
            &'a self,
            confidence_interpreter: &'b U,
        ) -> impl Future<Output = Self> + Send
        where
            T: 'a + Send + Sync,
            Self: AsyncReducePairsBulk<T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>, BoxFuture<'b, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>>,
            U: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
            'a: 'b;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing pairs
    ///
    /// While this is similar in function to [`ReducePairs`], the reduction functions
    /// take a validator that takes values in bulk
    ///
    /// [`ReducePairs`]: super::ReducePairs
    pub trait AsyncReducePairsBulk<T, U, Fut> {
        /// Takes a given schema and attempts to. Select how many pairs will be
        /// compared at once.
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        ///
        /// Default is 2
        fn bulk_reduce_pairs<V>(
            &self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send
        where
            T: Send + Sync,
            V: Fn(Phrase<T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send;

        /// Runs the reduce function until the it will not reduce anymore
        ///
        /// `confidence_interpreter` Takes an iterator of all possible permutations
        /// and produces an iterator of equal size with the confidence values of
        /// each string
        fn bulk_pairs_to_end<V>(
            &self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send
        where
            T: Send + Sync,
            V: Fn(Phrase<T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send;
    }

    // impl<'a, T> AsyncReducePairs<'a, T> for Phrase<T> {
    //     fn reduce_pairs<'b, U, Fut>(
    //         &'a self,
    //         number_of_pairs: Option<usize>,
    //         confidence_interpreter: &'b U,
    //     ) -> impl Future<Output = Self> + Send
    //     where
    //         T: 'a + Send + Sync,
    //         Self: AsyncReducePairsBulk<T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>, BoxFuture<'b, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>>,
    //         U: Fn(&Variation<T>) -> Fut + Send + Sync,
    //         Fut: Future<Output = f64> + Send,
    //         'a: 'b,
    //     {
    //         async move {
    //             self.bulk_reduce_pairs(number_of_pairs, async move |snip: Phrase<T>| {
    //                 Box::new(stream::iter(snip.iter_var()).then(async |line| (confidence_interpreter(&line).await, line)).collect::<Vec<(f64, Variation<T>)>>().await.iter())
    //             })
    //         }.boxed()
    //     }
    
    //     fn pairs_to_end<'b, U, Fut>(
    //         &'a self,
    //         confidence_interpreter: &'b U,
    //     ) -> impl Future<Output = Self> + Send
    //     where
    //         T: 'a + Send + Sync,
    //         Self: AsyncReducePairsBulk<T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>, Pin<Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>>,
    //         U: Fn(&Variation<T>) -> Fut + Send + Sync,
    //         Fut: Future<Output = f64> + Send,
    //         'a: 'b {
    //         todo!()
    //     }
    // }

    impl<T, U, Fut> AsyncReducePairsBulk<T, U, Fut> for Phrase<T>
    where
        T: Clone + Debug + Send + Sync,
        U: Iterator<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_pairs<V>(
            &self,
            number_of_pairs: Option<usize>,
            confidence_interpreter: V,
        ) -> Self
        where
            T: Clone + Debug + Send + Sync,
            V: Fn(Phrase<T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
        {
            // Check to make sure size is correctly placed or replace with own value
            let pair_size = match number_of_pairs {
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
            let new_sections = stream::iter(self.sections.clone())
                .chunks(pair_size)
                .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
                .then(async |pairs| {
                    // If its only 1 pair, we can skip this process
                    if pairs.len() == 1 {
                        stream::iter(pairs)
                    }
                    // If there is more than one pair, but each pair only has one
                    // value, then just return a single combined form. It will give
                    // future runs more information and clarity
                    else if pairs.iter().all(|v| v.len() == 1) {
                        stream::iter(vec![vec![Variation::join_var_vec(
                            pairs.iter().map(|s| &s[0]).collect::<Vec<&Variation<T>>>(),
                        )]])
                    } else {
                        // permuting values and collecting only viable options
                        let pair_snippet = Phrase::new(pairs.to_owned());
                        let pair_permutation = pair_snippet.permutations();
                        let combined: Vec<Section<T>> = vec![
                            confidence_interpreter(pair_snippet)
                                .await
                                .inspect(|(confidence, line)| {
                                    log::debug!("confidence, string: {confidence}, {line:?}")
                                })
                                // Keeping only half the values to make actual leeway
                                .k_largest_relaxed_by_key(
                                    (pair_permutation / 2_f64).ceil() as usize,
                                    |(confidence, _)| (confidence * 100_000_f64) as usize,
                                )
                                .inspect(|(confidence, line)| {
                                    log::debug!(
                                        "Accepted: confidence, string: {confidence}, {line:?}"
                                    )
                                })
                                .map(|(_, line)| line)
                                .collect(),
                        ];

                        // Go with originals if new choices aren't preferred
                        // aka if its empty or the permutations is the same as it originally was
                        if !combined[0].is_empty() && (combined[0].len() as f64) < pair_permutation
                        {
                            stream::iter(combined)
                        } else {
                            stream::iter(pairs)
                        }
                    }
                })
                .boxed()
                .flatten()
                .collect::<Vec<Section<T>>>()
                .await;
            Self::new(new_sections)
        }

        fn bulk_pairs_to_end<V>(
            &self,
            recursive_val: Option<(usize, usize)>,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send
        where
            V: Fn(Phrase<T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
        {
            if let Some((pair_size, last_size)) = recursive_val {
                // Currently in recursive loop
                // Collecting section len to determine if ending or not
                if pair_size < self.len_sections() {
                    async move {
                        self.clone()
                    }
                } else if last_size <= self.len_sections() {
                    self.bulk_pairs_to_end(
                        Some((pair_size + 1, usize::MAX)),
                        confidence_interpreter,
                    )
                } else {
                    self.bulk_pairs_to_end(
                        Some((pair_size, self.len_sections())),
                        confidence_interpreter,
                    )
                }
            } else {
                // Setting up initial recursion
                self.bulk_pairs_to_end(Some((2, usize::MAX)), confidence_interpreter)
            }
        }
    }
}
