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
//! To assume all variables, we can interpret `confidence_interpreter` as
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
//! [`reduce_halves`]: ReduceHalves::reduce_halves

use std::fmt::Debug;

use itertools::Itertools;

use crate::phrase::schema::{Permutation, Phrase, Section, SnippetExt, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<S: SnippetExt<Item = Self::Item>>: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn reduce_halves<L, C>(
        &self,
        size_checker: L,
        confidence_interpreter: C,
    ) -> Self
    where
        L: Fn(&S) -> bool,
        C: Fn(&Variation<Self::Item>) -> f64;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<S: SnippetExt<Item = Self::Item>, I: ?Sized>: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn bulk_reduce_halves<L, C>(
        &self,
        size_checker: L,
        confidence_interpreter: C,
    ) -> Self
    where
        L: Fn(&S) -> bool,
        C: FnMut(S) -> I,
        S: for<'a> From<&'a [Vec<Variation<Self::Item>>]>;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<'a, L, C>(
        size_checker: &L,
        phrase_snippet: &'a [Section<Self::Item>],
        confidence_interpreter: &mut C,
    ) -> Vec<Section<Self::Item>>
    where
        L: Fn(&S) -> bool,
        C: FnMut(S) -> I,
        S: From<&'a [Vec<Variation<Self::Item>>]>;
}

impl<T, S> ReduceHalves<S> for Phrase<T>
where
    T: Debug,
    S: for<'a> From<&'a [Vec<Variation<T>>]> + SnippetExt<Item = Self::Item>,
    Variation<T>: Clone,
{
    fn reduce_halves<L, C>(
        &self,
        size_checker: L,
        confidence_interpreter: C,
    ) -> Self
    where
        L: Fn(&S) -> bool,
        C: Fn(&Variation<Self::Item>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_halves(size_checker, move |snip| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<T, S, I> ReduceHalvesBulk<S, I> for Phrase<T>
where
    T: Debug,
    S: SnippetExt<Item = Self::Item>,
    I: IntoIterator<Item = (f64, Variation<T>)>,
    Variation<T>: Clone,
{
    fn bulk_reduce_halves<L, C>(
        &self,
        size_checker: L,
        mut confidence_interpreter: C,
    ) -> Self
    where
        L: Fn(&S) -> bool,
        C: FnMut(S) -> I,
        S: for<'a> From<&'a [Vec<Variation<Self::Item>>]>,
    {
        Self::from_iter(Self::bulk_reduce_schema_binary(
            &size_checker,
            self.as_ref(),
            &mut confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary<'a, W, X>(
        size_checker: &W,
        phrase_snippet: &'a [Section<Self::Item>],
        confidence_interpreter: &mut X,
    ) -> Vec<Section<T>>
    where
        W: Fn(&S) -> bool,
        X: FnMut(S) -> I,
        S: From<&'a [Vec<Variation<Self::Item>>]>,
    {
        // Leave early if section is empty or just one
        if phrase_snippet.len_sections() < 2 {
            phrase_snippet.as_ref().to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if size_checker(&phrase_snippet.into()) {
            let snippet_permutation = phrase_snippet.permutations();
            vec![
                confidence_interpreter(phrase_snippet.into())
                    .into_iter()
                    .inspect(|(confidence, line)| {
                        log::trace!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only square root of permitted permutations to allow rerunning the reduction
                    .k_largest_relaxed_by_key(
                        usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                        |(confidence, _)| (confidence * 100_000_f64) as usize,
                    )
                    .inspect(|(confidence, line)| {
                        log::trace!("Accepted: confidence, string: {confidence}, {line:?}")
                    })
                    .map(move |(_, line)| line)
                    .collect::<Section<T>>(),
            ]
        }
        // If permutations are still too big, split it again
        else {
            phrase_snippet
                .chunks(phrase_snippet.len_sections() / 2)
                .flat_map(move |s| {
                    Self::bulk_reduce_schema_binary(size_checker, s, confidence_interpreter)
                })
                .collect()
        }
    }
}

/// Provides and implements the reduction trait using the [`rayon`] library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::{fmt::Debug, sync::Arc};

    use itertools::Itertools;
    use rayon::{iter::ParallelIterator, slice::ParallelSlice};

    use crate::phrase::schema::{
        BorrowedSnippet, Permutation, Phrase, Section, SnippetExt, ThreadedSnippetExt, Variation
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves<S: ThreadedSnippetExt<Item = Self::Item>>: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn reduce_halves<L, C>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
            Arc<Self::Item>: Sync,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<S: ThreadedSnippetExt<Item = Self::Item>, I: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves<L, C>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync,
            Arc<Self::Item>: Sync,
            Variation<Self::Item>: Send,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
            I: Send + Sync;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<'a, L, C>(
            size_checker: &L,
            phrase_snippet: &'a BorrowedSnippet<Self::Item>,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            Arc<Self::Item>: Sync,
            Variation<Self::Item>: Clone + Send,
            S: From<&'a BorrowedSnippet<Self::Item>>,
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync,
            I: Send + Sync;
    }

    impl<T, S> ParReduceHalves<S> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        S: ThreadedSnippetExt<Item = Self::Item>,
        Variation<T>: Clone + Send,
    {
        fn reduce_halves<L, C>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
            Arc<T>: Sync,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, move |snip| {
                snip.par_iter_var()
                    .map(move |line| (conf_link(&line), line))
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }
    }

    impl<T, S, I> ParReduceHalvesBulk<S, I> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        S: ThreadedSnippetExt<Item = Self::Item>,
        I: IntoIterator<Item = (f64, Variation<T>)> + Send + Sync,
        Variation<T>: Clone,
    {
        fn bulk_reduce_halves<L, C>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync,
            Variation<T>: Send,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
        {
            Self::from_iter(Self::bulk_reduce_schema_binary(
                &size_checker,
                self.as_ref(),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<'s, L, C>(
            size_checker: &L,
            phrase_snippet: &'s BorrowedSnippet<T>,
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            T: 's,
            Variation<T>: Send,
            S: From<&'s BorrowedSnippet<Self::Item>>,
            L: Fn(S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.as_ref().to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(phrase_snippet.into()) {
                let snippet_permutation = phrase_snippet.permutations();
                vec![
                    confidence_interpreter(phrase_snippet.into())
                        .into_iter()
                        .inspect(|(confidence, line)| {
                            log::trace!("confidence, string: {confidence}, {line:?}")
                        })
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            // You may think you will want to use `sqrt().ceil()` but we want floor, because this
                            // prevents returned values from failing `size_checker` on the 2nd run of this function.
                            usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                            |(confidence, _)| (confidence * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::trace!("Accepted: confidence, string: {confidence}, {line:?}")
                        })
                        .map(move |(_, line)| line)
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                phrase_snippet
                    .par_chunks(phrase_snippet.len_sections() / 2)
                    .flat_map(move |s| {
                        Self::bulk_reduce_schema_binary(size_checker, s, confidence_interpreter)
                    })
                    .collect()
            }
        }
    }
}

/// Provides and implements the reduction trait using the asynchronous calls for smoother processing
#[cfg(feature = "async")]
pub mod r#async {
    use std::{fmt::Debug, sync::Arc};

    use async_trait::async_trait;
    use futures::{FutureExt, StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{
        BorrowedSnippet, Permutation, Phrase, Section, SnippetExt, ThreadedSnippetExt, Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalves<S: ThreadedSnippetExt<Item = Self::Item>>: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        async fn reduce_halves<L, C, FutBool, Fut>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> FutBool + Send + Sync,
            C: for<'c> Fn(&'c Variation<Self::Item>) -> Fut + Send + Sync,
            Variation<Self::Item>: Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = f64> + Send,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<S: ThreadedSnippetExt<Item = Self::Item>, I: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_reduce_halves<L, C, FutBool, Fut>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> FutBool + Send + Sync,
            C: Fn(S) -> Fut + Send + Sync,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>> + Send + Sync,
            Variation<Self::Item>: Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = I> + Send;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_reduce_schema_binary<'s, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: &'s BorrowedSnippet<Self::Item>,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            S: From<&'s BorrowedSnippet<Self::Item>> + Send + Sync,
            Variation<Self::Item>: Clone + Send,
            L: Fn(S) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(S) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send;
    }

    #[async_trait]
    impl<T, S> AsyncReduceHalves<S> for Phrase<T>
    where
        T: Debug,
        S: ThreadedSnippetExt<Item = Self::Item> + Send + Sync,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        async fn reduce_halves<L, C, FutBool, Fut>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> FutBool + Send + Sync,
            C: for<'c> Fn(&'c Variation<Self::Item>) -> Fut + Send + Sync,
            Arc<Self::Item>: Sync,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = f64> + Send,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, async |snip| {
                stream::iter(snip.par_iter_var())
                    .then(async move |line| (conf_link(&line).await, line))
                    .collect::<Vec<(f64, Variation<T>)>>()
                    .await
            })
            .boxed()
            .await
        }
    }

    #[async_trait]
    impl<T, S, I> AsyncReduceHalvesBulk<S, I> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        S: ThreadedSnippetExt<Item = Self::Item> + Send + Sync,
        I: IntoIterator<Item = (f64, Variation<T>)>,
        Variation<T>: Clone,
    {
        async fn bulk_reduce_halves<L, C, FutBool, Fut>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(S) -> FutBool + Send + Sync,
            C: Fn(S) -> Fut + Send + Sync,
            Arc<T>: Sync,
            S: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
            Variation<T>: Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = I> + Send,
        {
            Self::from_iter(
                Self::bulk_reduce_schema_binary(
                    &size_checker,
                    self.as_ref(),
                    &confidence_interpreter,
                )
                .await,
            )
        }

        async fn bulk_reduce_schema_binary<'s, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: &'s [Section<Self::Item>],
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            Arc<T>: Sync,
            S: From<&'s BorrowedSnippet<Self::Item>>,
            Variation<T>: Send,
            L: Fn(S) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(S) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.as_ref().to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(phrase_snippet.into()).await {
                let snippet_permutation = phrase_snippet.permutations();
                vec![
                    confidence_interpreter(phrase_snippet.into())
                        .await
                        .into_iter()
                        .inspect(|(confidence, line)| {
                            log::trace!("confidence, string: {confidence}, {line:?}")
                        })
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                            |(confidence, _)| (confidence * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::trace!("Accepted: confidence, string: {confidence}, {line:?}")
                        })
                        .map(move |(_, line)| line)
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                let phrase_len = phrase_snippet.len_sections();
                stream::iter(phrase_snippet.chunks(phrase_len / 2))
                    .then(async move |s| {
                        stream::iter(
                            Self::bulk_reduce_schema_binary(
                                size_checker,
                                s,
                                confidence_interpreter,
                            )
                            .await,
                        )
                    })
                    .boxed()
                    .flatten()
                    .collect()
                    .await
            }
        }
    }
}
