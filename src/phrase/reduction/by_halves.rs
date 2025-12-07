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
pub trait ReduceHalves: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn reduce_halves<U>(
        &self,
        size_checker: impl Fn(&U) -> bool,
        confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64,
    ) -> Self
    where
        U: for<'a> From<&'a [Vec<Variation<Self::Item>>]> + SnippetExt<Item = Self::Item>,
        Variation<Self::Item>: Clone;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<U: SnippetExt<Item = Self::Item>, V: ?Sized>: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn bulk_reduce_halves(
        &self,
        size_checker: impl Fn(&U) -> bool,
        confidence_interpreter: impl FnMut(U) -> V,
    ) -> Self
    where
        U: for<'a> From<&'a [Vec<Variation<Self::Item>>]>,
        Variation<Self::Item>: Clone;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<'a, W, X>(
        size_checker: &W,
        phrase_snippet: &'a [Section<Self::Item>],
        confidence_interpreter: &mut X,
    ) -> Vec<Section<Self::Item>>
    where
        Variation<Self::Item>: Clone,
        U: From<&'a [Vec<Variation<Self::Item>>]>,
        W: Fn(&U) -> bool,
        X: FnMut(U) -> V;
}

impl<T> ReduceHalves for Phrase<T>
where
    T: Debug,
{
    fn reduce_halves<U>(
        &self,
        size_checker: impl Fn(&U) -> bool,
        confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64,
    ) -> Self
    where
        T: Debug,
        U: for<'a> From<&'a [Vec<Variation<T>>]> + SnippetExt<Item = Self::Item>,
        Variation<T>: Clone,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_halves(size_checker, move |snip| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<T, U, V> ReduceHalvesBulk<U, V> for Phrase<T>
where
    T: Debug,
    U: SnippetExt<Item = Self::Item>,
    V: IntoIterator<Item = (f64, Variation<T>)>,
{
    fn bulk_reduce_halves(
        &self,
        size_checker: impl Fn(&U) -> bool,
        mut confidence_interpreter: impl FnMut(U) -> V,
    ) -> Self
    where
        U: for<'a> From<&'a [Vec<Variation<Self::Item>>]>,
        Variation<T>: Clone,
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
        T: Debug,
        U: From<&'a [Vec<Variation<Self::Item>>]>,
        Variation<T>: Clone,
        W: Fn(&U) -> bool,
        X: FnMut(U) -> V,
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
        BorrowedSnippet, Permutation, Phrase, Section, SnippetExt, ThreadedSnippetExt, Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn reduce_halves<V: ThreadedSnippetExt<Item = Self::Item>>(
            &self,
            size_checker: impl Fn(&V) -> bool + Send + Sync,
            confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
        ) -> Self
        where
            Arc<Self::Item>: Sync,
            V: for<'a> From<&'a BorrowedSnippet<Self::Item>>;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<U: ThreadedSnippetExt<Item = Self::Item>, V: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves(
            &self,
            size_checker: impl Fn(&U) -> bool + Send + Sync,
            confidence_interpreter: impl Fn(U) -> V + Send + Sync,
        ) -> Self
        where
            Arc<Self::Item>: Sync,
            Variation<Self::Item>: Clone + Send,
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
            V: Send + Sync;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<'a, W, X>(
            size_checker: &W,
            phrase_snippet: &'a BorrowedSnippet<Self::Item>,
            confidence_interpreter: &X,
        ) -> Vec<Section<Self::Item>>
        where
            Arc<Self::Item>: Sync,
            Variation<Self::Item>: Clone + Send,
            U: From<&'a BorrowedSnippet<Self::Item>>,
            W: Fn(&U) -> bool + Send + Sync,
            X: Fn(U) -> V + Send + Sync,
            V: Send + Sync;
    }

    impl<T> ParReduceHalves for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        fn reduce_halves<U>(
            &self,
            size_checker: impl Fn(&U) -> bool + Send + Sync,
            confidence_interpreter: impl Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
        ) -> Self
        where
            Arc<T>: Sync,
            U: ThreadedSnippetExt<Item = Self::Item>
                + for<'a> From<&'a BorrowedSnippet<Self::Item>>,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, move |snip| {
                snip.par_iter_var()
                    .map(move |line| (conf_link(&line), line))
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }
    }

    impl<T, U, V> ParReduceHalvesBulk<U, V> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        U: ThreadedSnippetExt<Item = Self::Item>,
        V: IntoIterator<Item = (f64, Variation<T>)>,
    {
        fn bulk_reduce_halves(
            &self,
            size_checker: impl Fn(&U) -> bool + Send + Sync,
            confidence_interpreter: impl Fn(U) -> V + Send + Sync,
        ) -> Self
        where
            Arc<T>: Sync,
            Variation<T>: Clone + Send,
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>>,
            V: Send + Sync,
        {
            Self::from_iter(Self::bulk_reduce_schema_binary(
                &size_checker,
                self.as_ref(),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<'a, W, X>(
            size_checker: &W,
            phrase_snippet: &'a BorrowedSnippet<T>,
            confidence_interpreter: &X,
        ) -> Vec<Section<T>>
        where
            T: 'a,
            Arc<T>: Sync,
            Variation<T>: Clone + Send,
            U: ThreadedSnippetExt<Item = Self::Item> + From<&'a BorrowedSnippet<Self::Item>>,
            V: Send + Sync,
            W: Fn(&U) -> bool + Send + Sync,
            X: Fn(U) -> V + Send + Sync,
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
    pub trait AsyncReduceHalves: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        async fn reduce_halves<U, FutBool, Fut>(
            &self,
            size_checker: impl Fn(U) -> FutBool + Send + Sync,
            confidence_interpreter: impl for<'c> Fn(&'c Variation<Self::Item>) -> Fut + Send + Sync,
        ) -> Self
        where
            Variation<Self::Item>: Clone + Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = f64> + Send,
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>>
                + ThreadedSnippetExt<Item = Self::Item>
                + Send
                + Sync;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<U: ThreadedSnippetExt<Item = Self::Item>, V: ?Sized>:
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
        async fn bulk_reduce_halves<FutBool, Fut>(
            &self,
            size_checker: impl Fn(U) -> FutBool + Send + Sync,
            confidence_interpreter: impl Fn(U) -> Fut + Send + Sync,
        ) -> Self
        where
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>> + Send + Sync,
            Variation<Self::Item>: Clone + Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = V> + Send;

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
        async fn bulk_reduce_schema_binary<'a, W, FutBool, X, Fut>(
            size_checker: &W,
            phrase_snippet: &'a BorrowedSnippet<Self::Item>,
            confidence_interpreter: &X,
        ) -> Vec<Section<Self::Item>>
        where
            U: From<&'a BorrowedSnippet<Self::Item>> + Send + Sync,
            Variation<Self::Item>: Clone + Send,
            W: Fn(U) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            X: Fn(U) -> Fut + Send + Sync,
            Fut: Future<Output = V> + Send;
    }

    #[async_trait]
    impl<T> AsyncReduceHalves for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
    {
        async fn reduce_halves<U, FutBool, Fut>(
            &self,
            size_checker: impl Fn(U) -> FutBool + Send + Sync,
            confidence_interpreter: impl for<'c> Fn(&'c Variation<Self::Item>) -> Fut + Send + Sync,
        ) -> Self
        where
            Arc<Self::Item>: Sync,
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>>
                + ThreadedSnippetExt<Item = Self::Item>
                + Send
                + Sync,
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
    impl<T, U, V> AsyncReduceHalvesBulk<U, V> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        U: ThreadedSnippetExt<Item = Self::Item>,
        V: IntoIterator<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_halves<FutBool, Fut>(
            &self,
            size_checker: impl Fn(U) -> FutBool + Send + Sync,
            confidence_interpreter: impl Fn(U) -> Fut + Send + Sync,
        ) -> Self
        where
            Arc<T>: Sync,
            U: for<'a> From<&'a BorrowedSnippet<Self::Item>> + Send + Sync,
            Variation<T>: Clone + Send,
            FutBool: Future<Output = bool> + Send,
            Fut: Future<Output = V> + Send,
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

        async fn bulk_reduce_schema_binary<'a, W, FutBool, X, Fut>(
            size_checker: &W,
            phrase_snippet: &'a [Section<Self::Item>],
            confidence_interpreter: &X,
        ) -> Vec<Section<T>>
        where
            Arc<T>: Sync,
            U: From<&'a BorrowedSnippet<Self::Item>> + SnippetExt<Item = Self::Item> + Send + Sync,
            Variation<T>: Clone + Send,
            W: Fn(U) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            X: Fn(U) -> Fut + Send + Sync,
            Fut: Future<Output = V> + Send,
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
