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

use std::{borrow::Borrow, fmt::Debug};

use itertools::Itertools;

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<'s, S: SnippetExt<Item = Self::Item>, B: SnippetExt<Item = Self::Item>>:
    SnippetExt
{
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
    where
        Variation<Self::Item>: Clone,
        L: Fn(&B) -> bool,
        C: Fn(&Variation<Self::Item>) -> f64,
        S: Borrow<B>;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<
    's,
    S: SnippetExt<Item = Self::Item>,
    B: SnippetExt<Item = Self::Item>,
    I: ?Sized,
>: SnippetExt
{
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn bulk_reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
    where
        L: Fn(&B) -> bool,
        C: FnMut(S) -> I;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<L, C>(
        size_checker: &L,
        phrase_snippet: S,
        confidence_interpreter: &mut C,
    ) -> Vec<Section<Self::Item>>
    where
        L: Fn(&B) -> bool,
        C: FnMut(S) -> I,
        S: Borrow<B>;
}

impl<'s, 'b, T> ReduceHalves<'s, Snippet<'b, T>, Snippet<'b, T>> for Phrase<T>
where
    T: Debug,
    's: 'b,
{
    fn reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
    where
        T: Debug,
        Variation<T>: Clone,
        L: Fn(&Snippet<'b, T>) -> bool,
        C: Fn(&Variation<T>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_halves(size_checker, move |s: Snippet<'b, T>| {
            s.into_iter_var().map(move |v| (conf_link(&v), v))
        })
    }
}

impl<'s, 'b, T, I> ReduceHalvesBulk<'s, Snippet<'b, T>, Snippet<'b, T>, I> for Phrase<T>
where
    T: Debug + 'b,
    Variation<T>: Clone,
    I: IntoIterator<Item = (f64, Variation<T>)>,
    's: 'b,
{
    fn bulk_reduce_halves<L, C>(&'s self, size_checker: L, mut confidence_interpreter: C) -> Self
    where
        L: Fn(&Snippet<'b, T>) -> bool,
        C: FnMut(Snippet<'b, T>) -> I,
    {
        Self::new(Self::bulk_reduce_schema_binary(
            &size_checker,
            self.as_snippet(),
            &mut confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary<L, C>(
        size_checker: &L,
        phrase_snippet: Snippet<'b, T>,
        confidence_interpreter: &mut C,
    ) -> Vec<Section<T>>
    where
        L: Fn(&Snippet<'b, T>) -> bool,
        C: FnMut(Snippet<'b, T>) -> I,
    {
        // Leave early if section is empty or just one
        if phrase_snippet.len_sections() < 2 {
            phrase_snippet.sections.to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if size_checker(&phrase_snippet) {
            let snippet_permutation = phrase_snippet.permutations();
            vec![
                confidence_interpreter(phrase_snippet)
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
                .sections
                .chunks(phrase_snippet.len_sections() / 2)
                .map_into()
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
    use std::{borrow::Borrow, fmt::Debug, sync::Arc};

    use itertools::Itertools;
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{
        Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt, Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves<
        's,
        S: ThreadedSnippetExt<Item = Self::Item>,
        B: ThreadedSnippetExt<Item = Self::Item>,
    >: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
        where
            L: Fn(&B) -> bool + Send + Sync,
            C: Fn(&Variation<Self::Item>) -> f64 + Send + Sync,
            S: Borrow<B>;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<'s, S: ThreadedSnippetExt<Item = Self::Item>, I: ?Sized>:
        ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
        where
            L: Fn(&S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<L, C>(
            size_checker: &L,
            phrase_snippet: S,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            L: Fn(&S) -> bool + Send + Sync,
            C: Fn(S) -> I + Send + Sync;
    }

    impl<'s, 'b, T> ParReduceHalves<'s, Snippet<'b, T>, Snippet<'b, T>> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
        's: 'b,
    {
        fn reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
        where
            L: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            C: Fn(&Variation<T>) -> f64 + Send + Sync,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, move |snip| {
                snip.par_iter_var()
                    .par_bridge()
                    .map(move |line| (conf_link(&line), line))
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }
    }

    impl<'s, 'b, T, I> ParReduceHalvesBulk<'s, Snippet<'b, T>, I> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
        I: IntoIterator<Item = (f64, Variation<T>)>,
        's: 'b,
    {
        fn bulk_reduce_halves<L, C>(&'s self, size_checker: L, confidence_interpreter: C) -> Self
        where
            L: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, T>) -> I + Send + Sync,
        {
            Self::new(Self::bulk_reduce_schema_binary(
                &size_checker,
                self.as_snippet(),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<L, C>(
            size_checker: &L,
            phrase_snippet: Snippet<'b, T>,
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            L: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, T>) -> I + Send + Sync,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(&phrase_snippet) {
                let snippet_permutation = phrase_snippet.permutations();
                vec![
                    confidence_interpreter(phrase_snippet)
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
                    .sections
                    .par_chunks(phrase_snippet.len_sections() / 2)
                    .map(|v| v.into())
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
    use futures::{StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{
        Permutation, Phrase, Section, Snippet, SnippetExt, ThreadedSnippetExt, Variation,
    };

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalves<'s>: ThreadedSnippetExt
    where
        Arc<Self::Item>: Sync,
    {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        async fn reduce_halves<'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(Snippet<'b, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(&Variation<Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
            's: 'b;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<'s, I: ?Sized>: ThreadedSnippetExt
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
        async fn bulk_reduce_halves<'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            Self::Item: 'b,
            L: Fn(Snippet<'b, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
            Variation<Self::Item>: Send,
            's: 'b;

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
        async fn bulk_reduce_schema_binary<'b, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: Snippet<'s, Self::Item>,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            Self::Item: 'b,
            L: Fn(Snippet<'b, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(Snippet<'b, Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
            Variation<Self::Item>: Send,
            's: 'b;
    }

    #[async_trait]
    impl<'s, T> AsyncReduceHalves<'s> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
        for<'a> Snippet<'a, T>: Clone,
    {
        async fn reduce_halves<'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
            's: 'b,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, async |snip: Snippet<'_, T>| {
                stream::iter(snip.iter_var())
                    .then(async move |line| (conf_link(&line).await, line))
                    .collect::<Vec<(f64, Variation<T>)>>()
                    .await
            })
            .await
        }
    }

    #[async_trait]
    impl<'s, T, I> AsyncReduceHalvesBulk<'s, I> for Phrase<T>
    where
        T: Debug,
        Arc<T>: Sync,
        Variation<T>: Clone + Send,
        for<'a> Snippet<'a, T>: Clone,
        I: IntoIterator<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_halves<'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            T: 'b,
            L: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
            's: 'b,
        {
            Self::new(
                Self::bulk_reduce_schema_binary(
                    &size_checker,
                    self.as_snippet(),
                    &confidence_interpreter,
                )
                .await,
            )
        }

        async fn bulk_reduce_schema_binary<'b, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: Snippet<'s, T>,
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            T: 'b,
            L: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
            's: 'b,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(phrase_snippet.clone()).await {
                let snippet_permutation = phrase_snippet.permutations();
                vec![
                    confidence_interpreter(phrase_snippet)
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
                stream::iter(phrase_snippet.sections.chunks(phrase_len / 2))
                    .map(Snippet::new)
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
