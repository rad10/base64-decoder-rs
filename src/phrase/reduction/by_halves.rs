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

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn reduce_halves<L, C>(&self, size_checker: L, confidence_interpreter: C) -> Self
    where
        Variation<Self::Item>: Clone,
        L: Fn(&Snippet<'_, Self::Item>) -> bool,
        C: Fn(&Variation<Self::Item>) -> f64;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<I: ?Sized>: SnippetExt {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `size_checker`
    /// to decide the size of the section.
    fn bulk_reduce_halves<'a, 'b, L, C>(
        &'a self,
        size_checker: L,
        confidence_interpreter: C,
    ) -> Self
    where
        Self::Item: 'b,
        Variation<Self::Item>: Clone,
        L: Fn(&Snippet<'b, Self::Item>) -> bool,
        C: FnMut(Snippet<'b, Self::Item>) -> I,
        'a: 'b;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<'a, 'b, L, C>(
        size_checker: &L,
        phrase_snippet: Snippet<'a, Self::Item>,
        confidence_interpreter: &mut C,
    ) -> Vec<Section<Self::Item>>
    where
        Variation<Self::Item>: Clone,
        L: Fn(&Snippet<'b, Self::Item>) -> bool,
        C: FnMut(Snippet<'b, Self::Item>) -> I,
        'a: 'b;
}

impl<T> ReduceHalves for Phrase<T>
where
    T: Debug,
{
    fn reduce_halves<L, C>(&self, size_checker: L, confidence_interpreter: C) -> Self
    where
        T: Debug,
        Variation<T>: Clone,
        L: Fn(&Snippet<'_, T>) -> bool,
        C: Fn(&Variation<T>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_halves(size_checker, move |snip| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<T, I> ReduceHalvesBulk<I> for Phrase<T>
where
    T: Debug,
    I: IntoIterator<Item = (f64, Variation<T>)>,
{
    fn bulk_reduce_halves<'a, 'b, L, C>(
        &'a self,
        size_checker: L,
        mut confidence_interpreter: C,
    ) -> Self
    where
        T: 'b,
        Variation<T>: Clone,
        L: Fn(&Snippet<'b, T>) -> bool,
        C: FnMut(Snippet<'b, T>) -> I,
        'a: 'b,
    {
        Self::new(Self::bulk_reduce_schema_binary(
            &size_checker,
            self.as_snippet(),
            &mut confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary<'a, 'b, L, C>(
        size_checker: &L,
        phrase_snippet: Snippet<'a, T>,
        confidence_interpreter: &mut C,
    ) -> Vec<Section<T>>
    where
        T: Debug,
        Variation<T>: Clone,
        L: Fn(&Snippet<'b, T>) -> bool,
        C: FnMut(Snippet<'b, T>) -> I,
        'a: 'b,
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
                .map(Snippet::new)
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
    use std::fmt::Debug;

    use itertools::Itertools;
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves: SnippetExt {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn reduce_halves<L, C>(&self, size_checker: L, confidence_interpreter: C) -> Self
        where
            Self::Item: Send + Sync,
            L: Fn(&Snippet<'_, Self::Item>) -> bool + Send + Sync,
            C: Fn(&Variation<Self::Item>) -> f64 + Send + Sync;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<I: ?Sized>: SnippetExt {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves<'a, 'b, L, C>(
            &'a self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            Self::Item: 'b + Send + Sync,
            Variation<Self::Item>: Clone,
            L: Fn(&Snippet<'b, Self::Item>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, Self::Item>) -> I + Send + Sync,
            I: Send + Sync,
            'a: 'b;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<'a, 'b, L, C>(
            size_checker: &L,
            phrase_snippet: Snippet<'a, Self::Item>,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            Self::Item: Send + Sync,
            Variation<Self::Item>: Clone,
            L: Fn(&Snippet<'b, Self::Item>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, Self::Item>) -> I + Send + Sync,
            I: Send + Sync,
            'a: 'b;
    }

    impl<T> ParReduceHalves for Phrase<T>
    where
        T: Debug,
        Variation<T>: Clone,
    {
        fn reduce_halves<L, C>(&self, size_checker: L, confidence_interpreter: C) -> Self
        where
            T: Send + Sync,
            L: Fn(&Snippet<'_, T>) -> bool + Send + Sync,
            C: Fn(&Variation<T>) -> f64 + Send + Sync,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, move |snip| {
                snip.into_iter_var()
                    .par_bridge()
                    .map(move |line| (conf_link(&line), line))
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }
    }

    impl<T, I> ParReduceHalvesBulk<I> for Phrase<T>
    where
        T: Debug,
        I: IntoIterator<Item = (f64, Variation<T>)>,
    {
        fn bulk_reduce_halves<'a, 'b, L, C>(
            &'a self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            T: 'b + Send + Sync,
            Variation<T>: Clone,
            L: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, T>) -> I + Send + Sync,
            I: Send + Sync,
            'a: 'b,
        {
            Self::new(Self::bulk_reduce_schema_binary(
                &size_checker,
                self.as_snippet(),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<'s, 'b, L, C>(
            size_checker: &L,
            phrase_snippet: Snippet<'s, T>,
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            T: 'b + Send + Sync,
            Variation<T>: Clone,
            L: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            C: Fn(Snippet<'b, T>) -> I + Send + Sync,
            I: Send + Sync,
            's: 'b,
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
                    .map(Snippet::new)
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
    use std::fmt::Debug;

    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, SnippetExt, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalves: SnippetExt {
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
            Self::Item: Send + Sync,
            Variation<Self::Item>: Clone,
            L: Fn(Snippet<'_, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(&Variation<Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<I: ?Sized>: SnippetExt {
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
        async fn bulk_reduce_halves<'s, 'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            Self: Clone,
            Snippet<'s, Self::Item>: Clone,
            Self::Item: 's + Send + Sync,
            Variation<Self::Item>: Clone,
            L: Fn(Snippet<'b, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
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
        async fn bulk_reduce_schema_binary<'s, 'b, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: Snippet<'s, Self::Item>,
            confidence_interpreter: &C,
        ) -> Vec<Section<Self::Item>>
        where
            Self::Item: 'b + Send + Sync,
            Snippet<'s, Self::Item>: Clone,
            Variation<Self::Item>: Clone,
            // TODO: Make V borrow snippet
            L: Fn(Snippet<'b, Self::Item>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, Self::Item>) -> Fut + Send + Sync,
            Fut: Future<Output = I> + Send,
            's: 'b;
    }

    #[async_trait]
    impl<T> AsyncReduceHalves for Phrase<T>
    where
        T: Clone + Debug,
    {
        async fn reduce_halves<L, C, FutBool, Fut>(
            &self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            T: Send + Sync,
            L: Fn(Snippet<'_, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            C: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
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
    impl<T, U> AsyncReduceHalvesBulk<U> for Phrase<T>
    where
        T: Debug,
        U: IntoIterator<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_halves<'s, 'b, L, C, FutBool, Fut>(
            &'s self,
            size_checker: L,
            confidence_interpreter: C,
        ) -> Self
        where
            T: Send + Sync,
            Snippet<'s, T>: Clone,
            Variation<T>: Clone,
            L: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
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

        async fn bulk_reduce_schema_binary<'s, 'b, L, C, FutBool, Fut>(
            size_checker: &L,
            phrase_snippet: Snippet<'s, T>,
            confidence_interpreter: &C,
        ) -> Vec<Section<T>>
        where
            Snippet<'s, T>: Clone,
            T: Send + Sync,
            Variation<T>: Clone,
            L: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            C: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
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
