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

use std::fmt::{Debug, Display};

use itertools::Itertools;

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<'a, T> {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn reduce_halves<'b, V, W>(
        &'a self,
        size_checker: &'b V,
        confidence_interpreter: &'b W,
    ) -> Self
    where
        Self: ReduceHalvesBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(&Variation<T>) -> f64 + 'b,
        'a: 'b;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn halves_to_end<'b, V, W>(
        &'a self,
        size_checker: &'b V,
        confidence_interpreter: &'b W,
    ) -> Self
    where
        Self: ReduceHalvesBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(&Variation<T>) -> f64 + 'b,
        'a: 'b;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<'a, T, U: ?Sized> {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn bulk_reduce_halves<V, W>(&'a self, size_checker: V, confidence_interpreter: W) -> Self
    where
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<V, W>(
        size_checker: V,
        phrase_snippet: Snippet<'a, T>,
        confidence_interpreter: W,
    ) -> Vec<Section<T>>
    where
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn bulk_halves_to_end<V, W>(
        &'a self,
        recursive_val: Option<usize>,
        size_checker: V,
        confidence_interpreter: W,
    ) -> Self
    where
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U;
}

impl<'a, T> ReduceHalves<'a, T> for Phrase<T> {
    fn reduce_halves<'b, V, W>(&'a self, size_checker: &'b V, confidence_interpreter: &'b W) -> Self
    where
        Self: ReduceHalvesBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(&Variation<T>) -> f64 + 'b,
        'a: 'b,
    {
        self.bulk_reduce_halves(size_checker, |snip| {
            Box::new(
                snip.into_iter_var()
                    .map(|line| (confidence_interpreter(&line), line)),
            )
        })
    }

    fn halves_to_end<'b, V, W>(&'a self, size_checker: &'b V, confidence_interpreter: &'b W) -> Self
    where
        Self: ReduceHalvesBulk<'a, T, Box<dyn Iterator<Item = (f64, Variation<T>)> + 'b>>,
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(&Variation<T>) -> f64 + 'b,
        'a: 'b,
    {
        self.bulk_halves_to_end(None, size_checker, |snip| {
            Box::new(
                snip.into_iter_var()
                    .map(|line| (confidence_interpreter(&line), line)),
            )
        })
    }
}

impl<'a, T, U> ReduceHalvesBulk<'a, T, U> for Phrase<T>
where
    T: Clone + Debug,
    U: Iterator<Item = (f64, Variation<T>)>,
    Variation<T>: Display,
{
    fn bulk_reduce_halves<V, W>(&'a self, size_checker: V, confidence_interpreter: W) -> Self
    where
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U,
    {
        Self::new(Self::bulk_reduce_schema_binary(
            size_checker,
            self.as_snippet(),
            confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary<V, W>(
        size_checker: V,
        phrase_snippet: Snippet<'a, T>,
        confidence_interpreter: W,
    ) -> Vec<Section<T>>
    where
        T: 'a + Clone + Debug,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U,
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
                    .inspect(|(confidence, line)| {
                        log::debug!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only square root of permitted permutations to allow rerunning the reduction
                    .k_largest_relaxed_by_key(
                        usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                        |(confidence, _)| (confidence * 100_000_f64) as usize,
                    )
                    .inspect(|(confidence, line)| {
                        log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                    })
                    .map(|(_, line)| line)
                    .collect::<Section<T>>(),
            ]
        }
        // If permutations are still too big, split it again
        else {
            phrase_snippet
                .sections
                .chunks(phrase_snippet.len_sections() / 2)
                .flat_map(|c| {
                    Self::bulk_reduce_schema_binary(
                        &size_checker,
                        Snippet::new(c),
                        &confidence_interpreter,
                    )
                })
                .collect()
        }
    }

    fn bulk_halves_to_end<V, W>(
        &'a self,
        recursive_val: Option<usize>,
        size_checker: V,
        confidence_interpreter: W,
    ) -> Self
    where
        T: 'a + Clone + Debug,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: Fn(Snippet<'a, T>) -> U,
        Variation<T>: Display,
    {
        if let Some(last_size) = recursive_val {
            // Currently in recursive loop
            // Collecting section len to determine if ending or not
            if last_size <= self.len_sections() {
                self.clone()
            } else {
                self.bulk_halves_to_end(
                    Some(self.len_sections()),
                    size_checker,
                    confidence_interpreter,
                )
            }
        } else {
            // Setting up initial recursion
            self.bulk_halves_to_end(Some(usize::MAX), size_checker, confidence_interpreter)
        }
    }
}

/// Provides and implements the reduction trait using the rayon library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::fmt::Debug;

    use itertools::Itertools;
    use rayon::{iter::ParallelIterator, slice::ParallelSlice};

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves<'a, T> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn reduce_halves<'b, V, W>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'b W,
        ) -> Self
        where
            Self: ParReduceHalvesBulk<
                    'a,
                    T,
                    Box<dyn Iterator<Item = (f64, Variation<T>)> + Send + Sync + 'b>,
                >,
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + 'b + Send + Sync,
            'a: 'b;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn halves_to_end<'b, V, W>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'b W,
        ) -> Self
        where
            Self: ParReduceHalvesBulk<
                    'a,
                    T,
                    Box<dyn Iterator<Item = (f64, Variation<T>)> + Send + Sync + 'b>,
                >,
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + 'b + Send + Sync,
            'a: 'b;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<'a, T, U: ?Sized> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves<V, W>(&'a self, size_checker: V, confidence_interpreter: W) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<V, W>(
            size_checker: V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: W,
        ) -> Vec<Section<T>>
        where
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn bulk_halves_to_end<V, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync;
    }

    impl<'a, T> ParReduceHalves<'a, T> for Phrase<T> {
        fn reduce_halves<'b, V, W>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'b W,
        ) -> Self
        where
            Self: ParReduceHalvesBulk<
                    'a,
                    T,
                    Box<dyn Iterator<Item = (f64, Variation<T>)> + Send + Sync + 'b>,
                >,
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + 'b + Send + Sync,
            'a: 'b,
        {
            self.bulk_reduce_halves(size_checker, |snip| {
                Box::new(
                    snip.into_iter_var()
                        .map(|line| (confidence_interpreter(&line), line)),
                )
            })
        }

        fn halves_to_end<'b, V, W>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'b W,
        ) -> Self
        where
            Self: ParReduceHalvesBulk<
                    'a,
                    T,
                    Box<dyn Iterator<Item = (f64, Variation<T>)> + Send + Sync + 'b>,
                >,
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + 'b + Send + Sync,
            'a: 'b,
        {
            self.bulk_halves_to_end(None, size_checker, |snip| {
                Box::new(
                    snip.into_iter_var()
                        .map(|line| (confidence_interpreter(&line), line)),
                )
            })
        }
    }

    impl<'a, T, U> ParReduceHalvesBulk<'a, T, U> for Phrase<T>
    where
        T: Clone + Debug,
        U: Iterator<Item = (f64, Variation<T>)>,
    {
        fn bulk_reduce_halves<V, W>(&'a self, size_checker: V, confidence_interpreter: W) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync,
        {
            Self::new(Self::bulk_reduce_schema_binary(
                size_checker,
                self.as_snippet(),
                confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<V, W>(
            size_checker: V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: W,
        ) -> Vec<Section<T>>
        where
            T: 'a + Clone + Debug + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync,
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
                        .inspect(|(confidence, line)| {
                            log::debug!("confidence, string: {confidence}, {line:?}")
                        })
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                            |(confidence, _)| (confidence * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                        })
                        .map(|(_, line)| line)
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                phrase_snippet
                    .sections
                    .par_chunks(phrase_snippet.len_sections() / 2)
                    .flat_map(|c| {
                        Self::bulk_reduce_schema_binary(
                            &size_checker,
                            Snippet::new(c),
                            &confidence_interpreter,
                        )
                    })
                    .collect()
            }
        }

        fn bulk_halves_to_end<V, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync,
        {
            if let Some(last_size) = recursive_val {
                // Currently in recursive loop
                // Collecting section len to determine if ending or not
                if last_size <= self.len_sections() {
                    self.clone()
                } else {
                    self.bulk_halves_to_end(
                        Some(self.len_sections()),
                        size_checker,
                        confidence_interpreter,
                    )
                }
            } else {
                // Setting up initial recursion
                self.bulk_halves_to_end(Some(usize::MAX), size_checker, confidence_interpreter)
            }
        }
    }
}

/// Provides and implements the reduction trait using the asynchronous calls for smoother processing
#[cfg(feature = "async")]
pub mod r#async {
    use std::{fmt::Debug, pin::Pin};

    use async_trait::async_trait;
    use futures::{Stream, StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, Section, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalves<'a, T> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        async fn reduce_halves<'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'static W,
        ) -> Self
        where
            Self:
                AsyncReduceHalvesBulk<'a, T, Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + 'b>>>,
            T: 'static + Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync + 'b,
            Fut: Future<Output = f64> + Send,
            'a: 'b;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        async fn halves_to_end<'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'static W,
        ) -> Self
        where
            Self:
                AsyncReduceHalvesBulk<'a, T, Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + 'b>>>,
            T: 'static + Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync + 'b,
            Fut: Future<Output = f64> + Send,
            'a: 'b;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<'a, T, U: ?Sized> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_reduce_halves<V, FutBool, W>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send;

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
        async fn bulk_reduce_schema_binary<V, FutBool, W>(
            size_checker: V,
            phrase_snippet: Phrase<T>,
            confidence_interpreter: W,
        ) -> Vec<Section<T>>
        where
            T: Send + Sync + 'async_trait,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_halves_to_end<V, FutBool, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send;
    }

    #[async_trait]
    impl<'a, T> AsyncReduceHalves<'a, T> for Phrase<T>
    where
        T: Clone + Debug,
    {
        async fn reduce_halves<'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'static W,
        ) -> Self
        where
            Self: AsyncReduceHalvesBulk<
                    'a,
                    T,
                    Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + Send + 'b>>,
                >,
            T: 'static + Clone + Debug + Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync + 'b,
            Fut: Future<Output = f64> + Send,
            'a: 'b,
        {
            Self::new(
                Self::bulk_reduce_schema_binary(size_checker, self.clone(), |snip: Phrase<T>| {
                    Box::pin(async move {
                        Box::pin(
                            stream::iter(snip.into_iter_var())
                                .then(async |line| (confidence_interpreter(&line).await, line)),
                        )
                            as Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + Send>>
                    })
                })
                .await,
            )
        }

        async fn halves_to_end<'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: &'b V,
            confidence_interpreter: &'static W,
        ) -> Self
        where
            Self: AsyncReduceHalvesBulk<
                    'a,
                    T,
                    Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + Send + 'b>>,
                >,
            T: 'static + Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync + 'b,
            Fut: Future<Output = f64> + Send,
            'a: 'b,
        {
            self.bulk_halves_to_end(None, size_checker, |snip: Phrase<T>| {
                Box::pin(async move {
                    Box::pin(
                        stream::iter(snip.into_iter_var())
                            .then(async |line| (confidence_interpreter(&line).await, line)),
                    ) as Pin<Box<dyn Stream<Item = (f64, Variation<T>)> + Send>>
                })
            })
            .await
        }
    }

    #[async_trait]
    impl<'a, T, U> AsyncReduceHalvesBulk<'a, T, U> for Phrase<T>
    where
        T: Clone + Debug,
        U: Stream<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_halves<V, FutBool, W>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send,
        {
            Self::new(
                Self::bulk_reduce_schema_binary(
                    &size_checker,
                    self.clone(),
                    &confidence_interpreter,
                )
                .await,
            )
        }

        async fn bulk_reduce_schema_binary<V, FutBool, W>(
            size_checker: V,
            phrase_snippet: Phrase<T>,
            confidence_interpreter: W,
        ) -> Vec<Section<T>>
        where
            T: Send + Sync + 'async_trait,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(&phrase_snippet).await {
                let snippet_permutation = phrase_snippet.permutations();
                vec![
                    confidence_interpreter(phrase_snippet)
                        .await
                        .inspect(|(confidence, line)| {
                            log::debug!("confidence, string: {confidence}, {line:?}")
                        })
                        .collect::<Vec<(f64, Variation<T>)>>()
                        .await
                        .into_iter()
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                            |(confidence, _)| (confidence * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::debug!("Accepted: confidence, string: {confidence}, {line:?}")
                        })
                        .map(|(_, line)| line)
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                stream::iter(phrase_snippet.sections.to_vec())
                    .chunks(phrase_snippet.len_sections() / 2)
                    .then(async |c| {
                        stream::iter(
                            Self::bulk_reduce_schema_binary(
                                &size_checker,
                                Phrase::new(c),
                                &confidence_interpreter,
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

        async fn bulk_halves_to_end<V, FutBool, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Clone + Send + Sync,
            V: Fn(&Phrase<T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Phrase<T>) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync,
            U: Send,
        {
            if let Some(last_size) = recursive_val {
                // Currently in recursive loop
                // Collecting section len to determine if ending or not
                if last_size <= self.len_sections() {
                    self.clone()
                } else {
                    self.bulk_halves_to_end(
                        Some(self.len_sections()),
                        size_checker,
                        confidence_interpreter,
                    )
                    .await
                }
            } else {
                // Setting up initial recursion
                self.bulk_halves_to_end(Some(usize::MAX), size_checker, confidence_interpreter)
                    .await
            }
        }
    }
}
