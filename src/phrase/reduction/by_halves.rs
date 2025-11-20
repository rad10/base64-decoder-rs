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

use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<T> {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn reduce_halves<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
    where
        Variation<T>: Clone,
        V: Fn(&Snippet<'_, T>) -> bool,
        W: Fn(&Variation<T>) -> f64;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn halves_to_end<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
    where
        Self: Clone,
        V: Fn(&Snippet<'_, T>) -> bool,
        W: Fn(&Variation<T>) -> f64;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<T, U: ?Sized> {
    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn bulk_reduce_halves<'a, 'b, V, W>(
        &'a self,
        size_checker: V,
        confidence_interpreter: W,
    ) -> Self
    where
        T: 'b,
        Variation<T>: Clone,
        V: Fn(&Snippet<'b, T>) -> bool,
        W: FnMut(Snippet<'b, T>) -> U,
        'a: 'b;

    /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    ///
    /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
    fn bulk_reduce_schema_binary<'a, 'b, V, W>(
        size_checker: &V,
        phrase_snippet: Snippet<'a, T>,
        confidence_interpreter: &mut W,
    ) -> Vec<Section<T>>
    where
        Variation<T>: Clone,
        V: Fn(&Snippet<'b, T>) -> bool,
        W: FnMut(Snippet<'b, T>) -> U,
        'a: 'b;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn bulk_halves_to_end<'a, V, W>(
        &'a self,
        recursive_val: Option<usize>,
        size_checker: V,
        confidence_interpreter: W,
    ) -> Self
    where
        Self: Clone,
        T: 'a,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: FnMut(Snippet<'a, T>) -> U;
}

impl<T> ReduceHalves<T> for Phrase<T>
where
    T: Debug,
{
    fn reduce_halves<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
    where
        T: Debug,
        Variation<T>: Clone,
        V: Fn(&Snippet<'_, T>) -> bool,
        W: Fn(&Variation<T>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_reduce_halves(size_checker, move |snip| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }

    fn halves_to_end<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
    where
        Self: Clone,
        V: Fn(&Snippet<'_, T>) -> bool,
        W: Fn(&Variation<T>) -> f64,
    {
        let conf_link = &confidence_interpreter;
        self.bulk_halves_to_end(None, size_checker, move |snip| {
            snip.into_iter_var()
                .map(move |line| (conf_link(&line), line))
        })
    }
}

impl<T, U> ReduceHalvesBulk<T, U> for Phrase<T>
where
    T: Debug,
    U: IntoIterator<Item = (f64, Variation<T>)>,
{
    fn bulk_reduce_halves<'a, 'b, V, W>(
        &'a self,
        size_checker: V,
        mut confidence_interpreter: W,
    ) -> Self
    where
        T: 'b,
        Variation<T>: Clone,
        V: Fn(&Snippet<'b, T>) -> bool,
        W: FnMut(Snippet<'b, T>) -> U,
        'a: 'b,
    {
        Self::new(Self::bulk_reduce_schema_binary(
            &size_checker,
            self.as_snippet(),
            &mut confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary<'a, 'b, V, W>(
        size_checker: &V,
        phrase_snippet: Snippet<'a, T>,
        confidence_interpreter: &mut W,
    ) -> Vec<Section<T>>
    where
        T: Debug,
        Variation<T>: Clone,
        V: Fn(&Snippet<'b, T>) -> bool,
        W: FnMut(Snippet<'b, T>) -> U,
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
                    .map(|(_, line)| line)
                    .collect::<Section<T>>(),
            ]
        }
        // If permutations are still too big, split it again
        else {
            phrase_snippet
                .sections
                .chunks(phrase_snippet.len_sections() / 2)
                .flat_map(move |c| {
                    Self::bulk_reduce_schema_binary(
                        size_checker,
                        Snippet::new(c),
                        confidence_interpreter,
                    )
                })
                .collect()
        }
    }

    fn bulk_halves_to_end<'a, V, W>(
        &'a self,
        recursive_val: Option<usize>,
        size_checker: V,
        confidence_interpreter: W,
    ) -> Self
    where
        Self: Clone,
        V: Fn(&Snippet<'a, T>) -> bool,
        W: FnMut(Snippet<'a, T>) -> U,
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
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalves<T> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn reduce_halves<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
        where
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + Send + Sync;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn halves_to_end<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
        where
            Self: Clone,
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + Send + Sync;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the [`rayon`] library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<T, U: ?Sized> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        fn bulk_reduce_halves<'a, 'b, V, W>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: 'b + Send + Sync,
            Variation<T>: Clone,
            V: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'b, T>) -> U + Send + Sync,
            U: Send + Sync,
            'a: 'b;

        /// A helper function to [`bulk_reduce_halves`]. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        ///
        /// [`bulk_reduce_halves`]: Self::bulk_reduce_halves
        fn bulk_reduce_schema_binary<'a, 'b, V, W>(
            size_checker: &V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: &W,
        ) -> Vec<Section<T>>
        where
            T: Send + Sync,
            Variation<T>: Clone,
            V: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'b, T>) -> U + Send + Sync,
            U: Send + Sync,
            'a: 'b;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn bulk_halves_to_end<'a, V, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
            T: 'a + Send + Sync,
            V: Fn(&Snippet<'a, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'a, T>) -> U + Send + Sync,
            U: Send + Sync;
    }

    impl<T> ParReduceHalves<T> for Phrase<T>
    where
        T: Debug,
        Variation<T>: Clone,
    {
        fn reduce_halves<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
        where
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + Send + Sync,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_reduce_halves(size_checker, move |snip| {
                snip.into_iter_var()
                    .par_bridge()
                    .map(move |line| (conf_link(&line), line))
                    .collect::<Vec<(f64, Variation<T>)>>()
            })
        }

        fn halves_to_end<V, W>(&self, size_checker: V, confidence_interpreter: W) -> Self
        where
            Self: Clone,
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> bool + Send + Sync,
            W: Fn(&Variation<T>) -> f64 + Send + Sync,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_halves_to_end(None, size_checker, move |snip| {
                snip.into_iter_var()
                    .map(move |line| (conf_link(&line), line))
            })
        }
    }

    impl<T, U> ParReduceHalvesBulk<T, U> for Phrase<T>
    where
        T: Debug,
        U: IntoIterator<Item = (f64, Variation<T>)>,
    {
        fn bulk_reduce_halves<'a, 'b, V, W>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: 'b + Send + Sync,
            Variation<T>: Clone,
            V: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'b, T>) -> U + Send + Sync,
            U: Send + Sync,
            'a: 'b,
        {
            Self::new(Self::bulk_reduce_schema_binary(
                &size_checker,
                self.as_snippet(),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary<'a, 'b, V, W>(
            size_checker: &V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: &W,
        ) -> Vec<Section<T>>
        where
            T: 'b + Send + Sync,
            Variation<T>: Clone,
            V: Fn(&Snippet<'b, T>) -> bool + Send + Sync,
            W: Fn(Snippet<'b, T>) -> U + Send + Sync,
            U: Send + Sync,
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
                            // You may think you will want to use `sqrt().ceil()` but we want floor, because this
                            // prevents returned values from failing `size_checker` on the 2nd run of this function.
                            usize::max(snippet_permutation.sqrt().floor() as usize, 1),
                            |(confidence, _)| (confidence * 100_000_f64) as usize,
                        )
                        .inspect(|(confidence, line)| {
                            log::trace!("Accepted: confidence, string: {confidence}, {line:?}")
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
                    .flat_map(move |c| {
                        Self::bulk_reduce_schema_binary(
                            size_checker,
                            Snippet::new(c),
                            confidence_interpreter,
                        )
                    })
                    .collect()
            }
        }

        fn bulk_halves_to_end<'a, V, W>(
            &'a self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
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
    use std::fmt::Debug;

    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalves<T> {
        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `size_checker`
        /// to decide the size of the section.
        async fn reduce_halves<V, FutBool, W, Fut>(
            &self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            Variation<T>: Clone,
            V: Fn(Snippet<'_, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        async fn halves_to_end<V, FutBool, W, Fut>(
            &self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    #[async_trait]
    pub trait AsyncReduceHalvesBulk<T, U: ?Sized> {
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
        async fn bulk_reduce_halves<'a, 'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
            Snippet<'a, T>: Clone,
            T: 'a + Send + Sync,
            Variation<T>: Clone,
            V: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            W: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
            U: Send,
            'a: 'b;

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
        async fn bulk_reduce_schema_binary<'a, 'b, V, FutBool, W, Fut>(
            size_checker: &V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: &W,
        ) -> Vec<Section<T>>
        where
            T: 'b + Send + Sync,
            Snippet<'a, T>: Clone,
            Variation<T>: Clone,
            // TODO: Make V borrow snippet
            V: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            W: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
            U: Send,
            'a: 'b;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        ///
        /// This uses [`Phrase`] instead of [`Snippet`] due to stream taking
        /// ownership of the phrases data instead of borrowing it.
        ///
        /// [`Phrase`]: crate::phrase::schema::Phrase
        /// [`Snippet`]: crate::phrase::schema::Snippet
        async fn bulk_halves_to_end<'a, V, FutBool, W, Fut>(
            &self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
            T: Send + Sync + 'a,
            V: Fn(&'a Snippet<'a, T>) -> FutBool + Send + Sync + 'a,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Snippet<'a, T>) -> Fut + Send + Sync + 'a,
            Fut: Future<Output = U> + Send,
            U: Send;
    }

    #[async_trait]
    impl<T> AsyncReduceHalves<T> for Phrase<T>
    where
        T: Clone + Debug,
    {
        async fn reduce_halves<V, FutBool, W, Fut>(
            &self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            V: Fn(Snippet<'_, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync,
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

        async fn halves_to_end<V, FutBool, W, Fut>(
            &self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            V: Fn(&Snippet<'_, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send,
            W: Fn(&Variation<T>) -> Fut + Send + Sync,
            Fut: Future<Output = f64> + Send,
        {
            let conf_link = &confidence_interpreter;
            self.bulk_halves_to_end(None, size_checker, async |snip: Snippet<'_, T>| {
                stream::iter(snip.iter_var())
                    .then(async move |line| (conf_link(&line).await, line))
                    .collect::<Vec<(f64, Variation<T>)>>()
                    .await
            })
            .await
        }
    }

    #[async_trait]
    impl<T, U> AsyncReduceHalvesBulk<T, U> for Phrase<T>
    where
        T: Debug,
        U: IntoIterator<Item = (f64, Variation<T>)>,
    {
        async fn bulk_reduce_halves<'a, 'b, V, FutBool, W, Fut>(
            &'a self,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            T: Send + Sync,
            Snippet<'a, T>: Clone,
            Variation<T>: Clone,
            V: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            W: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
            U: Send,
            'a: 'b,
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

        async fn bulk_reduce_schema_binary<'a, 'b, V, FutBool, W, Fut>(
            size_checker: &V,
            phrase_snippet: Snippet<'a, T>,
            confidence_interpreter: &W,
        ) -> Vec<Section<T>>
        where
            Snippet<'a, T>: Clone,
            T: Send + Sync,
            Variation<T>: Clone,
            V: Fn(Snippet<'b, T>) -> FutBool + Send + Sync,
            FutBool: Future<Output = bool> + Send + 'b,
            W: Fn(Snippet<'b, T>) -> Fut + Send + Sync,
            Fut: Future<Output = U> + Send,
            U: Send,
            'a: 'b,
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
                        .map(|(_, line)| line)
                        .collect::<Section<T>>(),
                ]
            }
            // If permutations are still too big, split it again
            else {
                let phrase_len = phrase_snippet.len_sections();
                stream::iter(phrase_snippet.sections.chunks(phrase_len / 2))
                    .then(async |c| {
                        stream::iter(
                            Self::bulk_reduce_schema_binary(
                                size_checker,
                                Snippet::new(c),
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

        async fn bulk_halves_to_end<'a, V, FutBool, W, Fut>(
            &self,
            recursive_val: Option<usize>,
            size_checker: V,
            confidence_interpreter: W,
        ) -> Self
        where
            Self: Clone,
            T: Send + Sync + 'a,
            V: Fn(&'a Snippet<'a, T>) -> FutBool + Send + Sync + 'a,
            FutBool: Future<Output = bool> + Send,
            W: Fn(Snippet<'a, T>) -> Fut + Send + Sync + 'a,
            Fut: Future<Output = U> + Send,
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
