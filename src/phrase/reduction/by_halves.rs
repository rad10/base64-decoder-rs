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
//! To assume all variables, we can interpret [`confidence_interpreter`] as
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

use std::fmt::{Debug, Display};

use itertools::Itertools;

use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Snippet, Variation};

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalves<U, V> {
    /// Defines the type of item that is collected from the phrase
    type Item;

    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self;

    /// A helper function to `reduce_halves`. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    fn reduce_schema_binary(
        size_checker: &U,
        phrase_snippet: Snippet<'_, Self::Item>,
        confidence_interpreter: &V,
    ) -> Vec<Section<Self::Item>>;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self;
}

/// Provides an interface to reduce an array like structure to through a
/// validator utilizing a recursive process
pub trait ReduceHalvesBulk<U, V> {
    /// Defines the type of item that is collected from the phrase
    type Item;

    /// This schema reduction strategy takes the reverse of pairs. While
    /// pairs will start with the smallest group, this function will work
    /// backwards and reduce using the largest valid permutation available.
    /// This largest available permutation will depend on `permutation_limit`
    /// to decide the size of the section.
    fn bulk_reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self;

    /// A helper function to `reduce_halves`. Takes a binary search
    /// approach by cutting the sections in half and running the validation
    /// check on all values in that section if the permutation value is low
    /// enough. Otherwise, cut it in half and try again.
    fn bulk_reduce_schema_binary(
        size_checker: &U,
        phrase_snippet: Snippet<'_, Self::Item>,
        confidence_interpreter: &V,
    ) -> Vec<Section<Self::Item>>;

    /// Reduces the phrase until the reduction function cannot reduce it
    /// anymore.
    fn bulk_halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self;
}

impl<T, U, V> ReduceHalves<U, V> for Phrase<T>
where
    T: Debug,
    U: Fn(&Snippet<'_, T>) -> bool,
    V: Fn(&Variation<T>) -> f64,
    Variation<T>: Clone + Display,
{
    type Item = T;

    fn reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
        Self::new(Self::reduce_schema_binary(
            &size_checker,
            Snippet::from(self.sections.as_slice()),
            &confidence_interpreter,
        ))
    }

    fn reduce_schema_binary(
        size_checker: &U,
        phrase_snippet: Snippet<'_, Self::Item>,
        confidence_interpreter: &V,
    ) -> Vec<Section<Self::Item>> {
        // Leave early if section is empty or just one
        if phrase_snippet.len_sections() < 2 {
            phrase_snippet.sections.to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if size_checker(&phrase_snippet) {
            vec![
                phrase_snippet
                    .iter_var()
                    .map(|line| (confidence_interpreter(&line), line))
                    .inspect(|(confidence, line)| {
                        log::debug!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only square root of permitted permutations to allow rerunning the reduction
                    .k_largest_relaxed_by_key(
                        phrase_snippet.permutations().sqrt().floor() as usize,
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
                    Self::reduce_schema_binary(
                        size_checker,
                        Snippet::new(c),
                        confidence_interpreter,
                    )
                })
                .collect()
        }
    }

    fn halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self
    where
        Variation<T>: Display,
    {
        // Begin by flattening single variation items
        let mut new_phrase = self.flatten_sections();
        // Set the last permutation to last size. Stop if permutation doesnt
        // shrink in any given instance
        let mut last_size = usize::MAX;
        while last_size > new_phrase.len_sections() {
            last_size = new_phrase.len_sections();
            new_phrase = new_phrase.reduce_halves(&size_checker, &confidence_interpreter);
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        new_phrase.convert_to_string(),
                        new_phrase.permutations(),
                    );
                }
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        new_phrase.sections,
                        new_phrase.len_sections(),
                        new_phrase.num_of_references(),
                        new_phrase.permutations(),
                    );
                }
                _ => (),
            };
        }
        new_phrase
    }
}

impl<T, U, V, W> ReduceHalvesBulk<U, V> for Phrase<T>
where
    T: Debug,
    U: Fn(&Snippet<'_, T>) -> bool,
    V: Fn(&Snippet<'_, T>) -> W,
    W: Iterator<Item = (f64, Variation<T>)>,
    Variation<T>: Clone + Display,
{
    type Item = T;

    fn bulk_reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
        Self::new(Self::bulk_reduce_schema_binary(
            &size_checker,
            Snippet::from(self.sections.as_slice()),
            &confidence_interpreter,
        ))
    }

    fn bulk_reduce_schema_binary(
        size_checker: &U,
        phrase_snippet: Snippet<'_, Self::Item>,
        confidence_interpreter: &V,
    ) -> Vec<Section<Self::Item>>
    where
        Variation<T>: Clone,
    {
        // Leave early if section is empty or just one
        if phrase_snippet.len_sections() < 2 {
            phrase_snippet.sections.to_vec()
        }
        // If the permutations within the sections is less than limit, then start crunching through them
        else if size_checker(&phrase_snippet) {
            vec![
                confidence_interpreter(&phrase_snippet)
                    .inspect(|(confidence, line)| {
                        log::debug!("confidence, string: {confidence}, {line:?}")
                    })
                    // Keeping only square root of permitted permutations to allow rerunning the reduction
                    .k_largest_relaxed_by_key(
                        phrase_snippet.permutations().sqrt().floor() as usize,
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
                        size_checker,
                        Snippet::new(c),
                        confidence_interpreter,
                    )
                })
                .collect()
        }
    }

    fn bulk_halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self
    where
        T: Debug,
        Variation<T>: Display,
    {
        // Begin by flattening single variation items
        let mut new_phrase = self.flatten_sections();
        // Set the last permutation to last size. Stop if permutation doesnt
        // shrink in any given instance
        let mut last_size = usize::MAX;
        while last_size > new_phrase.len_sections() {
            last_size = new_phrase.len_sections();
            new_phrase = new_phrase.bulk_reduce_halves(&size_checker, &confidence_interpreter);
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        new_phrase.convert_to_string(),
                        new_phrase.permutations()
                    );
                }
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        new_phrase.sections,
                        new_phrase.len_sections(),
                        new_phrase.num_of_references(),
                        new_phrase.permutations()
                    );
                }
                _ => (),
            };
        }
        new_phrase
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

    use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReduceHalves<U, V> {
        /// Defines the type of item that is collected from the phrase
        type Item;

        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self;

        /// A helper function to `reduce_halves`. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        fn reduce_schema_binary(
            size_checker: &U,
            sections: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> Vec<Section<Self::Item>>;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes the rayon library to validate pairs in parallel
    pub trait ParReduceHalvesBulk<U, V> {
        /// Defines the type of item that is collected from the phrase
        type Item;

        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn bulk_reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self;

        /// A helper function to `reduce_halves`. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        fn bulk_reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> Vec<Section<Self::Item>>;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn bulk_halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self;
    }

    impl<T, U, V> ParReduceHalves<U, V> for Phrase<T>
    where
        T: Debug + Send + Sync,
        U: Fn(&Snippet<'_, T>) -> bool + Sync,
        V: Fn(&Variation<T>) -> f64 + Sync,
        Variation<T>: Clone + Display,
    {
        type Item = T;

        fn reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
            Self::new(Self::reduce_schema_binary(
                &size_checker,
                Snippet::from(self.sections.as_slice()),
                &confidence_interpreter,
            ))
        }

        fn reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> Vec<Section<Self::Item>> {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(&phrase_snippet) {
                vec![
                    phrase_snippet
                        .iter_var()
                        .par_bridge()
                        .map(|line| (confidence_interpreter(&line), line))
                        .inspect(|(confidence, line)| {
                            log::debug!("confidence, string: {confidence}, {line:?}")
                        })
                        // Collecting here to drop to a regular iterator
                        .collect::<Vec<(f64, Variation<T>)>>()
                        .into_iter()
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            phrase_snippet.permutations().sqrt().floor() as usize,
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
                        Self::reduce_schema_binary(
                            size_checker,
                            Snippet::new(c),
                            confidence_interpreter,
                        )
                    })
                    .collect()
            }
        }

        fn halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self
        where
            Variation<T>: Display,
        {
            // Begin by flattening single variation items
            let mut new_phrase = self.flatten_sections();
            // Set the last permutation to last size. Stop if permutation doesnt
            // shrink in any given instance
            let mut last_size = usize::MAX;
            while last_size > new_phrase.len_sections() {
                last_size = new_phrase.len_sections();
                new_phrase = new_phrase.reduce_halves(&size_checker, &confidence_interpreter);
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            new_phrase.convert_to_string(),
                            new_phrase.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            new_phrase.sections,
                            new_phrase.len_sections(),
                            new_phrase.num_of_references(),
                            new_phrase.permutations()
                        );
                    }
                    _ => (),
                };
            }
            new_phrase
        }
    }

    impl<T, U, V, W> ParReduceHalvesBulk<U, V> for Phrase<T>
    where
        T: Debug + Send + Sync,
        U: Fn(&Snippet<'_, T>) -> bool + Sync,
        V: Fn(&Snippet<'_, T>) -> W + Sync,
        W: Iterator<Item = (f64, Variation<T>)>,
        Variation<T>: Clone + Display,
    {
        type Item = T;

        fn bulk_reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
            Self::new(Self::bulk_reduce_schema_binary(
                &size_checker,
                Snippet::from(self.sections.as_slice()),
                &confidence_interpreter,
            ))
        }

        fn bulk_reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> Vec<Section<Self::Item>>
        where
            Variation<T>: Clone,
        {
            // Leave early if section is empty or just one
            if phrase_snippet.len_sections() < 2 {
                phrase_snippet.sections.to_vec()
            }
            // If the permutations within the sections is less than limit, then start crunching through them
            else if size_checker(&phrase_snippet) {
                vec![
                    confidence_interpreter(&phrase_snippet)
                        .inspect(|(confidence, line)| {
                            log::debug!("confidence, string: {confidence}, {line:?}")
                        })
                        // Keeping only square root of permitted permutations to allow rerunning the reduction
                        .k_largest_relaxed_by_key(
                            phrase_snippet.permutations().sqrt().floor() as usize,
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
                            size_checker,
                            Snippet::new(c),
                            confidence_interpreter,
                        )
                    })
                    .collect()
            }
        }

        fn bulk_halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self
        where
            T: Debug,
            Variation<T>: Display,
        {
            // Begin by flattening single variation items
            let mut new_phrase = self.flatten_sections();
            // Set the last permutation to last size. Stop if permutation doesnt
            // shrink in any given instance
            let mut last_size = usize::MAX;
            while last_size > new_phrase.len_sections() {
                last_size = new_phrase.len_sections();
                new_phrase = new_phrase.bulk_reduce_halves(&size_checker, &confidence_interpreter);
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            new_phrase.convert_to_string(),
                            new_phrase.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            new_phrase.sections,
                            new_phrase.len_sections(),
                            new_phrase.num_of_references(),
                            new_phrase.permutations()
                        );
                    }
                    _ => (),
                };
            }
            new_phrase
        }
    }
}

/// Provides and implements the reduction trait using the asynchronous calls for smoother processing
#[cfg(feature = "async")]
pub mod r#async {
    use std::fmt::{Debug, Display};

    use futures::{Stream, StreamExt, stream};
    use itertools::Itertools;

    use crate::phrase::schema::{ConvertString, Permutation, Phrase, Section, Snippet, Variation};

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    pub trait AsyncReduceHalves<U, V> {
        /// Defines the type of item that is collected from the phrase
        type Item;

        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn reduce_halves(
            &self,
            size_checker: U,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send;

        /// A helper function to `reduce_halves`. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        fn reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> impl Future<Output = Vec<Section<Self::Item>>> + Send;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn halves_to_end(
            &self,
            size_checker: U,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send;
    }

    /// Provides an interface to reduce an array like structure to through a
    /// validator utilizing a recursive process
    ///
    /// Utilizes asynchronous tasks for asynchronous functions
    pub trait AsyncReduceHalvesBulk<U, V> {
        /// Defines the type of item that is collected from the phrase
        type Item;

        /// This schema reduction strategy takes the reverse of pairs. While
        /// pairs will start with the smallest group, this function will work
        /// backwards and reduce using the largest valid permutation available.
        /// This largest available permutation will depend on `permutation_limit`
        /// to decide the size of the section.
        fn bulk_reduce_halves(
            &self,
            size_checker: U,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send;

        /// A helper function to `reduce_halves`. Takes a binary search
        /// approach by cutting the sections in half and running the validation
        /// check on all values in that section if the permutation value is low
        /// enough. Otherwise, cut it in half and try again.
        fn bulk_reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> impl Future<Output = Vec<Section<Self::Item>>> + Send;

        /// Reduces the phrase until the reduction function cannot reduce it
        /// anymore.
        fn bulk_halves_to_end(
            &self,
            size_checker: U,
            confidence_interpreter: V,
        ) -> impl Future<Output = Self> + Send;
    }

    impl<T, U, V, FnFutBool, FnFut> AsyncReduceHalves<U, V> for Phrase<T>
    where
        T: Debug + Send + Sync,
        U: Fn(&Snippet<'_, T>) -> FnFutBool + Send + Sync,
        FnFutBool: Future<Output = bool> + Send,
        V: Fn(&'_ Variation<T>) -> FnFut + Send + Sync,
        FnFut: Future<Output = f64> + Send,
        Variation<T>: Clone + Display,
    {
        type Item = T;

        async fn reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
            Self::new(
                Self::reduce_schema_binary(
                    &size_checker,
                    Snippet::from(self.sections.as_slice()),
                    &confidence_interpreter,
                )
                .await,
            )
        }

        fn reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> impl Future<Output = Vec<Section<Self::Item>>> + Send {
            async move {
                // Leave early if section is empty or just one
                if phrase_snippet.len_sections() < 2 {
                    phrase_snippet.sections.to_vec()
                }
                // If the permutations within the sections is less than limit, then start crunching through them
                else if size_checker(&phrase_snippet).await {
                    vec![
                        stream::iter(phrase_snippet.iter_var())
                            .then(async |line| (confidence_interpreter(&line).await, line))
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            .collect::<Vec<(f64, Variation<T>)>>()
                            .await
                            .into_iter()
                            // Keeping only square root of permitted permutations to allow rerunning the reduction
                            .k_largest_relaxed_by_key(
                                phrase_snippet.permutations().sqrt().floor() as usize,
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
                                Self::reduce_schema_binary(
                                    size_checker,
                                    Snippet::new(c.as_slice()),
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

        async fn halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self {
            // Begin by flattening single variation items
            let mut new_phrase = self.flatten_sections();
            // Set the last permutation to last size. Stop if permutation doesnt
            // shrink in any given instance
            let mut last_size = usize::MAX;
            while last_size > new_phrase.len_sections() {
                last_size = new_phrase.len_sections();
                new_phrase = new_phrase
                    .reduce_halves(&size_checker, &confidence_interpreter)
                    .await;
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            new_phrase.convert_to_string(),
                            new_phrase.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            new_phrase.sections,
                            new_phrase.len_sections(),
                            new_phrase.num_of_references(),
                            new_phrase.permutations()
                        );
                    }
                    _ => (),
                };
            }
            new_phrase
        }
    }

    impl<T, U, V, FnFutBool, FnFut> AsyncReduceHalvesBulk<U, V> for Phrase<T>
    where
        T: Debug + Send + Sync,
        U: Fn(&Snippet<'_, T>) -> FnFutBool + Send + Sync,
        FnFutBool: Future<Output = bool> + Send,
        V: Fn(&Snippet<'_, T>) -> FnFut + Send + Sync,
        FnFut: Stream<Item = (f64, Variation<T>)> + Send,
        Variation<T>: Clone + Display,
    {
        type Item = T;

        async fn bulk_reduce_halves(&self, size_checker: U, confidence_interpreter: V) -> Self {
            Self::new(
                Self::bulk_reduce_schema_binary(
                    &size_checker,
                    Snippet::from(self.sections.as_slice()),
                    &confidence_interpreter,
                )
                .await,
            )
        }

        fn bulk_reduce_schema_binary(
            size_checker: &U,
            phrase_snippet: Snippet<'_, Self::Item>,
            confidence_interpreter: &V,
        ) -> impl Future<Output = Vec<Section<Self::Item>>> + Send {
            async move {
                // Leave early if section is empty or just one
                if phrase_snippet.len_sections() < 2 {
                    phrase_snippet.sections.to_vec()
                }
                // If the permutations within the sections is less than limit, then start crunching through them
                else if size_checker(&phrase_snippet).await {
                    vec![
                        confidence_interpreter(&phrase_snippet)
                            .inspect(|(confidence, line)| {
                                log::debug!("confidence, string: {confidence}, {line:?}")
                            })
                            .collect::<Vec<(f64, Variation<T>)>>()
                            .await
                            .into_iter()
                            // Keeping only square root of permitted permutations to allow rerunning the reduction
                            .k_largest_relaxed_by_key(
                                phrase_snippet.permutations().sqrt().floor() as usize,
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
                                    size_checker,
                                    Snippet::new(c.as_slice()),
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

        async fn bulk_halves_to_end(&self, size_checker: U, confidence_interpreter: V) -> Self {
            // Begin by flattening single variation items
            let mut new_phrase = self.flatten_sections();
            // Set the last permutation to last size. Stop if permutation doesnt
            // shrink in any given instance
            let mut last_size = usize::MAX;
            while last_size > new_phrase.len_sections() {
                last_size = new_phrase.len_sections();
                new_phrase = new_phrase
                    .bulk_reduce_halves(&size_checker, &confidence_interpreter)
                    .await;
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            new_phrase.convert_to_string(),
                            new_phrase.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                            new_phrase.sections,
                            new_phrase.len_sections(),
                            new_phrase.num_of_references(),
                            new_phrase.permutations()
                        );
                    }
                    _ => (),
                };
            }
            new_phrase
        }
    }
}
