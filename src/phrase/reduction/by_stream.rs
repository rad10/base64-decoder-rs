//! This reduction method attempts to reduce a snippet the same way a person
//! attempts to reduce a snippet by hand... by reading from left to right. This
//! reduces a snippet by starting from the beginning and adds sections to the
//! total and uses the best total to determine the best variation from each new
//! section.
//!
//! Due to the tactics in this method, this cannot occur in parallelism due to
//! the each section depending on the previous result.

use std::borrow::Borrow;

use itertools::Itertools;

use crate::phrase::schema::{BorrowedSnippet, Permutation, Phrase, SnippetExt, Variation};

/// Provides an interface to reduce a snippet by reading in data one section at
/// a time
pub trait ReduceReading<'s, B: SnippetExt<Item = Self::Item>>: SnippetExt + Sized {
    /// Reduces a phrases by validating a phrase from left to right.
    ///
    /// This is slower than other methods but comes with the benefit of being
    /// more accurate
    fn reduce_reading<L, C>(&'s self, base_determination: L, confidence_interpreter: C) -> Self
    where
        L: Fn(&B) -> bool,
        C: FnMut(&Variation<Self::Item>) -> f64,
        Self: Borrow<B>;
}

impl<T> ReduceReading<'_, Phrase<T>> for Phrase<T>
where
    Variation<T>: Clone,
{
    fn reduce_reading<L, C>(&'_ self, base_determination: L, mut confidence_interpreter: C) -> Self
    where
        L: Fn(&Phrase<T>) -> bool,
        C: FnMut(&Variation<Self::Item>) -> f64,
        Self: Borrow<Phrase<T>>,
    {
        let mut section_stream = Borrow::<BorrowedSnippet<T>>::borrow(self).iter();
        let mut final_buffer = Self::new([]);

        // Start by collecting a reasonable base to build the rest of the
        // string on
        while !base_determination(final_buffer.borrow())
            && let Some(section) = section_stream.next()
        {
            final_buffer.sections.push(section.to_vec());
        }

        // Reducing base to its reasonable values
        if final_buffer.len_sections() > 1 {
            let max_initial_buffer = final_buffer.len_sections();
            final_buffer.sections = vec![
                final_buffer
                    .sections
                    .into_iter_var()
                    .map(|v| (confidence_interpreter(&v), v))
                    .k_largest_relaxed_by_key(max_initial_buffer, |(confidence, _)| {
                        (confidence * 100_000_f64) as usize
                    })
                    .map(move |(_, line)| line)
                    .collect(),
            ];
        }

        // Taking the rest of the iterator and folding each section into this snippet
        section_stream.fold(final_buffer, |mut p_init, section| {
            // Add the section to our final phrase
            p_init.sections.push(section.to_vec());

            // Run our function again
            let p_init_permutation = p_init.permutations() / 2_f64;
            p_init.sections = vec![
                p_init
                    .sections
                    .into_iter_var()
                    .map(|v| (confidence_interpreter(&v), v))
                    .k_largest_relaxed_by_key(
                        usize::max(p_init_permutation as usize, 1),
                        |(confidence, _)| (confidence * 100_000_f64) as usize,
                    )
                    .map(move |(_, line)| line)
                    .collect(),
            ];
            p_init
        })
    }
}

/// Provides and implements the [`ReduceReading`] trait for use in futures
#[cfg(feature = "async")]
pub mod r#async {
    use std::{borrow::Borrow, sync::Arc};

    use async_trait::async_trait;
    use futures::StreamExt;
    use itertools::Itertools;

    use crate::phrase::schema::{Permutation, Phrase, SnippetExt, ThreadedSnippetExt, Variation};

    #[async_trait]
    /// Provides an interface to reduce a snippet by reading in data one section at
    /// a time in a streaming format
    pub trait AsyncReduceReadings<'s, B: ThreadedSnippetExt<Item = Self::Item>>:
        ThreadedSnippetExt + Sized
    where
        Arc<Self::Item>: Sync,
    {
        /// Reduces a phrases by validating a phrase from left to right. in an
        /// asynchronous stream
        ///
        /// This is slower than other methods but comes with the benefit of being
        /// more accurate
        async fn reduce_reading<L, C, LFut, CFut>(
            &'s self,
            base_determination: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(&B) -> LFut + Send + Sync,
            LFut: Future<Output = bool> + Send,
            C: Fn(&Variation<Self::Item>) -> CFut + Send + Sync,
            CFut: Future<Output = f64> + Send;
    }

    #[async_trait]
    impl<'s, T> AsyncReduceReadings<'s, Phrase<T>> for Phrase<T>
    where
        Self: Clone,
        Arc<T>: Send + Sync,
        Variation<T>: Clone,
    {
        async fn reduce_reading<L, C, LFut, CFut>(
            &'s self,
            base_determination: L,
            confidence_interpreter: C,
        ) -> Self
        where
            L: Fn(&Phrase<T>) -> LFut + Send + Sync,
            LFut: Future<Output = bool> + Send,
            C: Fn(&Variation<T>) -> CFut + Send + Sync,
            CFut: Future<Output = f64> + Send,
        {
            let mut section_stream = futures::stream::iter(self.sections.iter());
            let mut final_buffer = Self::new([]);

            // Start by collecting a reasonable base to build the rest of the
            // string on
            while !base_determination(final_buffer.borrow()).await
                && let Some(section) = section_stream.next().await
            {
                final_buffer.sections.push(section.to_vec());
            }

            // Reducing base to its reasonable values
            if final_buffer.len_sections() > 1 {
                let max_initial_buffer = final_buffer.len_sections();
                final_buffer.sections = vec![
                    futures::stream::iter(final_buffer.sections.into_iter_var())
                        .then(async |v| (confidence_interpreter(&v).await, v))
                        .collect::<Vec<_>>()
                        .await
                        .into_iter()
                        .k_largest_relaxed_by_key(max_initial_buffer, |(confidence, _)| {
                            (confidence * 100_000_f64) as usize
                        })
                        .map(move |(_, line)| line)
                        .collect(),
                ];
            }

            // Taking the rest of the iterator and folding each section into this snippet
            section_stream
                .fold(final_buffer, async |mut p_init, section| {
                    // Add the section to our final phrase
                    p_init.sections.push(section.to_vec());

                    // Run our function again
                    let p_init_permutation = p_init.permutations() / 2_f64;
                    p_init.sections = vec![
                        futures::stream::iter(p_init.sections.into_iter_var())
                            .then(async |v| (confidence_interpreter(&v).await, v))
                            .collect::<Vec<_>>()
                            .await
                            .into_iter()
                            .k_largest_relaxed_by_key(
                                usize::max(p_init_permutation as usize, 1),
                                |(confidence, _)| (confidence * 100_000_f64) as usize,
                            )
                            .map(move |(_, line)| line)
                            .collect(),
                    ];
                    p_init
                })
                .await
        }
    }
}
