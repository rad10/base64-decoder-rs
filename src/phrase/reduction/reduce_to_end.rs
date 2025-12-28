use std::fmt::Debug;
#[cfg(feature = "async")]
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::phrase::schema::snippet::ThreadedSnippetExt;
use crate::phrase::schema::snippet::{ConvertString, Permutation, SnippetExt};
#[cfg(feature = "async")]
use async_trait::async_trait;

/// Provides the ability for a phrase to continuously run a function until it
/// can no longer make the phrase smaller
pub trait ReduceToEnd: SnippetExt + Sized {
    /// Consumes the current phrase to produce a new phrase reduced as much as
    /// it possibly can
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::{reduction::{by_pairs::ReducePairs, reduce_to_end::ReduceToEnd}, schema::{snippet::{BorrowedSnippet, Phrase, SnippetExt}, variation::Variation}, validation::validate_with_whatlang};
    /// use std::sync::Arc;
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
    ///     vec![vec!["Hel", "lo ", "WoX", "ld!", "Thi", "is ", "is ", "my ", "str", "ing", "!"]],
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
    /// let reduced_result = phrase_string.reduce_to_end(move |base| base.reduce_pairs(2, validate_with_whatlang));
    /// assert!(reduced_result == reduced_phrase,
    /// "Reduced function {:?} did not match reduced result {:?}", reduced_result, reduced_phrase);
    /// ```
    fn reduce_to_end<F>(self, reduction_function: F) -> Self
    where
        F: FnMut(Self) -> Self;

    /// Reduces a phrase as much as humanly possible by mutating itself.
    ///
    /// This can be more memory efficient than using [`ReduceToEnd::reduce_to_end`]
    /// as it allows mutating its own contents directly
    fn reduce_to_end_mut<F>(self, reduction_function: F) -> Self
    where
        F: FnMut(&mut Self);
}

#[cfg(feature = "async")]
#[async_trait]
/// Provides the ability for a phrase to continuously run a function until it
/// can no longer make the phrase smaller
pub trait AsyncReduceToEnd: ThreadedSnippetExt + Sized
where
    Arc<Self::Item>: Sync,
{
    /// Consumes the current phrase to produce a new phrase reduced as much as
    /// it possibly can
    ///
    /// ```rust
    /// # futures::executor::block_on(async {
    /// use base64_bruteforcer_rs::phrase::{reduction::{by_pairs::r#async::AsyncReducePairs, reduce_to_end::AsyncReduceToEnd}, schema::{snippet::{BorrowedSnippet, Phrase, SnippetExt}, variation::Variation}, validation::validate_with_whatlang};
    /// use std::sync::Arc;
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
    ///     vec![vec!["Hel", "lo ", "WoX", "ld!", "Thi", "is ", "is ", "my ", "str", "ing", "!"]],
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
    /// let reduced_result = phrase_string.reduce_to_end(async move |base| base.reduce_pairs(2, async move |line| validate_with_whatlang(&line)).await).await;
    /// assert!(reduced_result == reduced_phrase,
    /// "Reduced function {:?} did not match reduced result {:?}", reduced_result, reduced_phrase);
    /// # });
    /// ```
    async fn reduce_to_end<F, Fut>(self, reduction_function: F) -> Self
    where
        F: FnMut(Self) -> Fut + Send + Sync,
        Fut: Future<Output = Self> + Send;

    /// Reduces a phrase as much as humanly possible by mutating itself.
    ///
    /// This can be more memory efficient than using [`AsyncReduceToEnd::reduce_to_end`]
    /// as it allows mutating its own contents directly
    async fn reduce_to_end_mut<F, Fut>(self, reduction_function: F) -> Self
    where
        F: FnMut(&mut Self) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send;
}

impl<U> ReduceToEnd for U
where
    U::Item: Debug,
    U: SnippetExt + ConvertString,
{
    fn reduce_to_end_mut<F>(mut self, mut reduction_function: F) -> Self
    where
        F: FnMut(&mut Self),
    {
        let mut last_size = usize::MAX;
        while last_size > self.len_sections() {
            last_size = self.len_sections();
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.permutations()
                    );
                },
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.len_sections(),
                        self.num_of_references(),
                        self.permutations()
                    );
                },
                _ => (),
            };

            // Update permutation depending on which loop
            reduction_function(&mut self);
        }
        self
    }

    fn reduce_to_end<F>(mut self, mut reduction_function: F) -> Self
    where
        F: FnMut(Self) -> Self,
    {
        let mut last_size = usize::MAX;
        while last_size > self.len_sections() {
            last_size = self.len_sections();
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.permutations()
                    );
                },
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.len_sections(),
                        self.num_of_references(),
                        self.permutations()
                    );
                },
                _ => (),
            };

            // Update permutation depending on which loop
            self = reduction_function(self);
        }
        self
    }
}

#[cfg(feature = "async")]
#[async_trait]
impl<U> AsyncReduceToEnd for U
where
    Arc<U::Item>: Sync,
    U: ThreadedSnippetExt + ConvertString + Send,
    U::Item: Debug,
{
    async fn reduce_to_end<F, Fut>(mut self, mut reduction_function: F) -> Self
    where
        F: FnMut(Self) -> Fut + Send + Sync,
        Fut: Future<Output = Self> + Send,
    {
        let mut last_size = usize::MAX;
        while last_size > self.len_sections() {
            last_size = self.len_sections();
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.permutations()
                    );
                },
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.len_sections(),
                        self.num_of_references(),
                        self.permutations()
                    );
                },
                _ => (),
            };

            // Update permutation depending on which loop
            self = reduction_function(self).await;
        }
        self
    }

    async fn reduce_to_end_mut<F, Fut>(mut self, mut reduction_function: F) -> Self
    where
        F: FnMut(&mut Self) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        let mut last_size = usize::MAX;
        while last_size > self.len_sections() {
            last_size = self.len_sections();
            match log::max_level() {
                log::LevelFilter::Info => {
                    log::info!(
                        "Schema: {:?}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.permutations()
                    );
                },
                x if x >= log::LevelFilter::Debug => {
                    log::debug!(
                        "Schema: {:?}\n# of sections: {}\n# of refs: {}\n# of permutations: {:e}",
                        self.convert_to_string(),
                        self.len_sections(),
                        self.num_of_references(),
                        self.permutations()
                    );
                },
                _ => (),
            };

            // Update permutation depending on which loop
            reduction_function(&mut self).await;
        }
        self
    }
}
