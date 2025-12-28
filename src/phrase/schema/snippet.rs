//! This module defines and implements all snippet types that are used in
//! reductions as well as housing base64 variations

use std::{borrow::Borrow, fmt::Display, sync::Arc};

use itertools::Itertools;
#[cfg(feature = "rayon")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator};

use crate::phrase::schema::variation::{Variation, VariationLen, VariationValue};

/// This represents the whole phrase including all of its variable sections.
///
/// This is the primary item used to calculate a textual [`Phrase`] from Base64
/// [`Variation`]s. This item will own the string in a [`Vec`] to allow reducing
/// the number of [`Permutation`]s down using any of the [`reduction`] methods
/// provided.
///
/// ## Examples
///
/// ```rust
/// use base64_bruteforcer_rs::phrase::schema::snippet::{
///     Permutation, Phrase, SnippetExt,
/// };
///
/// let phrase_string: Phrase<String> = [
///     vec!["Hel", "HeR"],
///     vec!["lo "],
///     vec!["Wor", "WoX"],
///     vec!["ld!"],
/// ]
/// .into_iter()
/// .map(|section| section.into_iter().map(String::from))
/// .collect();
///
/// // Provides useful stats for debugging
/// assert!(phrase_string.len_sections() == 4);
/// assert!(phrase_string.len_phrase() == 12);
/// assert!(phrase_string.num_of_references() == 6);
/// assert!(phrase_string.permutations() == 4.0);
///
/// // Provides an easy way to get all combinations of the variations
/// let mut value_iterator = phrase_string.iter_val();
///
/// assert!(value_iterator.next() == Some("Hello World!".to_string()));
/// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
/// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
/// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
/// assert!(value_iterator.next() == None);
///
/// let mut string_iterator = phrase_string.iter_str();
///
/// assert!(string_iterator.next() == Some("Hello World!".to_string()));
/// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
/// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
/// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
/// assert!(string_iterator.next() == None);
/// ```
///
/// [`reduction`]: crate::phrase::reduction
#[derive(Clone, Debug, Default)]
pub struct Phrase<T> {
    pub(crate) sections: Vec<Section<T>>,
}

/// This represents part of a phrase by containing a section of a phrases memory
/// within itself. This can function in most of the same ways a phrase can.
///
/// [`Snippet`] comes with the benefit that it borrows its section from a
/// properly owned type allowing it to analyze and interact with a section
/// without needing to clone the sections to begin work.
///
/// This provides an owned version of a borrow unlike [`BorrowedSnippet`] which
/// can only be used by reference.
///
/// ## Examples
///
/// ```rust
/// use base64_bruteforcer_rs::phrase::schema::snippet::{
///     ConvertString, Permutation, Phrase, Snippet, SnippetExt,
/// };
///
/// let phrase_string: Phrase<String> = [
///     vec!["Hel", "HeR"],
///     vec!["lo "],
///     vec!["Wor", "WoX"],
///     vec!["ld!"],
/// ]
/// .into_iter()
/// .map(|section| section.into_iter().map(String::from))
/// .collect();
///
/// let phrase_snippet = phrase_string.as_snippet();
///
/// // Snippets of the whole phrase are equal to their phrase counterpart
/// assert!(phrase_string == phrase_snippet);
///
/// // Provides useful stats for debugging
/// assert!(phrase_snippet.len_sections() == 4);
/// assert!(phrase_snippet.len_phrase() == 12);
/// assert!(phrase_snippet.num_of_references() == 6);
/// assert!(phrase_snippet.permutations() == 4.0);
///
/// // Provides an easy way to get all combinations of the variations
/// let mut value_iterator = phrase_snippet.iter_val();
///
/// assert!(value_iterator.next() == Some("Hello World!".to_string()));
/// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
/// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
/// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
/// assert!(value_iterator.next() == None);
///
/// let mut string_iterator = phrase_snippet.iter_str();
///
/// assert!(string_iterator.next() == Some("Hello World!".to_string()));
/// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
/// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
/// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
/// assert!(string_iterator.next() == None);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Snippet<'a, T> {
    pub(crate) sections: &'a [Section<T>],
}

/// Represents a variable section. Contains example strings that can take the
/// place of the section. This is used to determine which string is the correct
/// one to use.
pub type Section<T> = Vec<Variation<T>>;

/// This represents a generic phrase that is borrowed.
pub type BorrowedSnippet<T> = [Section<T>];

/// Converts a schema of a non string type into a string schema. This is
/// primarily used in debugging but can be used to pull a schema out beforehand.
///
/// ```rust
/// use base64_bruteforcer_rs::phrase::schema::snippet::{
///     ConvertString, Phrase,
/// };
/// let raw_schema = vec![
///     vec!["Hel".to_owned(), "HeR".to_owned()],
///     vec!["lo ".to_owned()],
///     vec!["Wor".to_owned(), "WoX".to_owned()],
///     vec!["ld!".to_owned()],
/// ];
///
/// let phrase_string: Phrase<String> = Phrase::from_iter(raw_schema.clone());
///
/// assert!(phrase_string.convert_to_string() == raw_schema);
/// ```
pub trait ConvertString {
    /// Produces a copy of the schema with variations converted to strings
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{
    ///     ConvertString, Phrase,
    /// };
    /// let raw_schema = vec![
    ///     vec!["Hel".to_owned(), "HeR".to_owned()],
    ///     vec!["lo ".to_owned()],
    ///     vec!["Wor".to_owned(), "WoX".to_owned()],
    ///     vec!["ld!".to_owned()],
    /// ];
    ///
    /// let phrase_string: Phrase<String> = Phrase::from_iter(raw_schema.clone());
    ///
    /// assert!(phrase_string.convert_to_string() == raw_schema);
    /// ```
    fn convert_to_string(&self) -> Vec<Vec<String>>;
}

/// Provides the permutation function to calculate how many permutations a
/// schema can produce
///
/// ```rust
/// use base64_bruteforcer_rs::phrase::schema::snippet::{Permutation, Phrase};
///
/// let phrase_string: Phrase<String> = [
///     vec!["Hel", "HeR"],
///     vec!["lo "],
///     vec!["Wor", "WoX"],
///     vec!["ld!"],
/// ]
/// .into_iter()
/// .map(|section| section.into_iter().map(String::from))
/// .collect();
///
/// assert!(phrase_string.permutations() == 4.0);
/// ```
pub trait Permutation {
    /// Produces the number of combinations this schema can produce
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Permutation, Phrase};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// assert!(phrase_string.permutations() == 4.0);
    /// ```
    fn permutations(&self) -> f64;
}

/// Provides the functions that provide are utilized by [`Phrase`] and
/// [`Snippet`]
///
/// This implements generic functions that should work for all snippet like
/// objects. This trait is not guaranteed to be thread safe. If you require
/// thread safety, It is recommended to use [`ThreadedSnippetExt`] instead as
/// it reimplements all functions within [`SnippetExt`] in a thread safe manner.
pub trait SnippetExt: Borrow<BorrowedSnippet<Self::Item>> {
    /// Describes the item that is implemented here
    type Item;

    /// Gives the number of sections that make up this phrase
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// assert!(phrase_string.len_sections() == 4);
    /// ```
    fn len_sections(&self) -> usize {
        self.borrow().len()
    }

    /// Gets the length of the phrase based on its value
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// assert!(phrase_string.len_phrase() == 12);
    /// ```
    fn len_phrase(&self) -> usize
    where
        Variation<Self::Item>: VariationLen,
    {
        // Used len at index 0 because all variations in that section should be
        // the same length. If that is not the case, something has gone terribly
        // wrong.
        self.borrow().iter().map(move |s| s[0].len()).sum()
    }

    /// Gives the number of referenced segments used to make this phrase. This
    /// is often used in debugging when developers are tracking memory issues
    /// so this may not be important to you
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// assert!(phrase_string.num_of_references() == 6);
    /// ```
    fn num_of_references(&self) -> usize {
        self.borrow()
            .iter()
            .flat_map(move |s| s.iter().map(move |v| v.num_of_refs()))
            .sum()
    }

    /// Produces a snippet of the phrase where sections can be referenced in
    /// memory rather than a whole new phrase
    fn as_snippet(&self) -> Snippet<'_, Self::Item> {
        Snippet {
            sections: self.borrow(),
        }
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, SnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter()
    ///     .map(ToOwned::to_owned)
    ///     .map(Arc::new)
    ///     .collect();
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.iter_var();
    ///
    /// assert!(value_iterator.next() == Some(string_variation));
    /// ```
    fn iter_var(&self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Variation<Self::Item>: Clone,
    {
        self.borrow()
            .iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, SnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter()
    ///     .map(ToOwned::to_owned)
    ///     .map(Arc::new)
    ///     .collect();
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.into_iter_var();
    ///
    /// assert!(value_iterator.next() == Some(string_variation));
    /// ```
    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone;

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.iter_val();
    ///
    /// assert!(value_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(value_iterator.next() == None);
    /// ```
    fn iter_val(&self) -> impl Iterator<Item = Self::Item>
    where
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.into_iter_val();
    ///
    /// assert!(value_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(value_iterator.next() == None);
    /// ```
    fn into_iter_val(self) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.into_iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all displayable variations that the phrase can take
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// let mut string_iterator = phrase_string.iter_str();
    ///
    /// assert!(string_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(string_iterator.next() == None);
    /// ```
    fn iter_str(&self) -> impl Iterator<Item = String>
    where
        Variation<Self::Item>: Clone + Display,
    {
        self.iter_var().map(move |v| v.to_string())
    }

    /// Permutate through all displayable variations that the phrase can take
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{Phrase, SnippetExt};
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// let mut string_iterator = phrase_string.into_iter_str();
    ///
    /// assert!(string_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(string_iterator.next() == None);
    /// ```
    fn into_iter_str(self) -> impl Iterator<Item = String>
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Display,
    {
        self.into_iter_var().map(move |v| v.to_string())
    }
}

// To explain [`ThreadedSnippetExt`]. The reason why all functions start with
// `par_` is because reimplementing all functions in [`SnippetExt`] is not very
// SemVer compliant as it introduces a requirement to reimplement a function
// for every change to the Ext rather than just only when threading comes into
// question

/// Provides the functions that provide are utilized by [`Phrase`] and
/// [`Snippet`] within a thread safe environment
pub trait ThreadedSnippetExt: SnippetExt
where
    Arc<Self::Item>: Sync,
{
    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    ///
    /// This provides the same object as [`SnippetExt::iter_var`] but in a form
    /// that is thread safe
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, ThreadedSnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter()
    ///     .map(ToOwned::to_owned)
    ///     .map(Arc::new)
    ///     .collect();
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.par_iter_var();
    ///
    /// assert!(value_iterator.next() == Some(string_variation));
    /// ```
    fn par_iter_var(&self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Variation<Self::Item>: Clone,
    {
        self.borrow()
            .iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    ///
    /// This provides the same object as [`SnippetExt::into_iter_var`] but in a
    /// form that is thread safe
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, ThreadedSnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter()
    ///     .map(ToOwned::to_owned)
    ///     .map(Arc::new)
    ///     .collect();
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.par_into_iter_var();
    ///
    /// assert!(value_iterator.next() == Some(string_variation));
    /// ```
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send;

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::iter_val`] but in a form
    /// that is thread safe
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{
    ///     Phrase, ThreadedSnippetExt,
    /// };
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.par_iter_val();
    ///
    /// assert!(value_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(value_iterator.next() == None);
    /// ```
    fn par_iter_val(&self) -> impl Iterator<Item = Self::Item> + Send
    where
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.par_iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::into_iter_val`] but in a
    /// form that is thread safe
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{
    ///     Phrase, ThreadedSnippetExt,
    /// };
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// // Provides an easy way to get all combinations of the variations
    /// let mut value_iterator = phrase_string.par_into_iter_val();
    ///
    /// assert!(value_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(value_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(value_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(value_iterator.next() == None);
    /// ```
    fn par_into_iter_val(self) -> impl Iterator<Item = Self::Item> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send + VariationValue<Item = Self::Item>,
    {
        self.par_into_iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::iter_str`] but in a form
    /// that is thread safe
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{
    ///     Phrase, ThreadedSnippetExt,
    /// };
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// let mut string_iterator = phrase_string.par_iter_str();
    ///
    /// assert!(string_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(string_iterator.next() == None);
    /// ```
    fn par_iter_str(&self) -> impl Iterator<Item = String> + Send
    where
        Variation<Self::Item>: Clone + Display,
    {
        self.par_iter_var().map(move |v| v.to_string())
    }

    /// Permutate through all displayable variations that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::iter_str`] but in a form
    /// that is thread safe
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::snippet::{
    ///     Phrase, ThreadedSnippetExt,
    /// };
    ///
    /// let phrase_string: Phrase<String> = [
    ///     vec!["Hel", "HeR"],
    ///     vec!["lo "],
    ///     vec!["Wor", "WoX"],
    ///     vec!["ld!"],
    /// ]
    /// .into_iter()
    /// .map(|section| section.into_iter().map(String::from))
    /// .collect();
    ///
    /// let mut string_iterator = phrase_string.par_into_iter_str();
    ///
    /// assert!(string_iterator.next() == Some("Hello World!".to_string()));
    /// assert!(string_iterator.next() == Some("Hello WoXld!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo World!".to_string()));
    /// assert!(string_iterator.next() == Some("HeRlo WoXld!".to_string()));
    /// assert!(string_iterator.next() == None);
    /// ```
    fn par_into_iter_str(self) -> impl Iterator<Item = String> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Display + Send,
    {
        self.par_into_iter_var().map(move |s| s.to_string())
    }
}

impl<T> SnippetExt for &BorrowedSnippet<T> {
    type Item = T;

    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone,
    {
        self.iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }
}

impl<T> ThreadedSnippetExt for &BorrowedSnippet<T>
where
    Arc<T>: Sync,
{
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Arc<Self::Item>: Sync,
        Variation<Self::Item>: Clone + Send,
    {
        SnippetExt::into_iter_var(self)
    }
}

impl<T> SnippetExt for Vec<Section<T>> {
    type Item = T;

    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone,
    {
        self.into_iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }
}

impl<T> ThreadedSnippetExt for Vec<Section<T>>
where
    Arc<T>: Sync,
{
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send,
    {
        SnippetExt::into_iter_var(self)
    }
}

impl<T> Borrow<BorrowedSnippet<T>> for Phrase<T> {
    fn borrow(&self) -> &BorrowedSnippet<T> {
        self.sections.as_slice()
    }
}

impl<T> Borrow<BorrowedSnippet<T>> for &Phrase<T> {
    fn borrow(&self) -> &BorrowedSnippet<T> {
        self.sections.as_slice()
    }
}

impl<T> Borrow<BorrowedSnippet<T>> for Snippet<'_, T> {
    fn borrow(&self) -> &BorrowedSnippet<T> {
        self.sections
    }
}

impl<T> Borrow<BorrowedSnippet<T>> for &Snippet<'_, T> {
    fn borrow(&self) -> &BorrowedSnippet<T> {
        self.sections
    }
}

impl<T> AsRef<BorrowedSnippet<T>> for Phrase<T> {
    fn as_ref(&self) -> &BorrowedSnippet<T> {
        self.sections.as_slice()
    }
}

impl<T> AsRef<BorrowedSnippet<T>> for Snippet<'_, T> {
    fn as_ref(&self) -> &BorrowedSnippet<T> {
        self.sections
    }
}

impl<T> SnippetExt for Phrase<T> {
    type Item = T;

    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone,
    {
        self.sections.into_iter_var()
    }
}

impl<T> ThreadedSnippetExt for Phrase<T>
where
    Arc<T>: Sync,
{
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send,
    {
        SnippetExt::into_iter_var(self)
    }
}

impl<T> SnippetExt for Snippet<'_, T> {
    type Item = T;

    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone,
    {
        self.sections.into_iter_var()
    }
}

impl<T> ThreadedSnippetExt for Snippet<'_, T>
where
    Arc<T>: Sync,
{
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send,
    {
        self.into_iter_var()
    }
}

impl<U> SnippetExt for &U
where
    U: SnippetExt,
    for<'a> &'a U: Borrow<BorrowedSnippet<U::Item>>,
{
    type Item = U::Item;

    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone,
    {
        self.iter_var()
    }
}

impl<U> ThreadedSnippetExt for &U
where
    Arc<U::Item>: Sync,
    U: ThreadedSnippetExt,
    for<'a> &'a U: Borrow<BorrowedSnippet<U::Item>>,
{
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send,
    {
        self.par_iter_var()
    }
}

impl<T> Phrase<T> {
    pub fn new(schema: impl AsRef<BorrowedSnippet<T>>) -> Self
    where
        Section<T>: Clone,
    {
        Self {
            sections: schema.as_ref().to_vec(),
        }
    }

    /// Goes through its internal schema and joins all adjacent sections that
    /// have one variation within itself into each other. While this will not
    /// reduce permutation, it can make further reduction easier by reducing the
    /// number of sections to begin with.
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, SnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let before_flatten: Phrase<String> = [
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
    /// let after_flatten: Phrase<String> = [
    ///     vec![vec!["Hel"], vec!["HeR"]],
    ///     vec![vec!["lo "]],
    ///     vec![vec!["Wor"], vec!["WoX"]],
    ///     vec![vec!["ld!"]],
    ///     vec![vec!["Thi"], vec!["ThR"]],
    ///     vec![vec!["is ", "is ", "my "]],
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
    /// assert!(before_flatten.flatten_sections() == after_flatten);
    /// ```
    pub fn flatten_sections(&self) -> Self
    where
        Variation<T>: Clone,
    {
        // Keeping an empty buffer to place all single variant sections into
        let mut singles_buffer: Vec<Variation<T>> = Vec::new();
        let mut new_sections: Vec<Section<T>> = Vec::new();
        let old_sections = self.sections.iter();

        for section in old_sections {
            // If theres more than one variation and our singles buffer is empty,
            // add it to the collection and move on to the next one
            if section.len() > 1 && !singles_buffer.is_empty() {
                // Combine our new single and push to the stack
                new_sections.push(vec![Variation::from_iter(singles_buffer.iter())]);
                // Empty the buffer for the next set of single variations
                singles_buffer.clear();
                // push the new value
                new_sections.push(section.to_vec());
            }
            // If theres more than one variation and our singles buffer is empty,
            // add it to the collection and move on to the next one
            else if section.len() > 1 {
                new_sections.push(section.to_vec());
            }
            // Otherwise, push it to the singles stack
            else {
                singles_buffer.push(section[0].clone());
            }
        }

        // Empty the singles buffer in case the last few lines ended on a single
        if !singles_buffer.is_empty() {
            new_sections.push(vec![Variation::from_iter(singles_buffer)]);
        }
        Self::from_iter(new_sections)
    }

    /// Goes through its internal schema and joins all adjacent sections that
    /// have one variation within itself into each other. While this will not
    /// reduce permutation, it can make further reduction easier by reducing the
    /// number of sections to begin with.
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use base64_bruteforcer_rs::phrase::schema::{
    ///     snippet::{Phrase, SnippetExt},
    ///     variation::Variation,
    /// };
    ///
    /// let before_flatten: Phrase<String> = [
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
    /// let after_flatten: Phrase<String> = [
    ///     vec![vec!["Hel"], vec!["HeR"]],
    ///     vec![vec!["lo "]],
    ///     vec![vec!["Wor"], vec!["WoX"]],
    ///     vec![vec!["ld!"]],
    ///     vec![vec!["Thi"], vec!["ThR"]],
    ///     vec![vec!["is ", "is ", "my "]],
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
    /// assert!(before_flatten.into_flatten_sections() == after_flatten);
    /// ```
    pub fn into_flatten_sections(self) -> Self {
        // Keeping an empty buffer to place all single variant sections into
        let mut singles_buffer: Vec<Variation<T>> = Vec::new();
        let mut new_sections: Vec<Section<T>> = Vec::new();
        let old_sections = self.sections.into_iter();

        for section in old_sections {
            // If theres more than one variation and our singles buffer is empty,
            // add it to the collection and move on to the next one
            if section.len() > 1 && !singles_buffer.is_empty() {
                // Combine our new single and push to the stack
                new_sections.push(vec![Variation::from_iter(singles_buffer)]);
                // Empty the buffer for the next set of single variations
                singles_buffer = Vec::new();
                // push the new value
                new_sections.push(section);
            }
            // If theres more than one variation and our singles buffer is empty,
            // add it to the collection and move on to the next one
            else if section.len() > 1 {
                new_sections.push(section);
            }
            // Otherwise, push it to the singles stack
            else {
                singles_buffer.push(section.into_iter().next().unwrap());
            }
        }

        // Empty the singles buffer in case the last few lines ended on a single
        if !singles_buffer.is_empty() {
            new_sections.push(vec![Variation::from_iter(singles_buffer)]);
        }
        Self::from_iter(new_sections)
    }

    /// Shrinks the backend's allocator to the exact size it currently is.
    ///
    /// This can be useful as a phrase should never increase in size, only
    /// decrease. So running this on initialization of a phrase can be a cheap
    /// method of saving on memory. Especially if the given phrase is memory
    /// intensive.
    pub fn shrink_to_fit(&mut self) {
        self.sections.iter_mut().for_each(|section| {
            section
                .iter_mut()
                .for_each(|variation| variation.shrink_to_fit());
            section.shrink_to_fit();
        });
        self.sections.shrink_to_fit();
    }
}

impl<'a, 'b, T> Snippet<'b, T>
where
    'a: 'b,
{
    pub const fn new(schema: &'a [Section<T>]) -> Self {
        Self { sections: schema }
    }
}

impl<T> From<&BorrowedSnippet<T>> for Phrase<T>
where
    Variation<T>: Clone,
{
    fn from(value: &BorrowedSnippet<T>) -> Self {
        Phrase::new(value)
    }
}

impl<T> From<Vec<Vec<Variation<T>>>> for Phrase<T>
where
    Variation<T>: Clone,
{
    fn from(value: Vec<Vec<Variation<T>>>) -> Self {
        Phrase { sections: value }
    }
}

impl<U, V, W> FromIterator<U> for Phrase<V>
where
    U: IntoIterator<Item = W>,
    W: Into<Variation<V>>,
{
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        Self {
            sections: iter
                .into_iter()
                .map(move |s| s.into_iter().map_into().collect())
                .collect(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<U, V, W> FromParallelIterator<U> for Phrase<V>
where
    Arc<V>: Send,
    U: IntoParallelIterator<Item = W> + Send,
    W: Into<Variation<V>>,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: rayon::prelude::IntoParallelIterator<Item = U>,
    {
        use rayon::iter::ParallelIterator;

        Phrase {
            sections: par_iter
                .into_par_iter()
                .map(move |s| s.into_par_iter().map(move |v| v.into()).collect())
                .collect(),
        }
    }
}

impl<T, const N: usize> From<Phrase<[T; N]>> for Phrase<Vec<T>>
where
    T: Default + PartialEq,
    [T; N]: Clone,
{
    fn from(value: Phrase<[T; N]>) -> Self {
        Self::from_iter(
            value
                .sections
                .into_iter()
                .map(|section| section.into_iter().map(Variation::<Vec<T>>::from)),
        )
    }
}

impl<T, const N: usize> From<Phrase<[T; N]>> for Phrase<String>
where
    T: Default + PartialEq,
    [T; N]: Clone,
    Variation<Vec<T>>: Display,
{
    fn from(value: Phrase<[T; N]>) -> Self {
        Self::from(Phrase::<Vec<T>>::from(value))
    }
}

impl<T> From<Phrase<Vec<T>>> for Phrase<String>
where
    Variation<Vec<T>>: Display,
{
    fn from(value: Phrase<Vec<T>>) -> Self {
        Self::from_iter(
            value
                .sections
                .iter()
                .map(move |s| s.iter().map(move |v| v.to_string())),
        )
    }
}

impl<U, const N: usize> From<U> for Phrase<[u32; N]>
where
    U: SnippetExt<Item = [char; N]>,
{
    fn from(value: U) -> Self {
        Self::from_iter(value.borrow().iter().map(move |s| {
            s.iter().map(move |v| {
                v.value()
                    .into_iter()
                    .map_into()
                    .collect_array::<N>()
                    .unwrap()
            })
        }))
    }
}

impl<U> From<U> for Phrase<Vec<u32>>
where
    U: SnippetExt<Item = Vec<char>>,
{
    fn from(value: U) -> Self {
        Self::from_iter(value.borrow().iter().map(move |s| {
            s.iter()
                .map(move |v| v.value().into_iter().map_into().collect::<Vec<u32>>())
        }))
    }
}

impl TryFrom<Phrase<Vec<u32>>> for Phrase<Vec<char>> {
    type Error = String;

    fn try_from(value: Phrase<Vec<u32>>) -> Result<Self, Self::Error> {
        Ok(Self::from_iter(
            value
                .sections
                .into_iter()
                .map(move |s| {
                    s.into_iter()
                        .map(move |v| {
                            v.into_value()
                                .into_iter()
                                .map(move |c| {
                                    char::from_u32(c).ok_or(format!("Failed to convert to Vec<char>. value {c} is not valid unicode"))
                                }) // Because of how char can error out, this ended up being very dirty
                                .try_collect()
                        }).try_collect()
                })
                .collect::<Result<Vec<Vec<Vec<char>>>, String>>()?,
        ))
    }
}

impl<'a: 'b, 'b, T> From<&'a BorrowedSnippet<T>> for Snippet<'b, T> {
    fn from(value: &'a BorrowedSnippet<T>) -> Self {
        Self { sections: value }
    }
}

impl<'a: 'b, 'b, T> From<Snippet<'a, T>> for &'b BorrowedSnippet<T> {
    fn from(value: Snippet<'a, T>) -> Self {
        value.sections
    }
}

impl<'a: 'b, 'b, T, U: SnippetExt<Item = T>> From<&'a U> for Snippet<'b, T> {
    fn from(value: &'a U) -> Self {
        Snippet {
            sections: value.borrow(),
        }
    }
}

impl<T> From<Snippet<'_, T>> for Phrase<T>
where
    Variation<T>: Clone,
{
    fn from(value: Snippet<'_, T>) -> Self {
        Self::new(value.sections)
    }
}

impl<T> From<Phrase<T>> for Vec<Section<T>> {
    fn from(value: Phrase<T>) -> Self {
        value.sections
    }
}

impl<T> From<Snippet<'_, T>> for Vec<Section<T>>
where
    Section<T>: Clone,
{
    fn from(value: Snippet<'_, T>) -> Self {
        value.sections.to_vec()
    }
}

impl<U: SnippetExt> PartialEq<U> for Phrase<U::Item>
where
    U::Item: PartialEq,
{
    fn eq(&self, other: &U) -> bool {
        Borrow::<BorrowedSnippet<U::Item>>::borrow(self) == other.borrow()
    }
}

impl<A: IntoIterator<Item = Variation<T>>, T> Extend<A> for Phrase<T> {
    fn extend<U: IntoIterator<Item = A>>(&mut self, iter: U) {
        self.sections.extend(
            iter.into_iter()
                .map(move |section| section.into_iter().collect::<Section<T>>()),
        );
    }
}

impl<T> Eq for Phrase<T> where Phrase<T>: PartialEq {}

impl<U: SnippetExt> PartialEq<U> for Snippet<'_, U::Item>
where
    U::Item: PartialEq,
{
    fn eq(&self, other: &U) -> bool {
        Borrow::<BorrowedSnippet<U::Item>>::borrow(self) == other.borrow()
    }
}

impl<'a, T> Eq for Snippet<'a, T> where Snippet<'a, T>: PartialEq {}

impl<U: SnippetExt> ConvertString for U
where
    Variation<U::Item>: Display,
{
    fn convert_to_string(&self) -> Vec<Vec<String>> {
        self.borrow()
            .iter()
            .map(move |section| {
                section
                    .iter()
                    .map(move |variation| variation.to_string())
                    .collect()
            })
            .collect()
    }
}

impl<U: SnippetExt> Permutation for U {
    fn permutations(&self) -> f64 {
        self.borrow().permutations()
    }
}
