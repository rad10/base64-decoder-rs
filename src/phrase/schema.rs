//! Module provides all schema for the phrase structure.

use std::{borrow::Borrow, fmt::Display, sync::Arc};

use itertools::Itertools;
#[cfg(feature = "rayon")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator};

/// This represents the whole phrase including all of its variable sections.
#[derive(Clone, Debug, Default)]
pub struct Phrase<T> {
    pub(crate) sections: Vec<Section<T>>,
}

/// This represents part of a phrase by containing a section of a phrases memory
/// within itself. This can function in all of the same ways a phrase can
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

/// A smart method of containing links to snippets. This helps reduce memory
/// usage when links get combined together
///
/// The idea behind this is that combinations are held within 3 characters per
/// snippet. When creating larger combinations of these characters, we can
/// contain the link to the data instead of directly copying it.
#[derive(Clone, Debug, Default)]
pub struct Variation<T> {
    pub(crate) links: Vec<Arc<T>>,
}

/// The goal of this trait is to allow [`Variation`] to produce an internal
/// value of itself
pub trait VariationValue {
    type Item;
    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    fn value(&self) -> Self::Item;

    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    fn into_value(self) -> Self::Item;
}

/// Provides [`Variation`] the ability to provide the length of itself
pub trait VariationLen {
    /// Provides the length of itself
    fn len(&self) -> usize;

    /// Determines if a variation has no links or is an empty item
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Provides a better debugging format for [`Variation`]s that helps in keeping
/// track of each of the base64 pieces being interpreted.
pub trait VariationDebug: Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), core::fmt::Error>;
    fn debug_string(&self) -> Result<String, std::fmt::Error>;
}

/// Converts a schema of a non string type into a string type
pub trait ConvertString {
    /// Produces a copy of the schema with variations converted to strings
    fn convert_to_string(&self) -> Vec<Vec<String>>;
}

/// Provides the permutation function to calculate how many permutations a
/// schema can produce
pub trait Permutation {
    /// Produces the number of combinations this schema can produce
    fn permutations(&self) -> f64;
}

impl<T> Variation<T> {
    /// Creates a new link using the given type
    pub fn new(value: T) -> Self {
        Self {
            links: vec![Arc::new(value)],
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Arc<T>> {
        self.links.iter()
    }

    /// Provides the number of segments that make up this variation
    ///
    /// Another way to describe this is to print the number of references used
    /// to create this variation
    pub fn num_of_refs(&self) -> usize {
        self.links.len()
    }

    /// Shrinks the backend's allocator to the exact size it currently is.
    ///
    /// This can be useful as a phrase should never increase in size, only
    /// decrease. So running this on initialization of a phrase can be a cheap
    /// method of saving on memory. Especially if the given phrase is memory
    /// intensive.
    pub fn shrink_to_fit(&mut self) {
        self.links.shrink_to_fit();
    }
}

impl<'a, V> FromIterator<&'a Variation<V>> for Variation<V> {
    fn from_iter<T: IntoIterator<Item = &'a Variation<V>>>(iter: T) -> Self {
        Self {
            links: iter
                .into_iter()
                .flat_map(move |v| v.links.iter())
                .cloned()
                .collect(),
        }
    }
}

impl<V, U: Into<Variation<V>>> FromIterator<U> for Variation<V> {
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        Self {
            links: iter
                .into_iter()
                .flat_map(move |v| v.into().links.into_iter())
                .collect(),
        }
    }
}

impl<T: Default + PartialEq, const N: usize> VariationLen for Variation<[T; N]> {
    fn len(&self) -> usize {
        N * self.links.len()
            // Subtract any tail \0 in combination
            - self
                .links
                .last()
                .map_or(0, |v| v.iter().filter(|c| **c == T::default()).count())
    }
}

impl<T> VariationLen for Variation<Vec<T>> {
    fn len(&self) -> usize {
        self.links.iter().map(|l| l.len()).sum()
    }
}

impl VariationLen for Variation<String> {
    fn len(&self) -> usize {
        self.links.iter().map(|l| l.len()).sum()
    }
}

impl Display for Variation<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links.iter().try_for_each(move |l| write!(f, "{l}"))
    }
}

impl VariationDebug for Variation<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        write!(f, "|")?;
        self.links.iter().try_for_each(move |l| write!(f, "{l}|"))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self.links.iter().join("|"))
    }
}

impl<const N: usize> Display for Variation<[u8; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .try_for_each(move |r| r.and_then(|l| write!(f, "{l}")))
    }
}

impl Display for Variation<Vec<u8>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .try_for_each(move |r| r.and_then(|l| write!(f, "{l}")))
    }
}

impl<const N: usize> VariationDebug for Variation<[u8; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .try_for_each(move |r| r.and_then(|l| write!(f, "{l}|")))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .collect::<Result<Vec<&str>, std::fmt::Error>>()?
            .into_iter()
            .join("|"))
    }
}

impl VariationDebug for Variation<Vec<u8>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .try_for_each(move |r| r.and_then(|l| write!(f, "{l}|")))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .collect::<Result<Vec<&str>, std::fmt::Error>>()?
            .into_iter()
            .join("|"))
    }
}

impl<const N: usize> Display for Variation<[u16; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .try_for_each(move |l| write!(f, "{l}"))
    }
}

impl Display for Variation<Vec<u16>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .try_for_each(move |l| write!(f, "{l}"))
    }
}

impl<const N: usize> VariationDebug for Variation<[u16; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .try_for_each(move |l| write!(f, "{l}|"))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .join("|"))
    }
}

impl VariationDebug for Variation<Vec<u16>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .try_for_each(move |l| write!(f, "{l}|"))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .join("|"))
    }
}

impl<const N: usize> Display for Variation<[u32; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .flat_map(move |v| v.iter())
            .map(move |u| char::from_u32(*u).ok_or(std::fmt::Error))
            .try_for_each(move |c| c.and_then(|l| write!(f, "{l}")))
    }
}

impl Display for Variation<Vec<u32>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .flat_map(move |v| v.iter())
            .map(move |u| char::from_u32(*u).ok_or(std::fmt::Error))
            .try_for_each(move |c| c.and_then(|l| write!(f, "{l}")))
    }
}

impl<const N: usize> VariationDebug for Variation<[u32; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |s| {
                s.iter()
                    .map(|c| char::from_u32(*c).ok_or(std::fmt::Error))
                    .collect::<Result<Vec<char>, std::fmt::Error>>()
            })
            .map(move |s| s.map(|i| String::from_iter(i)))
            .try_for_each(move |t| t.and_then(|l| write!(f, "{l}|")))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |s| {
                s.iter()
                    .map(|c| char::from_u32(*c).ok_or(std::fmt::Error))
                    .collect::<Result<Vec<char>, std::fmt::Error>>()
            })
            .map(move |s| s.map(|i| String::from_iter(i)))
            .collect::<Result<Vec<String>, std::fmt::Error>>()?
            .into_iter()
            .join("|"))
    }
}

impl VariationDebug for Variation<Vec<u32>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |s| {
                s.iter()
                    .map(|c| char::from_u32(*c).ok_or(std::fmt::Error))
                    .collect::<Result<Vec<char>, std::fmt::Error>>()
            })
            .map(move |s| s.map(|i| String::from_iter(i)))
            .try_for_each(move |t| t.and_then(|l| write!(f, "{l}|")))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |s| {
                s.iter()
                    .map(|c| char::from_u32(*c).ok_or(std::fmt::Error))
                    .collect::<Result<Vec<char>, std::fmt::Error>>()
            })
            .map(move |s| s.map(|i| String::from_iter(i)))
            .collect::<Result<Vec<String>, std::fmt::Error>>()?
            .into_iter()
            .join("|"))
    }
}

impl<const N: usize> Display for Variation<[char; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .flat_map(move |v| v.iter())
            .try_for_each(move |l| write!(f, "{l}"))
    }
}

impl Display for Variation<Vec<char>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.links
            .iter()
            .flat_map(move |v| v.iter())
            .try_for_each(move |l| write!(f, "{l}"))
    }
}

impl<const N: usize> VariationDebug for Variation<[char; N]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |v| String::from_iter(v.iter()))
            .try_for_each(move |l| write!(f, "{l}|"))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |v| String::from_iter(v.iter()))
            .join("|"))
    }
}

impl VariationDebug for Variation<Vec<char>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "|")?;
        self.links
            .iter()
            .map(move |v| String::from_iter(v.iter()))
            .try_for_each(move |l| write!(f, "{l}|"))
    }

    fn debug_string(&self) -> Result<String, std::fmt::Error> {
        Ok(self
            .links
            .iter()
            .map(move |v| String::from_iter(v.iter()))
            .join("|"))
    }
}

impl<A: Into<Arc<T>>, T> Extend<A> for Variation<T> {
    fn extend<U: IntoIterator<Item = A>>(&mut self, iter: U) {
        self.links.extend(iter.into_iter().map_into());
    }
}

impl VariationValue for Variation<String> {
    type Item = String;

    fn value(&self) -> String {
        self.links.iter().join("")
    }

    fn into_value(self) -> String {
        self.links.into_iter().join("")
    }
}

impl<T, const N: usize> VariationValue for Variation<[T; N]>
where
    [T; N]: Clone,
{
    type Item = Vec<T>;

    fn value(&self) -> Self::Item {
        self.links
            .iter()
            .cloned()
            .flat_map(Arc::unwrap_or_clone)
            .collect()
    }

    fn into_value(self) -> Self::Item {
        self.links
            .into_iter()
            .flat_map(Arc::unwrap_or_clone)
            .collect()
    }
}

impl<T> VariationValue for Variation<Vec<T>>
where
    Vec<T>: Clone,
{
    type Item = Vec<T>;

    fn value(&self) -> Self::Item {
        self.links
            .iter()
            .cloned()
            .flat_map(Arc::unwrap_or_clone)
            .collect()
    }

    fn into_value(self) -> Self::Item {
        self.links
            .into_iter()
            .flat_map(Arc::unwrap_or_clone)
            .collect()
    }
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
    fn len_sections(&self) -> usize {
        self.borrow().len()
    }

    /// Gets the length of the phrase based on its value
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
    fn into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>>
    where
        Self: Sized,
        Variation<Self::Item>: Clone;

    /// Permutate through all [`Variation`]s that the phrase can take
    fn iter_val(&self) -> impl Iterator<Item = Self::Item>
    where
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all [`Variation`]s that the phrase can take
    fn into_iter_val(self) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.into_iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all displayable variations that the phrase can take
    fn iter_str(&self) -> impl Iterator<Item = String>
    where
        Variation<Self::Item>: Clone + Display,
    {
        self.iter_var().map(move |v| v.to_string())
    }

    /// Permutate through all displayable variations that the phrase can take
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
    /// This provides the same object as [`SnippetExt::into_iter_var`] but in a form
    /// that is thread safe
    fn par_into_iter_var(self) -> impl Iterator<Item = Variation<Self::Item>> + Send
    where
        Self: Sized,
        Variation<Self::Item>: Clone + Send;

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::iter_val`] but in a form
    /// that is thread safe
    fn par_iter_val(&self) -> impl Iterator<Item = Self::Item> + Send
    where
        Variation<Self::Item>: Clone + VariationValue<Item = Self::Item>,
    {
        self.par_iter_var()
            .map(<Variation<Self::Item> as VariationValue>::into_value)
    }

    /// Permutate through all [`Variation`]s that the phrase can take
    ///
    /// This provides the same object as [`SnippetExt::into_iter_val`] but in a form
    /// that is thread safe
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
    pub fn new(schema: &'a [Section<T>]) -> Self {
        Self { sections: schema }
    }
}

impl<T> From<T> for Variation<T> {
    fn from(value: T) -> Self {
        Variation::new(value)
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

impl<T, U> From<U> for Phrase<String>
where
    U: SnippetExt<Item = Vec<T>>,
    Variation<Vec<T>>: Display,
{
    fn from(value: U) -> Self {
        Self::from_iter(
            value
                .borrow()
                .iter()
                .map(move |s| s.iter().map(move |v| v.to_string())),
        )
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

impl<'a, 'b, T> From<&'a BorrowedSnippet<T>> for Snippet<'b, T>
where
    'a: 'b,
{
    fn from(value: &'a BorrowedSnippet<T>) -> Self {
        Self { sections: value }
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

// Makes sense to handle it on a generic level since its literally just
// calculating the cartesian product
impl<T> Permutation for [Vec<T>] {
    fn permutations(&self) -> f64 {
        self.iter()
            .map(move |section| section.len() as f64)
            .product()
    }
}

impl<U: SnippetExt> Permutation for U {
    fn permutations(&self) -> f64 {
        self.borrow().permutations()
    }
}
