//! This module provides all the implementations and definitions of a variation

use std::{fmt::Display, sync::Arc};

use itertools::Itertools;

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

impl<T, const N: usize> From<Variation<[T; N]>> for Variation<Vec<T>>
where
    [T; N]: Clone,
{
    fn from(value: Variation<[T; N]>) -> Self {
        Self {
            links: value
                .links
                .into_iter()
                .map(Arc::unwrap_or_clone)
                .map(move |a| Arc::new(a.into_iter().collect::<Vec<T>>()))
                .collect(),
        }
    }
}

impl<T, const N: usize> From<&Variation<[T; N]>> for Variation<Vec<T>>
where
    [T; N]: Clone,
{
    fn from(value: &Variation<[T; N]>) -> Self {
        Self {
            links: value
                .links
                .iter()
                .cloned()
                .map(Arc::unwrap_or_clone)
                .map(move |a| Arc::new(a.into_iter().collect::<Vec<T>>()))
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
            .map(move |s| s.map(String::from_iter))
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
            .map(move |s| s.map(String::from_iter))
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
            .map(move |s| s.map(String::from_iter))
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
            .map(move |s| s.map(String::from_iter))
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

impl<T> From<T> for Variation<T> {
    fn from(value: T) -> Self {
        Variation::new(value)
    }
}
