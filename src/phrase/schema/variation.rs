//! This module provides all the implementations and definitions of a
//! [`Variation`]
//!
//! The [`Variation`] is used as a more memory optimized version of a
//! [`String`] that is used for [`SnippetExt`] objects. Because reductions
//! often shorten a string by creating a new section with the best options from
//! combining two or more sections, this is potentially a lot of copying.
//! Because this tool is meant to work with strings or raw bytes for efficiency,
//! the worst case scenario would be to reduce 16 possibilities to 8 by copying
//! two of the variations multiple times based on their combinations.
//!
//! While these combinations are all wanted, its not reasonable to hold multiple
//! copies in memory. Luckily any and all potential combinations are going to
//! be one of the produced values from [`Base64Parser`], so we can maintain
//! reasonable memory sizes by containing [`Arc`]s of each section instead.
//!
//! Because of this, the [`Variation`] is designed to act the same way that the
//! raw copy would while internally only holding references to each section
//! saving memory.

use std::{fmt::Display, sync::Arc};

use itertools::Itertools;

/// A smart method of containing links to snippets. This helps reduce memory
/// usage when links get combined together
///
/// The [`Variation`] is used as a more memory optimized version of a
/// [`String`] that is used for [`SnippetExt`] objects. Because reductions
/// often shorten a string by creating a new section with the best options from
/// combining two or more sections, this is potentially a lot of copying.
/// Because this tool is meant to work with strings or raw bytes for efficiency,
/// the worst case scenario would be to reduce 16 possibilities to 8 by copying
/// two of the variations multiple times based on their combinations.
///
/// While these combinations are all wanted, its not reasonable to hold multiple
/// copies in memory. Luckily any and all potential combinations are going to
/// be one of the produced values from [`FromBase64ToAscii`], so we can maintain
/// reasonable memory sizes by containing [`Arc`]s of each section instead.
///
/// Because of this, the [`Variation`] is designed to act the same way that the
/// raw copy would while internally only holding references to each section
/// saving memory.
///
/// [`SnippetExt`]: crate::phrase::schema::snippet::SnippetExt
/// [`FromBase64ToAscii`]: crate::base64_parser::FromBase64ToAscii
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Variation<T> {
    pub(crate) links: Vec<Arc<T>>,
}

/// The goal of this trait is to allow [`Variation`] to produce an internal
/// value of itself
///
/// [`Variation`] is designed to represent a whole collection of its value
/// while utilizing underlying snippets for memory efficiency. Since there is
/// no trait to equally represent all collections nor do all possible
/// variations guarantee the same approach to recreating the total value, this
/// trait is used instead to provide that value.
///
/// For this reason, all implementations of [`Variation`] should implement
/// [`VariationValue`] in order to ensure that the object can provide useful
/// data
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
///
/// This trait is intended to provide a method of getting the length of a
/// snippet without calculating the output of [`VariationValue::value`] first.
/// In the worst case, it is just the length of that result, but some cases
/// can quickly get the result without that such as the case with array based
/// variations.
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
///
/// This is done by placing '|' characters between each chunk which is used
/// primarily in printing strings
///
/// ```rust
/// use base64_bruteforcer_rs::phrase::schema::variation::{Variation, VariationDebug};
/// use std::sync::Arc;
///
/// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
///     .into_iter().map(ToOwned::to_owned).map(Arc::new).collect();
///
/// // Collecting debugging string
/// let debugging_string = string_variation.debug_string().unwrap();
///
/// assert!(debugging_string == "Hel|lo |Wor|ld!");
/// ```
pub trait VariationDebug: Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), core::fmt::Error>;
    /// Provides a better debugging string for [`Variation`]s that helps in keeping
    /// track of each of the base64 pieces being interpreted.
    ///
    /// This is done by placing '|' characters between each chunk which is used
    /// primarily in printing strings
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::variation::{Variation, VariationDebug};
    /// use std::sync::Arc;
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter().map(ToOwned::to_owned).map(Arc::new).collect();
    ///
    /// // Collecting debugging string
    /// let debugging_string = string_variation.debug_string().unwrap();
    ///
    /// assert!(debugging_string == "Hel|lo |Wor|ld!");
    /// ```
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
    ///
    /// This can be highly useful when the number of references in a large
    /// enough variation starts to impact memory
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::phrase::schema::variation::{Variation, VariationDebug};
    /// use std::sync::Arc;
    ///
    /// let string_variation: Variation<String> = ["Hel", "lo ", "Wor", "ld!"]
    ///     .into_iter().map(ToOwned::to_owned).map(Arc::new).collect();
    ///
    /// assert!(string_variation.num_of_refs() == 4);
    /// ```
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

impl<U> FromIterator<Arc<U>> for Variation<U> {
    fn from_iter<T: IntoIterator<Item = Arc<U>>>(iter: T) -> Self {
        Variation {
            links: iter.into_iter().collect(),
        }
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
        let mut link_iter = self.links.iter();
        // Collecting the last link to trim \0 characters
        let last_item = link_iter.next_back().map(move |last_item| {
            str::from_utf8(last_item.as_ref())
                .map_err(move |_| std::fmt::Error)
                .map(|s| s.trim_end_matches('\0'))
        });

        // Print out rest of the string. Fail early if display fails for any reason
        link_iter
            .map(move |to_str| str::from_utf8(to_str.as_slice()).map_err(move |_| std::fmt::Error))
            .try_for_each(|r| r.and_then(|l| write!(f, "{l}")))?;

        // If there is a last item, print it out
        if let Some(last_vec) = last_item {
            last_vec.and_then(|last_str| write!(f, "{last_str}"))?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    #[cfg(test)]
    /// This section contains all tests that can be done on byte variations
    mod testing_u8_variation {
        use lazy_static::lazy_static;

        use super::super::*;

        lazy_static! {
            /// This model is used for testing variations that are based on vec
            /// collections
            static ref VARIATION_VEC_U8: Variation<Vec<u8>> = [
                b"Hel", b"lo ", b"Wor", b"ld!",
            ].into_iter().map(Vec::from).map(Arc::new).collect();

            /// This model is used for testing variations based on array
            /// collections
            static ref VARIATION_ARR_U8: Variation<[u8; 3]> = [
                b"Hel", b"lo ", b"Wor", b"ld!",
            ].into_iter().map(ToOwned::to_owned).map(Arc::new).collect();

            /// This model is used for testing variations based on array
            /// collections
            ///
            /// Due to the fact that array variations are typically in groups
            /// of 3, this variation helps test shorter strings that aren't
            /// multiples of 3
            static ref VARIATION_ARR_U8_SHORT: Variation<[u8; 3]> = [
                b"Hel", b"lo ", b"Wor", b"ld!",
                b" Th", b"is ", b"is ", b"my ",
                b"str", b"ing", b"!\0\0",
            ].into_iter().map(ToOwned::to_owned).map(Arc::new).collect();


        }

        #[test]
        /// Tests that [`Variation<Vec<u8>>`] produces a correct [`VariationValue`]
        fn test_vec_u8_value() {
            assert!(
                VARIATION_VEC_U8.value() == b"Hello World!".to_vec(),
                "Variation did not give correct value"
            );
        }

        #[test]
        /// Tests that [`Variation<[u8; 3]>`] will provide a correct [`VariationValue`]
        fn test_arr_u8_value() {
            assert!(VARIATION_ARR_U8.value() == b"Hello World!".to_vec())
        }

        #[test]
        /// Tests to see that the length of the variation shows as the correct
        /// length
        fn test_vec_u8_len() {
            assert!(VARIATION_VEC_U8.len() == 12);
        }

        #[test]
        /// Tests if a [`Variation`] matches with the correct [`VariationLen`]
        fn test_arr_u8_len() {
            assert!(VARIATION_ARR_U8.len() == 12);
        }

        #[test]
        /// Tests if a [`Variation`] matches with the correct [`VariationLen`]
        ///
        /// This test ensures that the correct length is derived with a length
        /// less than a multiple of 3
        fn test_arr_u8_len_short() {
            assert!(VARIATION_ARR_U8_SHORT.len() == 31);
        }

        #[test]
        /// Ensures that converting a variation to a string produces the
        /// correct value
        fn test_vec_u8_as_string() {
            assert!(
                VARIATION_VEC_U8.to_string() == "Hello World!",
                "The created string [{}] does not match the given string [Hello World!]",
                VARIATION_VEC_U8.to_string().escape_debug()
            );
        }

        #[test]
        /// Ensures that converting a variation to a string produces the
        /// correct value
        fn test_arr_u8_as_string() {
            assert!(
                VARIATION_ARR_U8.to_string() == "Hello World!",
                "The created string [{}] does not match the given string [Hello World!]",
                VARIATION_ARR_U8.to_string().escape_debug()
            );
        }

        #[test]
        /// Ensures that converting a variation to a string produces the
        /// correct value
        ///
        /// Ensures that shortened strings automatically snip the trailing \0
        /// at the end
        fn test_arr_u8_short_as_string() {
            assert!(
                VARIATION_ARR_U8_SHORT.to_string() == "Hello World! This is my string!",
                "The created string [{}] does not match the given string [Hello World! This is my string!]",
                VARIATION_ARR_U8_SHORT.to_string().escape_debug()
            );
        }
    }
}
