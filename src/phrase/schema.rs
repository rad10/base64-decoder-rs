//! Module provides all schema for the phrase structure.

use std::{fmt::Display, sync::Arc};

use itertools::Itertools;

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

/// The goal of this trait is to allow Variation to produce an internal value
/// of itself
pub trait VariationValue {
    type Item;
    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    fn value(&self) -> Self::Item;

    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    fn into_value(self) -> Self::Item;
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

impl<T> Variation<Vec<T>> {
    /// Gets the len of the value within the variation
    pub fn len(&self) -> usize {
        self.links.iter().map(|l| l.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Variation<String> {
    /// Gets the len of the value within the variation
    pub fn len(&self) -> usize {
        self.links.iter().map(|l| l.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Display for Variation<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let combo = self.links.iter().join("");
        write!(f, "{combo}")
    }
}

impl Display for Variation<Vec<u8>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let combo = self
            .links
            .iter()
            .map(move |to_str| str::from_utf8(to_str.as_slice()).unwrap())
            .join("");
        write!(f, "{}", combo)
    }
}

impl Display for Variation<Vec<u16>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let combo = self
            .links
            .iter()
            .map(move |to_str| String::from_utf16_lossy(to_str.as_slice()))
            .join("");
        write!(f, "{}", combo)
    }
}

impl<T> Extend<Arc<T>> for Variation<T> {
    fn extend<U: IntoIterator<Item = Arc<T>>>(&mut self, iter: U) {
        self.links.extend(iter);
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

impl<T> Phrase<T> {
    pub fn new(schema: Vec<Section<T>>) -> Self {
        Self { sections: schema }
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
        Self::new(new_sections)
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
        Self::new(new_sections)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn iter_var(&self) -> impl Iterator<Item = Variation<T>>
    where
        Variation<T>: Clone,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn into_iter_var(self) -> impl Iterator<Item = Variation<T>>
    where
        Variation<T>: Clone,
    {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Gives the number of sections that make up this phrase
    pub fn len_sections(&self) -> usize {
        self.sections.len()
    }

    /// Gives the number of referenced segments used to make this phrase. This
    /// is often used in debugging when developers are tracking memory issues
    /// so this may not be important to you
    pub fn num_of_references(&self) -> usize {
        self.sections
            .iter()
            .flat_map(move |s| s.iter().map(move |v| v.num_of_refs()))
            .sum()
    }

    /// Produces a snippet of the phrase where sections can be referenced in
    /// memory rather than a whole new phrase
    pub fn as_snippet(&self) -> Snippet<'_, T> {
        Snippet::from(self)
    }
}

impl<T> Phrase<Vec<T>> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        // Used len at index 0 because all variations in that section should be
        // the same length. If that is not the case, something has gone terribly
        // wrong.
        self.sections.iter().map(move |s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Phrase<String> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(move |s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

impl<T> Snippet<'_, T> {
    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn iter_var(&self) -> impl Iterator<Item = Variation<T>>
    where
        Variation<T>: Clone,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn into_iter_var(self) -> impl Iterator<Item = Variation<T>>
    where
        Variation<T>: Clone,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(Variation::from_iter)
    }

    /// Gives the number of sections that make up this phrase
    pub fn len_sections(&self) -> usize {
        self.sections.len()
    }

    /// Gives the number of referenced segments used to make this phrase. This
    /// is often used in debugging when developers are tracking memory issues
    /// so this may not be important to you
    pub fn num_of_references(&self) -> usize {
        self.sections
            .iter()
            .flat_map(move |s| s.iter().map(move |v| v.num_of_refs()))
            .sum()
    }
}

impl<T> Snippet<'_, Vec<T>> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(move |s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Snippet<'_, String> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(move |s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Phrase<T>
where
    Variation<T>: VariationValue,
{
    /// Permutate through all variations that the phrase can take
    pub fn iter(&self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T> + Clone,
    {
        self.iter_var()
            .map(<Variation<T> as VariationValue>::into_value)
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`iter`]
    ///
    /// [`iter`]: Self::iter
    pub fn iter_val(&self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T> + Clone,
    {
        self.iter()
    }
}

impl<T> Phrase<T>
where
    Variation<T>: Clone + VariationValue<Item = T>,
{
    /// Permutate through all variations that the phrase can take
    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.into_iter_var()
            .map(<Variation<T> as VariationValue>::into_value)
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`into_iter`]
    ///
    /// [`into_iter`]: Self::into_iter
    pub fn into_iter_val(self) -> impl Iterator<Item = T> {
        self.into_iter()
    }
}

impl<T> Snippet<'_, T>
where
    Variation<T>: VariationValue<Item = T> + Clone,
{
    /// Permutate through all variations that the phrase can take
    pub fn iter(&self) -> impl Iterator<Item = T> {
        self.iter_var()
            .map(<Variation<T> as VariationValue>::into_value)
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.into_iter_var()
            .map(<Variation<T> as VariationValue>::into_value)
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`iter`]
    ///
    /// [`iter`]: Self::iter
    pub fn iter_val(&self) -> impl Iterator<Item = T> {
        self.iter()
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`into_iter`]
    ///
    /// [`into_iter`]: Self::into_iter
    pub fn into_iter_val(self) -> impl Iterator<Item = T> {
        self.into_iter()
    }
}

impl<T> Phrase<T> {
    /// Permutate through all variations that the phrase can take
    pub fn iter_str(&self) -> impl Iterator<Item = String>
    where
        Variation<T>: Clone + Display,
    {
        self.iter_var().map(move |v| v.to_string())
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter_str(self) -> impl Iterator<Item = String>
    where
        Variation<T>: Clone + Display,
    {
        self.into_iter_var().map(move |v| v.to_string())
    }
}

impl<T> Snippet<'_, T> {
    /// Permutate through all variations that the phrase can take
    pub fn iter_str(&self) -> impl Iterator<Item = String>
    where
        Variation<T>: Clone + Display,
    {
        self.iter_var().map(move |v| v.to_string())
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter_str(self) -> impl Iterator<Item = String>
    where
        Variation<T>: Clone + Display,
    {
        self.into_iter_var().map(move |v| v.to_string())
    }
}

impl<T> From<T> for Variation<T> {
    fn from(value: T) -> Self {
        Variation::new(value)
    }
}

impl<T, U, V, W> From<U> for Phrase<T>
where
    U: IntoIterator<Item = V>,
    V: IntoIterator<Item = W>,
    W: Into<Variation<T>>,
{
    fn from(value: U) -> Self {
        Self {
            sections: value
                .into_iter()
                .map(move |s| s.into_iter().map_into().collect())
                .collect(),
        }
    }
}

impl<T> From<Phrase<Vec<T>>> for Phrase<String>
where
    Variation<Vec<T>>: Display,
{
    fn from(value: Phrase<Vec<T>>) -> Self {
        Self::from(
            value
                .sections
                .into_iter()
                .map(move |s| s.into_iter().map(move |v| v.to_string())),
        )
    }
}

impl From<Phrase<Vec<char>>> for Phrase<String> {
    fn from(value: Phrase<Vec<char>>) -> Self {
        Self::from(value.sections.into_iter().map(move |s| {
            s.into_iter()
                .map(move |v| v.value().into_iter().collect::<String>())
        }))
    }
}

impl From<Phrase<Vec<char>>> for Phrase<Vec<u32>> {
    fn from(value: Phrase<Vec<char>>) -> Self {
        Self::from(value.sections.into_iter().map(move |s| {
            s.into_iter()
                .map(move |v| v.value().into_iter().map_into().collect::<Vec<u32>>())
        }))
    }
}

impl TryFrom<Phrase<Vec<u32>>> for Phrase<Vec<char>> {
    type Error = String;

    fn try_from(value: Phrase<Vec<u32>>) -> Result<Self, Self::Error> {
        Ok(Self::from(
            value
                .sections
                .into_iter()
                .map(move |s| {
                    s.into_iter()
                        .map(move |v| {
                            v.into_value()
                                .into_iter()
                                .map(move |c| {
                                    char::from_u32(c).ok_or(format!(
                            "Failed to convert to Vec<char>. value {c} is not valid unicode",
                        ))
                                }) // Because of how char can error out, this ended up being very dirty
                                .collect::<Result<Vec<char>, String>>()
                        })
                        .collect::<Result<Vec<Vec<char>>, String>>()
                })
                .collect::<Result<Vec<Vec<Vec<char>>>, String>>()?,
        ))
    }
}

impl TryFrom<Phrase<Vec<u32>>> for Phrase<String> {
    type Error = String;

    fn try_from(value: Phrase<Vec<u32>>) -> Result<Self, Self::Error> {
        Ok(Phrase::<Vec<char>>::try_from(value)?.into())
    }
}

impl<'a, 'b, T> From<&'a [Section<T>]> for Snippet<'b, T>
where
    'a: 'b,
{
    fn from(value: &'a [Section<T>]) -> Self {
        Self { sections: value }
    }
}

impl<'a, 'b, T> From<&'a Phrase<T>> for Snippet<'b, T>
where
    'a: 'b,
{
    fn from(value: &'a Phrase<T>) -> Self {
        Self {
            sections: &value.sections,
        }
    }
}

impl<'a, T> From<Snippet<'a, T>> for Phrase<T>
where
    Variation<T>: Clone,
{
    fn from(value: Snippet<'a, T>) -> Self {
        Self::new(value.sections.to_vec())
    }
}

impl<T> ConvertString for Phrase<T>
where
    Variation<T>: Display,
{
    fn convert_to_string(&self) -> Vec<Vec<String>> {
        self.sections
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

impl<T> Permutation for Phrase<T> {
    fn permutations(&self) -> f64 {
        self.sections.permutations()
    }
}

impl<T> Permutation for Snippet<'_, T> {
    fn permutations(&self) -> f64 {
        self.sections.permutations()
    }
}
