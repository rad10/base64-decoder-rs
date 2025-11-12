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

/// Provides an iterator to print all schema permutations
pub trait DisplayLines<T> {
    /// Produces an iterator of the every permutable line from the schema
    fn produce_lines(&self) -> impl Iterator<Item = T>;
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

    /// Takes two links and creates a new link containing each value in concatenation
    pub fn combine(&self, other: &Variation<T>) -> Self
    where
        Self: Sized,
    {
        Self {
            links: [self.links.clone(), other.links.clone()].concat(),
        }
    }

    /// Takes an array of variations and combines them in order into a single variation
    pub fn join<'a>(container: [&'a Variation<T>]) -> Variation<T>
    where
        [&'a Variation<T>]: Sized,
    {
        Self {
            links: container
                .iter()
                .flat_map(|v| v.links.iter())
                .cloned()
                .collect(),
        }
    }

    /// Takes an array of variations and combines them in order into a single variation
    pub fn join_var_vec(container: Vec<&Variation<T>>) -> Variation<T> {
        Self {
            links: container
                .iter()
                .flat_map(|v| v.links.iter())
                .cloned()
                .collect(),
        }
    }

    /// Takes an array of variations and combines them in order into a single variation
    pub fn join_into_var_vec(container: Vec<Variation<T>>) -> Variation<T> {
        Self {
            links: container
                .into_iter()
                .flat_map(|v| v.links.into_iter())
                .collect(),
        }
    }

    /// Takes an array of variations and combines them in order into a single variation
    pub fn join_var_slice(container: &[&Variation<T>]) -> Variation<T> {
        Self {
            links: container
                .iter()
                .flat_map(|v| v.links.iter())
                .cloned()
                .collect(),
        }
    }

    pub fn join_into_var_slice(container: &[Variation<T>]) -> Variation<T> {
        Self {
            links: container
                .into_iter()
                .flat_map(|v| v.links.iter())
                .cloned()
                .collect(),
        }
    }

    /// Provides the number of segments that make up this variation
    ///
    /// Another way to describe this is to print the number of references used
    /// to create this variation
    pub fn num_of_refs(&self) -> usize {
        self.links.len()
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
            .map(|tostr| str::from_utf8(tostr.as_slice()).unwrap())
            .join("");
        write!(f, "{}", combo)
    }
}

impl Display for Variation<Vec<u16>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let combo = self
            .links
            .iter()
            .map(|tostr| String::from_utf16_lossy(tostr.as_slice()))
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
            .flat_map(|v| Arc::unwrap_or_clone(v.to_owned()))
            .collect()
    }

    fn into_value(self) -> Self::Item {
        self.links
            .into_iter()
            .flat_map(|v| Arc::unwrap_or_clone(v))
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
        Section<T>: Clone,
    {
        // Creating a copy of sections here because we cant have double mutability
        let mut old_sections = self.sections.clone();
        let mut new_sections: Vec<Section<T>> = Vec::new();

        // Cant do this in a chain method due to the strategy taken here.
        // Because were using pop, we are doing this in reverse. This means
        // variation needs to be handled with care, and we need to remember to
        // reverse the array after were all done.
        while let Some(mut item) = old_sections.pop() {
            // If theres more than one variation, add it to the collection and
            // move on to the next one
            if item.len() > 1 {
                new_sections.push(item);
            } else {
                // Reverse items string in preparation
                item[0].links.reverse();
                // Since we know that this item is only one value, we feed values
                // to the current item until we meet a section with multiple
                // variations
                while let Some(mut second_item) = old_sections.pop() {
                    // Escape and place items if this new item has multiple variations
                    if second_item.len() > 1 {
                        // Our loop ends. Clean up item and place it and this
                        // new item into the vector
                        item[0].links.reverse();
                        new_sections.push(item);
                        new_sections.push(second_item);
                        break;
                    }
                    // Since the new item has only one value, reverse the inner
                    // links of the new item and combine it with the old one
                    second_item[0].links.reverse();
                    item[0] = item[0].combine(&second_item[0]);
                }
            }
        }
        // At the end of the loop, we reverse the vec and replace sections
        new_sections.reverse();

        Self::new(new_sections)
    }

    /// Goes through its internal schema and joins all adjacent sections that
    /// have one variation within itself into each other. While this will not
    /// reduce permutation, it can make further reduction easier by reducing the
    /// number of sections to begin with.
    pub fn into_flatten_sections(self) -> Self
    where
        Section<T>: Clone,
    {
        // Creating a copy of sections here because we cant have double mutability
        let mut old_sections = self.sections.clone();
        let mut new_sections: Vec<Section<T>> = Vec::new();

        // Cant do this in a chain method due to the strategy taken here.
        // Because were using pop, we are doing this in reverse. This means
        // variation needs to be handled with care, and we need to remember to
        // reverse the array after were all done.
        while let Some(mut item) = old_sections.pop() {
            // If theres more than one variation, add it to the collection and
            // move on to the next one
            if item.len() > 1 {
                new_sections.push(item);
            } else {
                // Reverse items string in preparation
                item[0].links.reverse();
                // Since we know that this item is only one value, we feed values
                // to the current item until we meet a section with multiple
                // variations
                while let Some(mut second_item) = old_sections.pop() {
                    // Escape and place items if this new item has multiple variations
                    if second_item.len() > 1 {
                        // Our loop ends. Clean up item and place it and this
                        // new item into the vector
                        item[0].links.reverse();
                        new_sections.push(item);
                        new_sections.push(second_item);
                        break;
                    }
                    // Since the new item has only one value, reverse the inner
                    // links of the new item and combine it with the old one
                    second_item[0].links.reverse();
                    item[0] = item[0].combine(&second_item[0]);
                }
            }
        }
        // At the end of the loop, we reverse the vec and replace sections
        new_sections.reverse();

        Self::new(new_sections)
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn iter_var(&self) -> impl Iterator<Item = Variation<T>> {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
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
            .map(|v| Variation::join_into_var_vec(v))
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
            .flat_map(|s| s.iter().map(|v| v.num_of_refs()))
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
        self.sections.iter().map(|s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Phrase<String> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(|s| s[0].len()).sum()
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
    pub fn iter_var(&self) -> impl Iterator<Item = Variation<T>> {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
    }

    /// Creates an iterator of all possible combinations based on the memory
    /// efficient variation structure
    pub fn into_iter_var(self) -> impl Iterator<Item = Variation<T>> {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
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
            .flat_map(|s| s.iter().map(|v| v.num_of_refs()))
            .sum()
    }
}

impl<T> Snippet<'_, Vec<T>> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(|s| s[0].len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Snippet<'_, String> {
    /// Gets the length of the phrase
    pub fn len(&self) -> usize {
        self.sections.iter().map(|s| s[0].len()).sum()
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
        Variation<T>: VariationValue<Item = T>,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.value())
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`iter`]
    ///
    /// [`iter`]: Self::iter
    pub fn iter_val(&self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.iter()
    }
}

impl<T> Phrase<T>
where
    Variation<T>: Clone + VariationValue,
{
    /// Permutate through all variations that the phrase can take
    pub fn into_iter(self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_into_var_vec(v))
            .map(|v| v.into_value())
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`into_iter`]
    ///
    /// [`into_iter`]: Self::into_iter
    pub fn into_iter_val(self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.into_iter()
    }
}

impl<T> Snippet<'_, T> {
    /// Permutate through all variations that the phrase can take
    pub fn iter(&self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.value())
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter(self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.into_value())
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`iter`]
    ///
    /// [`iter`]: Self::iter
    pub fn iter_val(&self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.iter()
    }

    /// Permutate through all variations that the phrase can take
    ///
    /// Same as [`into_iter`]
    ///
    /// [`into_iter`]: Self::into_iter
    pub fn into_iter_val(self) -> impl Iterator<Item = T>
    where
        Variation<T>: VariationValue<Item = T>,
    {
        self.into_iter()
    }
}

impl<T> Phrase<T>
where
    Variation<T>: Display,
{
    /// Permutate through all variations that the phrase can take
    pub fn iter_str(&self) -> impl Iterator<Item = String> {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.to_string())
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter_str(self) -> impl Iterator<Item = String>
    where
        Variation<T>: Clone,
    {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_into_var_vec(v))
            .map(|v| v.to_string())
    }
}

impl<T> Snippet<'_, T>
where
    Variation<T>: Display,
{
    /// Permutate through all variations that the phrase can take
    pub fn iter_str(&self) -> impl Iterator<Item = String> {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.to_string())
    }

    /// Permutate through all variations that the phrase can take
    pub fn into_iter_str(self) -> impl Iterator<Item = String> {
        self.sections
            .into_iter()
            .multi_cartesian_product()
            .map(|v| Variation::join_var_vec(v))
            .map(|v| v.to_string())
    }
}

impl<T> From<Vec<Vec<T>>> for Phrase<T>
where
    T: Clone,
{
    fn from(value: Vec<Vec<T>>) -> Self {
        Self {
            sections: value
                .iter()
                .map(|section| {
                    section
                        .iter()
                        .map(|variation| Variation::new(variation.to_owned()))
                        .collect::<Vec<Variation<T>>>()
                })
                .collect::<Vec<Section<T>>>(),
        }
    }
}

impl<T> From<&[Vec<T>]> for Phrase<T>
where
    T: Clone,
{
    fn from(value: &[Vec<T>]) -> Self {
        Self {
            sections: value
                .iter()
                .map(|section| {
                    section
                        .iter()
                        .map(|variation| Variation::new(variation.to_owned()))
                        .collect::<Vec<Variation<T>>>()
                })
                .collect::<Vec<Section<T>>>(),
        }
    }
}

impl<T> From<Vec<Section<T>>> for Phrase<T> {
    fn from(value: Vec<Section<T>>) -> Self {
        Self::new(value)
    }
}

impl<T> From<[Section<T>]> for Phrase<T>
where
    [Section<T>]: Sized,
    Variation<T>: Clone,
{
    fn from(value: [Section<T>]) -> Self {
        Self {
            sections: value.to_vec(),
        }
    }
}

impl<T> From<&[Section<T>]> for Phrase<T>
where
    T: Clone,
{
    fn from(value: &[Section<T>]) -> Self {
        Self {
            sections: value.to_vec(),
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
                .iter()
                .map(|s| s.iter().map(|v| v.to_string()).collect::<Vec<String>>())
                .collect::<Vec<Vec<String>>>(),
        )
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

impl<T> DisplayLines<String> for Phrase<T>
where
    T: Clone,
    Variation<T>: Display,
{
    fn produce_lines(&self) -> impl Iterator<Item = String>
    where
        Variation<T>: Display,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| v.into_iter().map(|s| s.to_string()).join(""))
    }
}

impl<T> ConvertString for Phrase<T>
where
    Variation<T>: Display,
{
    fn convert_to_string(&self) -> Vec<Vec<String>> {
        self.sections
            .iter()
            .map(|section| {
                section
                    .iter()
                    .map(|variation| variation.to_string())
                    .collect()
            })
            .collect()
    }
}

// Makes sense to handle it on a generic level since its literally just
// calculating the cartesian product
impl<T> Permutation for [Vec<T>] {
    fn permutations(&self) -> f64 {
        self.iter().map(|section| section.len() as f64).product()
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
