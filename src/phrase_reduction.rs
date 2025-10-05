//! This module provides the schema and structures for how string snippets get
//! selected and reduced down to better options
#![allow(dead_code)]

use std::{fmt::Display, sync::Arc};

use crate::{
    base64_parser::{Base64Bruteforcer, ConvertString, DisplayLines, Permutation},
    phrase_solving::{SchemaReduce, StringBruteforcer, determine_accuracy_whatlang},
};
use itertools::Itertools;
use rayon::{
    iter::{ParallelBridge, ParallelIterator},
    slice::ParallelSlice,
};

/// This represents the whole phrase including all of its variable sections.
#[derive(Clone, Debug, Default)]
pub struct Phrase<T> {
    sections: Vec<Section<T>>,
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
    links: Vec<Arc<T>>,
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

impl<T> Variation<T>
where
    T: Sized,
{
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
    pub fn join(container: Vec<&Variation<T>>) -> Variation<T> {
        Self {
            links: container
                .into_iter()
                .flat_map(|v| v.links.iter())
                .cloned()
                .collect(),
        }
    }
}

impl<T> Extend<Arc<T>> for Variation<T> {
    fn extend<U: IntoIterator<Item = Arc<T>>>(&mut self, iter: U) {
        self.links.extend(iter);
    }
}

impl Variation<String> {
    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    pub fn value(&self) -> String {
        self.links.iter().join("")
    }
}

impl<T> Variation<Vec<T>> {
    /// Takes the underlying value and produces a combined variant of the raw
    /// value
    pub fn value(&self) -> Vec<T>
    where
        Vec<T>: Clone,
    {
        self.links
            .iter()
            .flat_map(|v| Arc::unwrap_or_clone(v.to_owned()))
            .collect()
    }
}

impl<T> Phrase<T> {
    pub fn new(schema: Vec<Section<T>>) -> Self {
        Self { sections: schema }
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
                        .collect_vec()
                })
                .collect_vec(),
        }
    }
}

impl From<StringBruteforcer> for Phrase<String> {
    fn from(value: StringBruteforcer) -> Self {
        Self::from(value.schema)
    }
}

impl<T> From<Base64Bruteforcer<T>> for Phrase<String>
where
    Base64Bruteforcer<T>: ConvertString,
{
    fn from(value: Base64Bruteforcer<T>) -> Self {
        Self::from(value.convert_to_string())
    }
}

impl<T> From<Base64Bruteforcer<T>> for Phrase<Vec<T>>
where
    T: Clone,
{
    fn from(value: Base64Bruteforcer<T>) -> Self
    where
        T: Clone,
    {
        Self {
            sections: value
                .schema
                .iter()
                .map(|section| {
                    section
                        .iter()
                        .map(|variation| Variation::new(variation.to_owned()))
                        .collect_vec()
                })
                .collect_vec(),
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

impl<T> DisplayLines<Vec<T>> for Phrase<Vec<T>>
where
    T: Clone,
{
    fn produce_lines(&self) -> impl Iterator<Item = Vec<T>>
    where
        T: Clone,
    {
        self.sections
            .iter()
            .multi_cartesian_product()
            .map(|v| v.into_iter().map(|t| t.value()).concat())
    }
}

impl<T> Permutation for Phrase<T> {
    fn permutations(&self) -> f64 {
        self.sections
            .iter()
            .map(|section| section.len() as f64)
            .product()
    }
}

impl ConvertString for Phrase<String> {
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

impl SchemaReduce for Phrase<String> {
    fn reduce_to_end(&mut self) {
        let mut pair_size = 2;
        while pair_size <= self.sections.len() {
            let mut last_size = usize::MAX;
            while last_size > self.sections.len() {
                last_size = self.sections.len();
                self.reduce_schema(Some(pair_size));
                match log::max_level() {
                    log::LevelFilter::Info => {
                        log::info!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.convert_to_string(),
                            self.permutations()
                        );
                    }
                    x if x >= log::LevelFilter::Debug => {
                        log::debug!(
                            "Schema: {:?}\n# of permutations: {:e}",
                            self.sections,
                            self.permutations()
                        );
                    }
                    _ => (),
                };
            }
            pair_size += 1;
            log::debug!("Increasing pair size to {pair_size}");
        }
    }

    fn reduce_schema(&mut self, number_of_pairs: Option<usize>) {
        // Check to make sure size is correctly placed or replace with own value
        let pair_size = match number_of_pairs {
            Some(0..2) | None => 2, // Overwrite any stupid options with the
            // default
            Some(n) if n < self.sections.len() => n,
            Some(_) => self.sections.len(), // If the number is bigger than the
                                            // source itself, just use the length of the source. Its not
                                            // recommended to ever do this since its no different than checking
                                            // line by line.
        };

        // Take and operate on each pair in the schema. Will either combine a
        // pair into one section or (worst case scenario) leave the pairs as is
        self.sections = self
            .sections
            .par_chunks(pair_size)
            .inspect(|pairs| log::debug!("Visible pair: {pairs:?}"))
            .map(|pairs| {
                // If its only 1 pair, we can skip this process
                if pairs.len() == 1 {
                    return pairs.to_vec();
                }

                // If there is more than one pair, but each pair only has one
                // value, then just return a single combined form. It will give
                // future runs more information and clarity
                if pairs.iter().all(|v| v.len() == 1) {
                    return vec![vec![pairs[0][0].combine(&pairs[1][0])]];
                }

                // permuting values and collecting only viable options
                let combined: Vec<Vec<Variation<String>>> = vec![
                    pairs
                        .iter()
                        // Get all combinations of the variations
                        .multi_cartesian_product()
                        .par_bridge()
                        // Join them together to get the string to test against
                        .map(Variation::join)
                        .inspect(|line| log::debug!("detect.string: {line}"))
                        // if confidence if over 10%, it moves on to the next round
                        .filter(|line| determine_accuracy_whatlang(line.value().as_str(), 0.10))
                        .collect(),
                ];

                // Go with originals if new choices aren't preferred
                if combined[0].is_empty() {
                    pairs.to_vec()
                } else {
                    combined
                }
            })
            .collect::<Vec<Vec<Section<String>>>>()
            .concat();
    }
}
