//! This module contains the structs used to convert utf8 and utf16 to
//! combinations of potential valid characters

use std::marker::PhantomData;

use base64::{Engine, prelude::BASE64_STANDARD};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;

/// Provides the traits to convert streams of lowercased base64 bytes into
/// valid combinations of base64
pub trait BruteforceBase64: IntoIterator<Item = u8> {
    /// This will ingest the base64 string and produce all possible combinations
    /// in an iterative feed.
    ///
    /// ```rust
    /// use base64_bruteforcer_rs::base64_parser::BruteforceBase64;
    ///
    /// let example_base64 = b"sgvsbg8gd29ybgqh";
    ///
    /// let expected: Vec<Vec<[u8; 3]>> = vec![
    ///     vec![[178, 11, 236], [178, 11, 210], [178, 5, 108], [178, 5, 82], [176, 107, 236], [176, 107, 210], [176, 101, 108], [176, 101, 82], [74, 11, 236], [74, 11, 210], [74, 5, 108], [74, 5, 82], [72, 107, 236], [72, 107, 210], [72, 101, 108], [72, 101, 82]],
    ///     vec![[110, 15, 32], [110, 15, 6], [108, 111, 32], [108, 111, 6], [6, 15, 32], [6, 15, 6], [4, 111, 32], [4, 111, 6]],
    ///     vec![[119, 111, 114], [119, 111, 88], [15, 111, 114], [15, 111, 88]],
    ///     vec![[110, 10, 161], [110, 10, 135], [110, 4, 33], [110, 4, 7], [108, 106, 161], [108, 106, 135], [108, 100, 33], [108, 100, 7], [6, 10, 161], [6, 10, 135], [6, 4, 33], [6, 4, 7], [4, 106, 161], [4, 106, 135], [4, 100, 33], [4, 100, 7]]
    /// ];
    ///
    /// let result: Vec<Vec<[u8; 3]>> = example_base64.parse_base64().collect();
    ///
    /// assert!(result == expected, "Expected result {:?} did not match with {:?}", expected, result);
    /// ```
    fn parse_base64(self) -> Base64Parser<Self::IntoIter>;
}

impl<U: IntoIterator<Item = u8>> BruteforceBase64 for U {
    fn parse_base64(self) -> Base64Parser<Self::IntoIter> {
        Base64Parser {
            iterator: self.into_iter(),
        }
    }
}

/// Filters through a section of variants to reduce valid values to logical or
/// accurate values. Requires a replacement value if it ends up filtering out
/// all values
fn filter_variants<T, F>(collection: Vec<[T; 3]>, default: [T; 3], function: F) -> Vec<[T; 3]>
where
    F: FnMut(&[T; 3]) -> bool,
{
    Some(
        collection
            .into_iter()
            .filter(function)
            .collect::<Vec<[T; 3]>>(),
    )
    .filter(|c| !c.is_empty())
    .unwrap_or(vec![default])
}

/// Iterative Parser for combined lowercased base64
///
/// Used for the [`parse_base64`] method
///
/// [`parse_base64`]: BruteforceBase64::parse_base64
#[derive(Clone, Debug)]
pub struct Base64Parser<I: Iterator<Item = u8>> {
    /// Contains the lowercased base64 iterator
    iterator: I,
}

/// Converts bytes from [`Base64Parser`] into sets of the given type T
///
/// Used in the [`convert_to_type`] method
///
/// [`convert_to_type`]: Base64Parser::convert_to_type
#[derive(Clone, Debug)]
pub struct TypeIter<I: Iterator<Item = Vec<[u8; 3]>>, T: Sized> {
    _phantom: PhantomData<T>,
    iterator: I,
}

impl<I: Iterator<Item = u8>> Base64Parser<I> {
    /// Converts a base64 permutations into a different type
    pub fn convert_to_type<T: Sized + Clone>(self) -> TypeIter<Self, T> {
        TypeIter {
            _phantom: PhantomData,
            iterator: self,
        }
    }

    /// Filters out permutations based on a predicate
    ///
    /// Allows providing a default value if all permutations are filtered out
    pub fn filter_or<P>(
        self,
        other: [u8; 3],
        predicate: P,
    ) -> std::iter::Map<Self, impl FnMut(<Self as Iterator>::Item) -> <Self as Iterator>::Item>
    where
        P: FnMut(&[u8; 3]) -> bool + Clone,
    {
        self.map(move |variants| filter_variants(variants, other, predicate.clone()))
    }

    /// Filters out permutations based on a predicate
    ///
    /// Uses default value if all permutations are filtered out
    pub fn filter_or_default<P>(
        self,
        predicate: P,
    ) -> std::iter::Map<Self, impl FnMut(<Self as Iterator>::Item) -> <Self as Iterator>::Item>
    where
        P: FnMut(&[u8; 3]) -> bool + Clone,
    {
        self.map(move |variants| filter_variants(variants, [u8::default(); 3], predicate.clone()))
    }
}

impl<I: Iterator<Item = u8>> Iterator for Base64Parser<I> {
    type Item = Vec<[u8; 3]>;

    fn next(&mut self) -> Option<Self::Item> {
        // Collecting the first 4 values and gain the base64 variations
        let chunk = [
            self.iterator.next()?, // Put fail condition here instead as the
            // first character should never be '='
            self.iterator.next().unwrap_or(b'='),
            self.iterator.next().unwrap_or(b'='),
            self.iterator.next().unwrap_or(b'='),
        ];
        log::trace!("chunk: {}", chunk.escape_ascii());

        // Getting all casing combinations
        let combos = chunk.map(move |chr| {
            if chr.is_ascii_lowercase() {
                vec![chr, chr.to_ascii_uppercase()]
            } else {
                vec![chr]
            }
        });

        Some(
            combos
                .into_iter()
                .multi_cartesian_product()
                .inspect(|option| log::trace!("Testing value: {}", option.escape_ascii()))
                // Filtering our all strings that do not produce a value
                .filter_map(move |snippet| {
                    let mut result = [0_u8; 3];
                    BASE64_STANDARD.decode_slice(snippet, &mut result).ok()?;
                    Some(result)
                })
                .collect(),
        )
    }
}

impl<I: ExactSizeIterator<Item = u8>> ExactSizeIterator for Base64Parser<I> {
    fn len(&self) -> usize {
        self.iterator.len().div_ceil(4)
    }
}

// Double ended can only possibly work if the exact size is known
impl<I: DoubleEndedIterator<Item = u8> + ExactSizeIterator<Item = u8>> DoubleEndedIterator
    for Base64Parser<I>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // return if remaining length is 0
        if self.iterator.len() == 0 {
            return None;
        }

        // Need to check if fully divisible by 4
        let mut chunk = if self.iterator.len().is_multiple_of(4) {
            [
                self.iterator.next_back()?,
                self.iterator.next_back()?,
                self.iterator.next_back()?,
                self.iterator.next_back()?,
            ]
        } else {
            // The only time it shouldn't be divisible by 4 is on the first item
            [0, 1, 2, 3].map(move |ind| {
                if ind < 4 - self.iterator.len() % 4 {
                    b'='
                } else {
                    self.iterator.next_back().unwrap()
                }
            })
        };

        chunk.reverse();
        log::trace!("chunk: {}", chunk.escape_ascii());

        // Getting all casing combinations
        let combos = chunk.map(move |chr| {
            if chr.is_ascii_lowercase() {
                vec![chr, chr.to_ascii_uppercase()]
            } else {
                vec![chr]
            }
        });

        Some(
            combos
                .into_iter()
                .multi_cartesian_product()
                .inspect(|option| log::trace!("Testing value: {}", option.escape_ascii()))
                // Filtering our all strings that do not produce a value
                .filter_map(move |snippet| {
                    let mut result = [0_u8; 3];
                    BASE64_STANDARD.decode_slice(snippet, &mut result).ok()?;
                    Some(result)
                })
                .collect(),
        )
    }
}

impl<I: Iterator<Item = Vec<[u8; 3]>>, T> TypeIter<I, T>
where
    TypeIter<I, T>: Iterator<Item = Vec<[T; 3]>>,
{
    /// Filters out permutations based on a predicate
    ///
    /// Allows providing a default value if all permutations are filtered out
    pub fn filter_or<P>(
        self,
        other: [T; 3],
        predicate: P,
    ) -> std::iter::Map<Self, impl FnMut(<Self as Iterator>::Item) -> <Self as Iterator>::Item>
    where
        Self: Sized,
        [T; 3]: Clone,
        P: FnMut(&[T; 3]) -> bool + Clone,
    {
        self.map(move |variants| filter_variants(variants, other.clone(), predicate.clone()))
    }

    /// Filters out permutations based on a predicate
    ///
    /// Uses default value if all permutations are filtered out
    pub fn filter_or_default<P>(
        self,
        predicate: P,
    ) -> std::iter::Map<Self, impl FnMut(<Self as Iterator>::Item) -> <Self as Iterator>::Item>
    where
        Self: Sized,
        T: Copy + Default,
        P: FnMut(&[T; 3]) -> bool + Clone,
    {
        self.map(move |variants| filter_variants(variants, [T::default(); 3], predicate.clone()))
    }
}

impl<I: ExactSizeIterator<Item = Vec<[u8; 3]>>, T: Sized> ExactSizeIterator for TypeIter<I, T>
where
    TypeIter<I, T>: Iterator,
{
    fn len(&self) -> usize {
        self.iterator.len() / size_of::<T>()
    }
}

impl<I: Iterator<Item = Vec<[u8; 3]>>> Iterator for TypeIter<I, u16> {
    type Item = Vec<[u16; 3]>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = [(); size_of::<u16>()].map(|_| self.iterator.next().unwrap_or(vec![[0; 3]]));
        log::trace!("conversion: {:?}", chunk);

        // Check and fail if the first item is none
        if chunk[0] == vec![[0; 3]] {
            return None;
        }

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                // converting into single lines
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<6>()
                        .unwrap()
                })
                // converting unto u16
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u16_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: DoubleEndedIterator<Item = Vec<[u8; 3]>> + ExactSizeIterator<Item = Vec<[u8; 3]>>>
    DoubleEndedIterator for TypeIter<I, u16>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iterator.len() == 0 {
            return None;
        }

        // Iterator should always be divisible by its size, so anything not cleanly divisible is the first item
        let mut chunk = if self.iterator.len().is_multiple_of(size_of::<u16>()) {
            [(); size_of::<u16>()].map(|_| self.iterator.next().unwrap())
        } else {
            (0..size_of::<u16>())
                .map(move |ind| {
                    if ind < size_of::<u16>() - self.iterator.len() % size_of::<u16>() {
                        vec![[0_u8; 3]]
                    } else {
                        self.iterator.next_back().unwrap()
                    }
                })
                .collect_array::<2>()
                .unwrap()
        };

        chunk.reverse();

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<6>()
                        .unwrap()
                })
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u16_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: Iterator<Item = Vec<[u8; 3]>>> Iterator for TypeIter<I, u32> {
    type Item = Vec<[u32; 3]>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = [(); size_of::<u32>()].map(|_| self.iterator.next().unwrap_or(vec![[0; 3]]));
        log::trace!("conversion: {:?}", chunk);

        // Check and fail if the first item is none
        if chunk[0] == vec![[0; 3]] {
            return None;
        }

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                // converting into single lines
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<12>()
                        .unwrap()
                })
                // converting unto u16
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u32_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: DoubleEndedIterator<Item = Vec<[u8; 3]>> + ExactSizeIterator<Item = Vec<[u8; 3]>>>
    DoubleEndedIterator for TypeIter<I, u32>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iterator.len() == 0 {
            return None;
        }

        // Iterator should always be divisible by its size, so anything not cleanly divisible is the first item
        let mut chunk = if self.iterator.len().is_multiple_of(size_of::<u32>()) {
            [(); size_of::<u32>()].map(|_| self.iterator.next().unwrap())
        } else {
            (0..size_of::<u32>())
                .map(move |ind| {
                    if ind < size_of::<u32>() - self.iterator.len() % size_of::<u32>() {
                        vec![[0_u8; 3]]
                    } else {
                        self.iterator.next_back().unwrap()
                    }
                })
                .collect_array::<4>()
                .unwrap()
        };

        chunk.reverse();

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<12>()
                        .unwrap()
                })
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u32_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: Iterator<Item = Vec<[u8; 3]>>> Iterator for TypeIter<I, u64> {
    type Item = Vec<[u64; 3]>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = [(); size_of::<u64>()].map(|_| self.iterator.next().unwrap_or(vec![[0; 3]]));
        log::trace!("conversion: {:?}", chunk);

        // Check and fail if the first item is none
        if chunk[0] == vec![[0; 3]] {
            return None;
        }

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                // converting into single lines
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<24>()
                        .unwrap()
                })
                // converting unto u16
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u64_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: DoubleEndedIterator<Item = Vec<[u8; 3]>> + ExactSizeIterator<Item = Vec<[u8; 3]>>>
    DoubleEndedIterator for TypeIter<I, u64>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iterator.len() == 0 {
            return None;
        }

        // Iterator should always be divisible by its size, so anything not cleanly divisible is the first item
        let mut chunk = if self.iterator.len().is_multiple_of(size_of::<u64>()) {
            [(); size_of::<u64>()].map(|_| self.iterator.next().unwrap())
        } else {
            (0..size_of::<u64>())
                .map(move |ind| {
                    if ind < size_of::<u64>() - self.iterator.len() % size_of::<u64>() {
                        vec![[0_u8; 3]]
                    } else {
                        self.iterator.next_back().unwrap()
                    }
                })
                .collect_array::<8>()
                .unwrap()
        };

        chunk.reverse();

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<24>()
                        .unwrap()
                })
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u64_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: Iterator<Item = Vec<[u8; 3]>>> Iterator for TypeIter<I, u128> {
    type Item = Vec<[u128; 3]>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = [(); size_of::<u128>()].map(|_| self.iterator.next().unwrap_or(vec![[0; 3]]));
        log::trace!("conversion: {:?}", chunk);

        // Check and fail if the first item is none
        if chunk[0] == vec![[0; 3]] {
            return None;
        }

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                // converting into single lines
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<48>()
                        .unwrap()
                })
                // converting unto u16
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u128_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

impl<I: DoubleEndedIterator<Item = Vec<[u8; 3]>> + ExactSizeIterator<Item = Vec<[u8; 3]>>>
    DoubleEndedIterator for TypeIter<I, u128>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iterator.len() == 0 {
            return None;
        }

        // Iterator should always be divisible by its size, so anything not cleanly divisible is the first item
        let mut chunk = if self.iterator.len().is_multiple_of(size_of::<u128>()) {
            [(); size_of::<u128>()].map(|_| self.iterator.next().unwrap())
        } else {
            (0..size_of::<u128>())
                .map(move |ind| {
                    if ind < size_of::<u128>() - self.iterator.len() % size_of::<u128>() {
                        vec![[0_u8; 3]]
                    } else {
                        self.iterator.next_back().unwrap()
                    }
                })
                .collect_array::<16>()
                .unwrap()
        };

        chunk.reverse();

        // Unwrapping and combining
        Some(
            chunk
                .into_iter()
                .multi_cartesian_product()
                .map(|to_single| {
                    to_single
                        .into_iter()
                        .flatten()
                        .collect_array::<48>()
                        .unwrap()
                })
                .map(|conversion| {
                    let mut result_buffer = [0; 3];
                    LittleEndian::read_u128_into(&conversion, &mut result_buffer);
                    result_buffer
                })
                .collect(),
        )
    }
}

/// Provides and implements the base64 ingestion using the rayon library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use std::marker::PhantomData;

    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
        plumbing::{Producer, ProducerCallback, bridge},
    };

    use crate::base64_parser::{Base64Parser, TypeIter, filter_variants};

    /// Provides the traits to convert streams of lowercased base64 bytes into
    /// valid combinations of base64
    pub trait ParallelBruteforceBase64: IntoParallelIterator<Item = u8> {
        /// This will ingest the base64 string and produce all possible combinations
        /// in an iterative feed.
        ///
        /// ```rust
        /// use base64_bruteforcer_rs::base64_parser::rayon::ParallelBruteforceBase64;
        /// use rayon::iter::ParallelIterator;
        ///
        /// let example_base64 = b"sgvsbg8gd29ybgqh";
        ///
        /// let expected: Vec<Vec<[u8; 3]>> = vec![
        ///     vec![[178, 11, 236], [178, 11, 210], [178, 5, 108], [178, 5, 82], [176, 107, 236], [176, 107, 210], [176, 101, 108], [176, 101, 82], [74, 11, 236], [74, 11, 210], [74, 5, 108], [74, 5, 82], [72, 107, 236], [72, 107, 210], [72, 101, 108], [72, 101, 82]],
        ///     vec![[110, 15, 32], [110, 15, 6], [108, 111, 32], [108, 111, 6], [6, 15, 32], [6, 15, 6], [4, 111, 32], [4, 111, 6]],
        ///     vec![[119, 111, 114], [119, 111, 88], [15, 111, 114], [15, 111, 88]],
        ///     vec![[110, 10, 161], [110, 10, 135], [110, 4, 33], [110, 4, 7], [108, 106, 161], [108, 106, 135], [108, 100, 33], [108, 100, 7], [6, 10, 161], [6, 10, 135], [6, 4, 33], [6, 4, 7], [4, 106, 161], [4, 106, 135], [4, 100, 33], [4, 100, 7]]
        /// ];
        ///
        /// let result: Vec<Vec<[u8; 3]>> = example_base64.par_parse_base64().collect();
        ///
        /// assert!(result == expected, "Expected result {:?} did not match with {:?}", expected, result);
        /// ```
        fn par_parse_base64(self) -> ParBase64Parser<Self::Iter>;
    }

    impl<U: IntoParallelIterator<Item = u8>> ParallelBruteforceBase64 for U {
        fn par_parse_base64(self) -> ParBase64Parser<Self::Iter> {
            ParBase64Parser {
                iterator: self.into_par_iter(),
            }
        }
    }

    /// Iterative Parser for combined lowercased base64
    #[derive(Clone, Debug)]
    pub struct ParBase64Parser<I: ParallelIterator<Item = u8>> {
        /// Contains the lowercased base64 iterator
        iterator: I,
    }

    struct Base64Producer<P: Producer> {
        len: usize,
        base: P,
    }

    struct TypeProducer<P: Producer, T: Sized> {
        len: usize,
        base: P,
        _phantom: PhantomData<T>,
    }

    /// Converts bytes from [`ParBase64Parser`] into sets of the given type T
    ///
    /// Used in the [`convert_to_type`] method
    ///
    /// [`convert_to_type`]: ParBase64Parser::convert_to_type
    #[derive(Clone, Debug)]
    pub struct ParTypeIter<I: ParallelIterator<Item = Vec<[u8; 3]>>, T: Sized> {
        _phantom: PhantomData<T>,
        iterator: I,
    }

    impl<I: IndexedParallelIterator<Item = u8>> ParBase64Parser<I> {
        /// Converts a base64 permutations into a different type
        pub fn convert_to_type<T: Sized + Clone>(self) -> ParTypeIter<Self, T> {
            ParTypeIter {
                _phantom: PhantomData,
                iterator: self,
            }
        }

        /// Filters out permutations based on a predicate
        ///
        /// Allows providing a default value if all permutations are filtered out
        pub fn filter_or<P>(
            self,
            other: [u8; 3],
            predicate: P,
        ) -> rayon::iter::Map<Self, impl Fn(Vec<[u8; 3]>) -> Vec<[u8; 3]> + Send + Sync>
        where
            P: Fn(&[u8; 3]) -> bool + Clone + Send + Sync,
            Vec<[u8; 3]>: Send,
            Self: Sized + Send,
        {
            self.map(move |variants| filter_variants(variants, other, predicate.clone()))
        }

        /// Filters out permutations based on a predicate
        ///
        /// Uses default value if all permutations are filtered out
        pub fn filter_or_default<P>(
            self,
            predicate: P,
        ) -> rayon::iter::Map<Self, impl Fn(Vec<[u8; 3]>) -> Vec<[u8; 3]>>
        where
            P: Fn(&[u8; 3]) -> bool + Clone + Send + Sync,
        {
            self.map(move |variants| {
                filter_variants(variants, [u8::default(); 3], predicate.clone())
            })
        }
    }

    impl<I: IndexedParallelIterator<Item = u8>> ParallelIterator for ParBase64Parser<I> {
        type Item = Vec<[u8; 3]>;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<I> IndexedParallelIterator for ParBase64Parser<I>
    where
        I: IndexedParallelIterator<Item = u8>,
    {
        fn len(&self) -> usize {
            self.iterator.len().div_ceil(4)
        }

        fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
            bridge(self, consumer)
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            let len = self.iterator.len();
            return self.iterator.with_producer(Callback { len, callback });

            struct Callback<CB> {
                len: usize,
                callback: CB,
            }

            impl<CB> ProducerCallback<u8> for Callback<CB>
            where
                CB: ProducerCallback<Vec<[u8; 3]>>,
            {
                type Output = CB::Output;

                fn callback<P>(self, producer: P) -> Self::Output
                where
                    P: Producer<Item = u8>,
                {
                    let base = Base64Producer {
                        len: self.len,
                        base: producer,
                    };
                    self.callback.callback(base)
                }
            }
        }
    }

    impl<P: Producer<Item = u8>> Producer for Base64Producer<P> {
        type Item = Vec<[u8; 3]>;

        type IntoIter = Base64Parser<P::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            Base64Parser {
                iterator: self.base.into_iter(),
            }
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let elem_index = Ord::min(index * 4, self.len);
            let (left, right) = self.base.split_at(elem_index);
            (
                Base64Producer {
                    len: elem_index,
                    base: left,
                },
                Base64Producer {
                    len: self.len - elem_index,
                    base: right,
                },
            )
        }

        fn min_len(&self) -> usize {
            self.base.min_len().div_ceil(4)
        }

        fn max_len(&self) -> usize {
            self.base.max_len() / 4
        }
    }

    impl<I: ParallelIterator<Item = Vec<[u8; 3]>>, T> ParTypeIter<I, T>
    where
        ParTypeIter<I, T>: ParallelIterator<Item = Vec<[T; 3]>>,
    {
        /// Filters out permutations based on a predicate
        ///
        /// Allows providing a default value if all permutations are filtered out
        pub fn filter_or<P>(
            self,
            other: [T; 3],
            predicate: P,
        ) -> rayon::iter::Map<
            Self,
            impl Fn(<Self as ParallelIterator>::Item) -> <Self as ParallelIterator>::Item,
        >
        where
            Self: Sized + Send,
            T: Send + Sync,
            [T; 3]: Clone,
            P: Fn(&[T; 3]) -> bool + Clone + Send + Sync,
        {
            self.map(move |variants| filter_variants(variants, other.clone(), predicate.clone()))
        }

        /// Filters out permutations based on a predicate
        ///
        /// Uses default value if all permutations are filtered out
        pub fn filter_or_default<P>(
            self,
            predicate: P,
        ) -> rayon::iter::Map<
            Self,
            impl Fn(<Self as ParallelIterator>::Item) -> <Self as ParallelIterator>::Item,
        >
        where
            T: Copy + Default + Send,
            P: Fn(&[T; 3]) -> bool + Clone + Send + Sync,
        {
            self.map(move |variants| {
                filter_variants(variants, [T::default(); 3], predicate.clone())
            })
        }
    }

    impl<I: IndexedParallelIterator<Item = Vec<[u8; 3]>>, T: Send> ParallelIterator
        for ParTypeIter<I, T>
    where
        ParTypeIter<I, T>: IndexedParallelIterator,
    {
        type Item = Vec<[T; 3]>;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<I: IndexedParallelIterator<Item = Vec<[u8; 3]>>> IndexedParallelIterator
        for ParTypeIter<I, u16>
    {
        fn len(&self) -> usize {
            self.iterator.len() / size_of::<u16>()
        }

        fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
            bridge(self, consumer)
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            let len = self.iterator.len();
            return self.iterator.with_producer(Callback { len, callback });
            struct Callback<CB> {
                len: usize,
                callback: CB,
            }

            impl<CB> ProducerCallback<Vec<[u8; 3]>> for Callback<CB>
            where
                CB: ProducerCallback<Vec<[u16; 3]>>,
            {
                type Output = CB::Output;

                fn callback<P>(self, producer: P) -> Self::Output
                where
                    P: Producer<Item = Vec<[u8; 3]>>,
                {
                    let base = TypeProducer {
                        len: self.len,
                        base: producer,
                        _phantom: PhantomData,
                    };
                    self.callback.callback(base)
                }
            }
        }
    }

    impl<I: IndexedParallelIterator<Item = Vec<[u8; 3]>>> IndexedParallelIterator
        for ParTypeIter<I, u32>
    {
        fn len(&self) -> usize {
            self.iterator.len() / size_of::<u32>()
        }

        fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
            bridge(self, consumer)
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            let len = self.iterator.len();
            return self.iterator.with_producer(Callback { len, callback });
            struct Callback<CB> {
                len: usize,
                callback: CB,
            }

            impl<CB> ProducerCallback<Vec<[u8; 3]>> for Callback<CB>
            where
                CB: ProducerCallback<Vec<[u32; 3]>>,
            {
                type Output = CB::Output;

                fn callback<P>(self, producer: P) -> Self::Output
                where
                    P: Producer<Item = Vec<[u8; 3]>>,
                {
                    let base = TypeProducer {
                        len: self.len,
                        base: producer,
                        _phantom: PhantomData,
                    };
                    self.callback.callback(base)
                }
            }
        }
    }

    impl<I: IndexedParallelIterator<Item = Vec<[u8; 3]>>> IndexedParallelIterator
        for ParTypeIter<I, u64>
    {
        fn len(&self) -> usize {
            self.iterator.len() / size_of::<u64>()
        }

        fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
            bridge(self, consumer)
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            let len = self.iterator.len();
            return self.iterator.with_producer(Callback { len, callback });
            struct Callback<CB> {
                len: usize,
                callback: CB,
            }

            impl<CB> ProducerCallback<Vec<[u8; 3]>> for Callback<CB>
            where
                CB: ProducerCallback<Vec<[u64; 3]>>,
            {
                type Output = CB::Output;

                fn callback<P>(self, producer: P) -> Self::Output
                where
                    P: Producer<Item = Vec<[u8; 3]>>,
                {
                    let base = TypeProducer {
                        len: self.len,
                        base: producer,
                        _phantom: PhantomData,
                    };
                    self.callback.callback(base)
                }
            }
        }
    }

    impl<I: IndexedParallelIterator<Item = Vec<[u8; 3]>>> IndexedParallelIterator
        for ParTypeIter<I, u128>
    {
        fn len(&self) -> usize {
            self.iterator.len() / size_of::<u128>()
        }

        fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
            bridge(self, consumer)
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            let len = self.iterator.len();
            return self.iterator.with_producer(Callback { len, callback });
            struct Callback<CB> {
                len: usize,
                callback: CB,
            }

            impl<CB> ProducerCallback<Vec<[u8; 3]>> for Callback<CB>
            where
                CB: ProducerCallback<Vec<[u128; 3]>>,
            {
                type Output = CB::Output;

                fn callback<P>(self, producer: P) -> Self::Output
                where
                    P: Producer<Item = Vec<[u8; 3]>>,
                {
                    let base = TypeProducer {
                        len: self.len,
                        base: producer,
                        _phantom: PhantomData,
                    };
                    self.callback.callback(base)
                }
            }
        }
    }

    impl<P: Producer<Item = Vec<[u8; 3]>>, T: Send> Producer for TypeProducer<P, T>
    where
        TypeIter<P::IntoIter, T>: DoubleEndedIterator<Item = Vec<[T; 3]>>,
    {
        type Item = Vec<[T; 3]>;

        type IntoIter = TypeIter<P::IntoIter, T>;

        fn into_iter(self) -> Self::IntoIter {
            TypeIter {
                _phantom: PhantomData::<T>,
                iterator: self.base.into_iter(),
            }
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let elem_index = Ord::min(index * size_of::<T>(), self.len);
            let (left, right) = self.base.split_at(elem_index);
            (
                TypeProducer {
                    len: elem_index,
                    base: left,
                    _phantom: PhantomData,
                },
                TypeProducer {
                    len: self.len - elem_index,
                    base: right,
                    _phantom: PhantomData,
                },
            )
        }

        fn min_len(&self) -> usize {
            self.base.min_len().div_ceil(size_of::<T>())
        }

        fn max_len(&self) -> usize {
            self.base.max_len() / size_of::<T>()
        }
    }
}

/// Provides and implements base64 ingestion using the [`futures`] library to stream data in
#[cfg(feature = "async")]
pub mod r#async {
    use std::{
        marker::PhantomData,
        pin::Pin,
        task::{Context, Poll},
    };

    use base64::{Engine, prelude::BASE64_STANDARD};
    use byteorder::{ByteOrder, LittleEndian};
    use futures::{Stream, StreamExt, ready};
    use itertools::Itertools;
    use pin_project_lite::pin_project;

    use crate::base64_parser::filter_variants;

    /// Provides the traits to convert streams of lowercased base64 bytes into
    /// valid combinations of base64
    pub trait StreamBruteforceBase64: Stream<Item = u8> + Sized {
        /// This will ingest the base64 string and produce all possible combinations
        /// in an iterative feed.
        ///
        /// ```rust
        /// # futures::executor::block_on(async {
        /// use base64_bruteforcer_rs::base64_parser::r#async::StreamBruteforceBase64;
        /// use futures::stream::{iter, StreamExt};
        ///
        /// let example_base64 = iter(b"sgvsbg8gd29ybgqh".to_owned());
        ///
        /// let expected: Vec<Vec<[u8; 3]>> = vec![
        ///     vec![[178, 11, 236], [178, 11, 210], [178, 5, 108], [178, 5, 82], [176, 107, 236], [176, 107, 210], [176, 101, 108], [176, 101, 82], [74, 11, 236], [74, 11, 210], [74, 5, 108], [74, 5, 82], [72, 107, 236], [72, 107, 210], [72, 101, 108], [72, 101, 82]],
        ///     vec![[110, 15, 32], [110, 15, 6], [108, 111, 32], [108, 111, 6], [6, 15, 32], [6, 15, 6], [4, 111, 32], [4, 111, 6]],
        ///     vec![[119, 111, 114], [119, 111, 88], [15, 111, 114], [15, 111, 88]],
        ///     vec![[110, 10, 161], [110, 10, 135], [110, 4, 33], [110, 4, 7], [108, 106, 161], [108, 106, 135], [108, 100, 33], [108, 100, 7], [6, 10, 161], [6, 10, 135], [6, 4, 33], [6, 4, 7], [4, 106, 161], [4, 106, 135], [4, 100, 33], [4, 100, 7]]
        /// ];
        ///
        /// let result: Vec<Vec<[u8; 3]>> = example_base64.parse_base64_stream().collect().await;
        ///
        /// assert!(result == expected, "Expected result {:?} did not match with {:?}", expected, result);
        /// # });
        /// ```
        fn parse_base64_stream(self) -> Base64ParsingStream<Self>;
    }

    impl<U: StreamExt<Item = u8>> StreamBruteforceBase64 for U {
        fn parse_base64_stream(self) -> Base64ParsingStream<U> {
            Base64ParsingStream {
                stream: self,
                inner_collection: Vec::with_capacity(4),
            }
        }
    }

    pin_project! {
        /// Iterative Parser for combined lowercased base64
        #[derive(Clone, Debug)]
        #[must_use = "streams do nothing unless polled"]
        pub struct Base64ParsingStream<S: Stream<Item = u8>> {
            #[pin]
            stream: S,
            inner_collection: Vec<S::Item>,
        }
    }

    pin_project! {
        /// Converts bytes from [`Base64ParsingStream`] into sets of the given type T
        ///
        /// Used in the [`convert_to_type`] method
        ///
        /// [`convert_to_type`]: Base64ParsingStream::convert_to_type
        #[derive(Clone, Debug)]
        pub struct TypeStream<S: Stream<Item = Vec<[u8; 3]>>, T: Sized> {
            _phantom: PhantomData<T>,
            #[pin]
            stream: S,
            inner_collection: Vec<S::Item>,
        }
    }

    impl<S: Stream<Item = u8>> Base64ParsingStream<S> {
        /// Converts a base64 permutations into a different type
        pub fn convert_to_type<T: Sized + Clone>(self) -> TypeStream<Self, T>
        where
            Self: Sized,
        {
            TypeStream {
                _phantom: PhantomData,
                stream: self,
                inner_collection: Vec::with_capacity(size_of::<T>()),
            }
        }

        /// Filters out permutations based on a predicate
        ///
        /// Allows providing a default value if all permutations are filtered out
        pub fn filter_or<P>(
            self,
            other: [u8; 3],
            predicate: P,
        ) -> futures::stream::Map<Self, impl FnMut(<Self as Stream>::Item) -> <Self as Stream>::Item>
        where
            Self: Sized,
            P: FnMut(&[u8; 3]) -> bool + Clone,
        {
            self.map(move |variants| filter_variants(variants, other, predicate.clone()))
        }

        /// Filters out permutations based on a predicate
        ///
        /// Uses default value if all permutations are filtered out
        pub fn filter_or_default<P>(
            self,
            predicate: P,
        ) -> futures::stream::Map<Self, impl FnMut(<Self as Stream>::Item) -> <Self as Stream>::Item>
        where
            Self: Sized,
            P: FnMut(&[u8; 3]) -> bool + Clone,
        {
            self.map(move |variants| {
                filter_variants(variants, [u8::default(); 3], predicate.clone())
            })
        }
    }

    impl<S: Stream<Item = u8>> Stream for Base64ParsingStream<S> {
        type Item = Vec<[u8; 3]>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut this = self.as_mut().project();

            // Collecting items
            loop {
                match ready!(this.stream.as_mut().poll_next(cx)) {
                    // Push the item into the buffer and check whether it is full.
                    // If so, replace our buffer with a new and empty one and return
                    // the full one.
                    Some(byte) => {
                        this.inner_collection.push(byte);

                        // Stop polling if we have enough items in the inner collection
                        if this.inner_collection.len() >= 4 {
                            break;
                        }
                    },
                    // Fail early if collection is empty
                    None if this.inner_collection.is_empty() => return Poll::Ready(None),
                    None => break,
                }
            }

            // padding the vec if necessary
            while this.inner_collection.len() < 4 {
                this.inner_collection.push(b'=');
            }

            // Getting all casing combinations
            let combos = this
                .inner_collection
                .iter()
                .map(move |chr| {
                    if chr.is_ascii_lowercase() {
                        vec![*chr, chr.to_ascii_uppercase()]
                    } else {
                        vec![*chr]
                    }
                })
                .collect_array::<4>()
                .unwrap();

            // Items have been used in collection. Clearing it out for next use
            this.inner_collection.clear();

            Poll::Ready(Some(
                combos
                .into_iter()
                .multi_cartesian_product()
                .inspect(|option| log::trace!("Testing value: {}", option.escape_ascii()))
                // Filtering our all strings that do not produce a value
                .filter_map(move |snippet| {
                    let mut result = [0_u8; 3];
                    BASE64_STANDARD.decode_slice(snippet, &mut result).ok()?;
                    Some(result)
                })
                .collect(),
            ))
        }
    }

    impl<S: Stream<Item = Vec<[u8; 3]>>, T> TypeStream<S, T>
    where
        TypeStream<S, T>: Stream<Item = Vec<[T; 3]>>,
    {
        /// Filters out permutations based on a predicate
        ///
        /// Allows providing a default value if all permutations are filtered out
        pub fn filter_or<P>(
            self,
            other: [T; 3],
            predicate: P,
        ) -> futures::stream::Map<Self, impl FnMut(<Self as Stream>::Item) -> <Self as Stream>::Item>
        where
            Self: Sized,
            [T; 3]: Clone,
            P: FnMut(&[T; 3]) -> bool + Clone,
        {
            self.map(move |variants| filter_variants(variants, other.clone(), predicate.clone()))
        }

        /// Filters out permutations based on a predicate
        ///
        /// Uses default value if all permutations are filtered out
        pub fn filter_or_default<P>(
            self,
            predicate: P,
        ) -> futures::stream::Map<Self, impl FnMut(<Self as Stream>::Item) -> <Self as Stream>::Item>
        where
            T: Copy + Default,
            Self: Sized,
            P: FnMut(&[T; 3]) -> bool + Clone,
        {
            self.map(move |variants| {
                filter_variants(variants, [T::default(); 3], predicate.clone())
            })
        }
    }

    impl<S: Stream<Item = Vec<[u8; 3]>>> Stream for TypeStream<S, u16> {
        type Item = Vec<[u16; 3]>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut this = self.as_mut().project();

            // Collecting items
            loop {
                match ready!(this.stream.as_mut().poll_next(cx)) {
                    // Push the item into the buffer and check whether it is full.
                    // If so, replace our buffer with a new and empty one and return
                    // the full one.
                    Some(byte) => {
                        this.inner_collection.push(byte);

                        // Stop polling if we have enough items in the inner collection
                        if this.inner_collection.len() >= size_of::<u16>() {
                            break;
                        }
                    },
                    // Fail early if collection is empty
                    None if this.inner_collection.is_empty() => return Poll::Ready(None),
                    None => break,
                }
            }

            // padding the vec if necessary
            while this.inner_collection.len() < size_of::<u16>() {
                this.inner_collection.push(vec![[0; 3]]);
            }

            // Unwrapping and combining
            let final_buffer = this.inner_collection
                    .iter()
                    .multi_cartesian_product()
                    // converting into single lines
                    .map(|to_single| {
                        to_single
                            .into_iter()
                            .flatten()
                            .cloned()
                            .collect_array::<6>()
                            .unwrap()
                    })
                    // converting unto u16
                    .map(|conversion| {
                        let mut result_buffer = [0; 3];
                        LittleEndian::read_u16_into(&conversion, &mut result_buffer);
                        result_buffer
                    })
                    .collect();

            // Internal buffer used as intended. Clearing its data
            this.inner_collection.clear();

            Poll::Ready(Some(final_buffer))
        }
    }

    impl<S: Stream<Item = Vec<[u8; 3]>>> Stream for TypeStream<S, u32> {
        type Item = Vec<[u32; 3]>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut this = self.as_mut().project();

            // Collecting items
            loop {
                match ready!(this.stream.as_mut().poll_next(cx)) {
                    // Push the item into the buffer and check whether it is full.
                    // If so, replace our buffer with a new and empty one and return
                    // the full one.
                    Some(byte) => {
                        this.inner_collection.push(byte);

                        // Stop polling if we have enough items in the inner collection
                        if this.inner_collection.len() >= size_of::<u32>() {
                            break;
                        }
                    },
                    // Fail early if collection is empty
                    None if this.inner_collection.is_empty() => return Poll::Ready(None),
                    None => break,
                }
            }

            // padding the vec if necessary
            while this.inner_collection.len() < size_of::<u32>() {
                this.inner_collection.push(vec![[0; 3]]);
            }

            // Unwrapping and combining
            let final_buffer = this.inner_collection
                    .iter()
                    .multi_cartesian_product()
                    // converting into single lines
                    .map(|to_single| {
                        to_single
                            .into_iter()
                            .flatten()
                            .cloned()
                            .collect_array::<6>()
                            .unwrap()
                    })
                    // converting unto u16
                    .map(|conversion| {
                        let mut result_buffer = [0; 3];
                        LittleEndian::read_u32_into(&conversion, &mut result_buffer);
                        result_buffer
                    })
                    .collect();

            // Internal buffer used as intended. Clearing its data
            this.inner_collection.clear();

            Poll::Ready(Some(final_buffer))
        }
    }

    impl<S: Stream<Item = Vec<[u8; 3]>>> Stream for TypeStream<S, u64> {
        type Item = Vec<[u64; 3]>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut this = self.as_mut().project();

            // Collecting items
            loop {
                match ready!(this.stream.as_mut().poll_next(cx)) {
                    // Push the item into the buffer and check whether it is full.
                    // If so, replace our buffer with a new and empty one and return
                    // the full one.
                    Some(byte) => {
                        this.inner_collection.push(byte);

                        // Stop polling if we have enough items in the inner collection
                        if this.inner_collection.len() >= size_of::<u64>() {
                            break;
                        }
                    },
                    // Fail early if collection is empty
                    None if this.inner_collection.is_empty() => return Poll::Ready(None),
                    None => break,
                }
            }

            // padding the vec if necessary
            while this.inner_collection.len() < size_of::<u64>() {
                this.inner_collection.push(vec![[0; 3]]);
            }

            // Unwrapping and combining
            let final_buffer = this.inner_collection
                    .iter()
                    .multi_cartesian_product()
                    // converting into single lines
                    .map(|to_single| {
                        to_single
                            .into_iter()
                            .flatten()
                            .cloned()
                            .collect_array::<12>()
                            .unwrap()
                    })
                    // converting unto u16
                    .map(|conversion| {
                        let mut result_buffer = [0; 3];
                        LittleEndian::read_u64_into(&conversion, &mut result_buffer);
                        result_buffer
                    })
                    .collect();

            // Internal buffer used as intended. Clearing its data
            this.inner_collection.clear();

            Poll::Ready(Some(final_buffer))
        }
    }

    impl<S: Stream<Item = Vec<[u8; 3]>>> Stream for TypeStream<S, u128> {
        type Item = Vec<[u128; 3]>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let mut this = self.as_mut().project();

            // Collecting items
            loop {
                match ready!(this.stream.as_mut().poll_next(cx)) {
                    // Push the item into the buffer and check whether it is full.
                    // If so, replace our buffer with a new and empty one and return
                    // the full one.
                    Some(byte) => {
                        this.inner_collection.push(byte);

                        // Stop polling if we have enough items in the inner collection
                        if this.inner_collection.len() >= size_of::<u128>() {
                            break;
                        }
                    },
                    // Fail early if collection is empty
                    None if this.inner_collection.is_empty() => return Poll::Ready(None),
                    None => break,
                }
            }

            // padding the vec if necessary
            while this.inner_collection.len() < size_of::<u128>() {
                this.inner_collection.push(vec![[0; 3]]);
            }

            // Unwrapping and combining
            let final_buffer = this.inner_collection
                    .iter()
                    .multi_cartesian_product()
                    // converting into single lines
                    .map(|to_single| {
                        to_single
                            .into_iter()
                            .flatten()
                            .cloned()
                            .collect_array::<48>()
                            .unwrap()
                    })
                    // converting unto u16
                    .map(|conversion| {
                        let mut result_buffer = [0; 3];
                        LittleEndian::read_u128_into(&conversion, &mut result_buffer);
                        result_buffer
                    })
                    .collect();

            // Internal buffer used as intended. Clearing its data
            this.inner_collection.clear();

            Poll::Ready(Some(final_buffer))
        }
    }
}
