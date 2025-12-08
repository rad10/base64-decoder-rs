//! This module contains the structs used to convert utf8 and utf16 to
//! combinations of potential valid characters

use base64::{Engine, prelude::BASE64_STANDARD};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;

use crate::phrase::schema::Phrase;

/// Bruteforces all combinations of a lowercase base64 string and converts it
/// into the given struct. Provides [`parse_base64`] to create a new version of
/// [`Self`]
///
/// [`Self`]: FromBase64
/// [`parse_base64`]: FromBase64::parse_base64
pub trait FromBase64 {
    type Type: Sized + Clone + From<u8>;

    /// Denotes the number of base64 characters to process to get 3 standard characters
    ///
    /// This is easily calculated by dividing the number of bits in [`Type`] by 2, or by multiplying the number of bytes in [`Type`] by 4
    ///
    /// [`Type`]: FromBase64::Type
    const CHARS: usize = size_of::<Self::Type>() * 4;

    /// A helper function used to convert resulting bytes into the desired type.
    /// This should panic if the given buffer does not evenly fit the types size
    fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type>;

    /// This function will be used to filter valid characters in a decoded
    /// base64 snippet if a custom function isn't provided by the user
    fn default_validation(piece: &Vec<Self::Type>) -> bool;

    /// Parses a collection of [`Self::Type`] and produces [`Self`] from the
    /// combinations
    ///
    /// [`Self::Type`]: FromBase64::Type
    /// [`Self`]: FromBase64
    fn parse_base64(
        b64_string: impl IntoIterator<Item = u8>,
        validator: Option<fn(&Vec<Self::Type>) -> bool>,
    ) -> Self
    where
        Self: FromIterator<Vec<Vec<Self::Type>>>,
    {
        Self::from_iter(
            b64_string
                .into_iter()
                // Splitting iterator into chunks of the types size
                .chunks(Self::CHARS)
                .into_iter()
                // Converting chunks into variations
                .map(move |piece| {
                    piece
                        .map(move |c| {
                            if c.is_ascii_lowercase() {
                                vec![c.to_ascii_uppercase(), c]
                            } else {
                                vec![c]
                            }
                        })
                        .multi_cartesian_product()
                        .inspect(|option| log::debug!("Testing value: {}", option.escape_ascii()))
                        // Filtering our all scrings that do not produce a value
                        .filter_map(move |combo| BASE64_STANDARD.decode(combo).ok())
                        // Converting bytes into type
                        .map(Self::convert_bytes_to_type)
                        // Running either the custom filter or the default filter
                        .filter(validator.unwrap_or(Self::default_validation))
                })
                // Setting default value for each segment if they're empty
                .map(move |final_collect| {
                    let check_empty = final_collect.collect::<Vec<Vec<Self::Type>>>();
                    if check_empty.is_empty() {
                        vec![vec![Self::Type::from(b'?'); 3]]
                    } else {
                        check_empty
                    }
                }),
        )
    }
}

impl FromBase64 for Phrase<Vec<u8>> {
    type Type = u8;

    fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
        // Skipping assertion since the types are the same
        bytes
    }

    fn default_validation(piece: &Vec<Self::Type>) -> bool {
        piece
            .iter()
            // Checking each character to ensure they're readable characters
            .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == b'\0')
    }
}

impl FromBase64 for Phrase<Vec<u16>> {
    type Type = u16;

    fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
        // Ensure that the length fits the given size
        // This should only have a chance to trigger if someone is manually
        // using it
        assert!(
            bytes.len().is_multiple_of(size_of::<Self::Type>()),
            "Bytes given cannot fit into given type"
        );

        // Since bytes guarantees to fit, start packing it in
        let mut output = vec![0_u16; bytes.len() / size_of::<Self::Type>()];
        LittleEndian::read_u16_into(bytes.as_slice(), &mut output);
        output
    }

    fn default_validation(piece: &Vec<Self::Type>) -> bool {
        // Need to check if it can convert into a string
        String::from_utf16(piece).is_ok_and(move |s| {
            s.chars()
                // Checking each character to ensure they're readable characters
                .all(move |c| c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0')
        })
    }
}

impl FromBase64 for Phrase<Vec<u32>> {
    type Type = u32;

    fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
        // Ensure that the length fits the given size
        // This should only have a chance to trigger if someone is manually
        // using it
        assert!(
            bytes.len().is_multiple_of(size_of::<Self::Type>()),
            "Bytes given cannot fit into given type"
        );

        // Since bytes guarantees to fit, start packing it in
        let mut output = vec![0_u32; bytes.len() / size_of::<Self::Type>()];
        LittleEndian::read_u32_into(bytes.as_slice(), &mut output);
        output
    }

    fn default_validation(piece: &Vec<Self::Type>) -> bool {
        // Check to see if this can be converted into a string
        piece.iter().all(|u| {
            // Fail if any of the u32 are invalid unicode
            char::from_u32(*u)
                .is_some_and(move |c| c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0')
        })
    }
}

/// Provides and implements the base64 ingestion using the rayon library to speed up processes
#[cfg(feature = "rayon")]
pub mod rayon {
    use base64::{Engine, prelude::BASE64_STANDARD};
    use byteorder::{ByteOrder, LittleEndian};
    use itertools::Itertools;
    use rayon::{
        iter::{
            FromParallelIterator, IndexedParallelIterator, IntoParallelRefIterator,
            ParallelIterator,
        },
        str::ParallelString,
    };

    use crate::phrase::schema::Phrase;

    /// Bruteforces all combinations of a lowercase base64 string and converts it
    /// into the given struct. Provides [`parse_base64`] to create a new version of
    /// [`Self`]
    ///
    /// This is similar to [`FromBase64`] except this trait implements the same
    /// functions and features using the [`rayon`] library
    ///
    /// [`Self`]: FromParBase64
    /// [`parse_base64`]: FromParBase64::parse_base64
    /// [`FromBase64`]: super::FromBase64
    pub trait FromParBase64 {
        type Type: Sized + Clone + From<u8> + Send;

        /// Denotes the number of base64 characters to process to get 3 standard characters
        ///
        /// This is easily calculated by dividing the number of bits in [`Type`] by 2, or by multiplying the number of bytes in [`Type`] by 4
        ///
        /// [`Type`]: FromParBase64::Type
        const CHARS: usize = size_of::<Self::Type>() * 4;

        /// A helper function used to convert resulting bytes into the desired type.
        /// This should panic if the given buffer does not evenly fit the types size
        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type>;

        /// This function will be used to filter valid characters in a decoded
        /// base64 snippet if a custom function isn't provided by the user
        fn default_validation(piece: &Vec<Self::Type>) -> bool;

        /// Parses a collection of [`Self::Type`] and produces [`Self`] from the
        /// combinations
        ///
        /// [`Self::Type`]: FromParBase64::Type
        /// [`Self`]: FromParBase64
        fn par_parse_base64(
            b64_string: impl IndexedParallelIterator<Item = u8>,
            validator: Option<fn(&Vec<Self::Type>) -> bool>,
        ) -> Self
        where
            Self: Sized + FromParallelIterator<Vec<Vec<Self::Type>>>,
        {
            Self::from_par_iter(
                b64_string
                    // Splitting iterator into chunks of the types size
                    .chunks(Self::CHARS)
                    // Converting chunks into variations
                    .map(move |piece| {
                        piece
                            .into_iter()
                            .map(move |c| {
                                if c.is_ascii_lowercase() {
                                    vec![c.to_ascii_uppercase(), c]
                                } else {
                                    vec![c]
                                }
                            })
                            .multi_cartesian_product()
                            .inspect(|option| {
                                log::debug!("Testing value: {}", option.escape_ascii())
                            })
                            // Filtering our all scrings that do not produce a value
                            .filter_map(move |combo| BASE64_STANDARD.decode(combo).ok())
                            // Converting bytes into type
                            .map(Self::convert_bytes_to_type)
                            // Running either the custom filter or the default filter
                            .filter(validator.unwrap_or(Self::default_validation))
                    })
                    // Setting default value for each segment if they're empty
                    .map(move |final_collect| {
                        let check_empty = final_collect.collect::<Vec<Vec<Self::Type>>>();
                        if check_empty.is_empty() {
                            vec![vec![Self::Type::from(b'?'); 3]]
                        } else {
                            check_empty
                        }
                    }),
            )
        }
    }

    impl FromParBase64 for Phrase<Vec<u8>> {
        type Type = u8;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Skipping assertion since the types are the same
            bytes
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            piece
                .par_iter()
                // Checking each character to ensure they're readable characters
                .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == b'\0')
        }
    }

    impl FromParBase64 for Phrase<Vec<u16>> {
        type Type = u16;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Ensure that the length fits the given size
            // This should only have a chance to trigger if someone is manually
            // using it
            assert!(
                bytes.len().is_multiple_of(size_of::<Self::Type>()),
                "Bytes given cannot fit into given type"
            );

            // Since bytes guarantees to fit, start packing it in
            let mut output = vec![0_u16; bytes.len() / size_of::<Self::Type>()];
            LittleEndian::read_u16_into(bytes.as_slice(), &mut output);
            output
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            // Need to check if it can convert into a string
            String::from_utf16(piece).is_ok_and(move |s| {
                s.par_chars()
                    // Checking each character to ensure they're readable characters
                    .all(move |c| c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0')
            })
        }
    }

    impl FromParBase64 for Phrase<Vec<u32>> {
        type Type = u32;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Ensure that the length fits the given size
            // This should only have a chance to trigger if someone is manually
            // using it
            assert!(
                bytes.len().is_multiple_of(size_of::<Self::Type>()),
                "Bytes given cannot fit into given type"
            );

            // Since bytes guarantees to fit, start packing it in
            let mut output = vec![0_u32; bytes.len() / size_of::<Self::Type>()];
            LittleEndian::read_u32_into(bytes.as_slice(), &mut output);
            output
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            // Check to see if this can be converted into a string
            piece.par_iter().all(|u| {
                // Fail if any of the u32 are invalid unicode
                char::from_u32(*u).is_some_and(move |c| {
                    c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                })
            })
        }
    }
}

/// Provides and implements base64 ingestion using the [`futures`] library to stream data in
#[cfg(feature = "async")]
pub mod r#async {
    use base64::{Engine, prelude::BASE64_STANDARD};
    use byteorder::{ByteOrder, LittleEndian};
    use futures::{Stream, StreamExt};
    use itertools::Itertools;

    use crate::phrase::schema::Phrase;

    /// Bruteforces all combinations of a lowercase base64 string and converts it
    /// into the given struct. Provides [`parse_base64`] to create a new version of
    /// [`Self`]
    ///
    /// This is similar to [`FromBase64`] except this trait implements the same
    /// functions and features using [`futures`] streams
    ///
    /// [`Self`]: FromBase64Stream
    /// [`parse_base64`]: FromBase64Stream::parse_base64
    /// [`FromBase64`]: super::FromBase64
    pub trait FromBase64Stream {
        type Type: Sized + Clone + From<u8> + Send;

        /// Denotes the number of base64 characters to process to get 3 standard characters
        ///
        /// This is easily calculated by dividing the number of bits in [`Type`] by 2, or by multiplying the number of bytes in [`Type`] by 4
        ///
        /// [`Type`]: FromBase64Stream::Type
        const CHARS: usize = size_of::<Self::Type>() * 4;

        /// A helper function used to convert resulting bytes into the desired type.
        /// This should panic if the given buffer does not evenly fit the types size
        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type>;

        /// This function will be used to filter valid characters in a decoded
        /// base64 snippet if a custom function isn't provided by the user
        fn default_validation(piece: &Vec<Self::Type>) -> bool;

        /// Parses a collection of [`Self::Type`] and produces [`Self`] from the
        /// combinations
        ///
        /// [`Self::Type`]: FromBase64Stream::Type
        /// [`Self`]: FromBase64Stream
        fn parse_base64_stream(
            b64_string: impl Stream<Item = u8>,
            validator: Option<fn(&Vec<Self::Type>) -> bool>,
        ) -> impl Future<Output = Self>
        where
            Self: FromIterator<Vec<Vec<Self::Type>>>,
        {
            async move {
                Self::from_iter(
                    b64_string
                        // Splitting iterator into chunks of the types size
                        .chunks(Self::CHARS)
                        // Converting chunks into variations
                        .map(move |piece| {
                            piece
                                .into_iter()
                                .map(move |c| {
                                    if c.is_ascii_lowercase() {
                                        vec![c.to_ascii_uppercase(), c]
                                    } else {
                                        vec![c]
                                    }
                                })
                                .multi_cartesian_product()
                                .inspect(|option| {
                                    log::debug!("Testing value: {}", option.escape_ascii())
                                })
                                // Filtering our all scrings that do not produce a value
                                .filter_map(move |combo| BASE64_STANDARD.decode(combo).ok())
                                // Converting bytes into type
                                .map(Self::convert_bytes_to_type)
                                // Running either the custom filter or the default filter
                                .filter(validator.unwrap_or(Self::default_validation))
                        })
                        // Setting default value for each segment if they're empty
                        .map(move |final_collect| {
                            let check_empty = final_collect.collect::<Vec<Vec<Self::Type>>>();
                            if check_empty.is_empty() {
                                vec![vec![Self::Type::from(b'?'); 3]]
                            } else {
                                check_empty
                            }
                        })
                        .collect::<Vec<Vec<Vec<Self::Type>>>>()
                        .await,
                )
            }
        }
    }

    impl FromBase64Stream for Phrase<Vec<u8>> {
        type Type = u8;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Skipping assertion since the types are the same
            bytes
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            piece
                .iter()
                // Checking each character to ensure they're readable characters
                .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == b'\0')
        }
    }

    impl FromBase64Stream for Phrase<Vec<u16>> {
        type Type = u16;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Ensure that the length fits the given size
            // This should only have a chance to trigger if someone is manually
            // using it
            assert!(
                bytes.len().is_multiple_of(size_of::<Self::Type>()),
                "Bytes given cannot fit into given type"
            );

            // Since bytes guarantees to fit, start packing it in
            let mut output = vec![0_u16; bytes.len() / size_of::<Self::Type>()];
            LittleEndian::read_u16_into(bytes.as_slice(), &mut output);
            output
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            // Need to check if it can convert into a string
            String::from_utf16(piece).is_ok_and(move |s| {
                s.chars()
                    // Checking each character to ensure they're readable characters
                    .all(move |c| c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0')
            })
        }
    }

    impl FromBase64Stream for Phrase<Vec<u32>> {
        type Type = u32;

        fn convert_bytes_to_type(bytes: Vec<u8>) -> Vec<Self::Type> {
            // Ensure that the length fits the given size
            // This should only have a chance to trigger if someone is manually
            // using it
            assert!(
                bytes.len().is_multiple_of(size_of::<Self::Type>()),
                "Bytes given cannot fit into given type"
            );

            // Since bytes guarantees to fit, start packing it in
            let mut output = vec![0_u32; bytes.len() / size_of::<Self::Type>()];
            LittleEndian::read_u32_into(bytes.as_slice(), &mut output);
            output
        }

        fn default_validation(piece: &Vec<Self::Type>) -> bool {
            // Check to see if this can be converted into a string
            piece.iter().all(|u| {
                // Fail if any of the u32 are invalid unicode
                char::from_u32(*u).is_some_and(move |c| {
                    c.is_ascii_graphic() || c.is_ascii_whitespace() || c == '\0'
                })
            })
        }
    }
}
