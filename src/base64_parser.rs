//! This module contains the structs used to convert utf8 and utf16 to
//! combinations of potential valid characters
use base64::{Engine, prelude::BASE64_STANDARD};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;

/// Bruteforces all lowercased base64 combinations to determine the correct
/// base64 string
#[derive(Debug, Clone)]
pub struct Base64Bruteforcer<T> {
    /// Contains the permutation schema generated from collecting combinations
    /// from the bruteforcer
    pub schema: Vec<Vec<Vec<T>>>,
}

impl<T> Default for Base64Bruteforcer<T> {
    fn default() -> Self {
        Self {
            schema: Default::default(),
        }
    }
}

impl<T> Base64Bruteforcer<T> {
    /// Calculates the number of permutatable combinations this bruteforcers
    /// schema can produce
    pub fn permutations(&mut self) -> f64 {
        return self
            .schema
            .iter()
            .map(|section| section.len())
            .map(|size| size as f64)
            .product();
    }
}
impl Base64Bruteforcer<u8> {
    /// Takes a base64 slice (4 characters) and creates a vector containing the valid
    /// combinations of 3 characters. This can be useful if multiple valid sets
    /// appear and you need to compare sets to get the correct values
    fn get_valid_combinations(b64_slice: &[u8]) -> impl Iterator<Item = Vec<u8>> {
        let set_list = b64_slice.iter().map(|c| {
            if c.is_ascii_lowercase() {
                vec![*c, c.to_ascii_uppercase()]
            } else {
                vec![*c]
            }
        });

        return set_list
            .multi_cartesian_product()
            .filter_map(|combo| BASE64_STANDARD.decode(combo).ok());
    }

    /// Collects a vec of every possible valid combinations turning every 4
    /// base64 characters into multiple combinations of 3 characters. It will
    /// automatically filter out combinations that are not ascii readable, so
    /// any converted binaries will not be able to be deciphered with this
    /// function.
    pub fn collect_combinations(&mut self, b64_string: &[u8]) {
        self.schema = b64_string
            .iter()
            .chunks(4)
            .into_iter()
            .map(|piece| piece.map(|c| c.to_owned()).collect_vec())
            .map(|piece| {
                Self::get_valid_combinations(piece.as_slice())
                    .filter(|check_utf8| String::from_utf8(check_utf8.to_owned()).is_ok())
                    .filter(|check_ascii| check_ascii.is_ascii())
                    // Checking that variations do not have control characters in them
                    .filter(|check_control_char| {
                        check_control_char
                            .iter()
                            .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == 0)
                    })
                    .collect_vec()
            })
            .map(|replace_empty| {
                if replace_empty.is_empty() {
                    vec![b"???".to_vec()]
                } else {
                    replace_empty
                }
            })
            .collect();
    }

    /// Converts utf8 bytes into a rust string. This ends up more helpful when
    /// doing NLP processing on the lines created
    pub fn convert_to_string(&mut self) -> Vec<Vec<String>> {
        return self
            .schema
            .iter()
            .map(|sections| {
                sections
                    .iter()
                    .map(|variations| String::from_utf8(variations.to_owned()).unwrap())
                    .collect_vec()
            })
            .collect_vec();
    }

    /// Turns the schema into an iterator of every possible combination
    pub fn produce_lines(&mut self) -> impl Iterator<Item = Vec<u8>> {
        return self
            .schema
            .clone()
            .into_iter()
            .multi_cartesian_product()
            .map(|section| section.concat());
    }
}

impl Base64Bruteforcer<u16> {
    /// Takes a base64 slice (8 characters) and creates a vector containing the valid
    /// combinations of 3 characters. This can be useful if multiple valid sets
    /// appear and you need to compare sets to get the correct values
    fn get_valid_combinations(b64_slice: &[u8]) -> impl Iterator<Item = Vec<u16>> {
        return Base64Bruteforcer::<u8>::get_valid_combinations(b64_slice)
            // Checking that output given is actual utf16. Array will always be
            // divisible by 2
            .filter(|correct_length| correct_length.len() % 2 == 0)
            .map(|convert_utf16le| {
                log::debug!(
                    "convert.content: {:?}",
                    convert_utf16le.escape_ascii().to_string()
                );
                let mut output: Vec<u16> = vec![0].repeat(convert_utf16le.len() / 2);
                // https://stackoverflow.com/a/50244328
                LittleEndian::read_u16_into(convert_utf16le.as_slice(), output.as_mut_slice());
                output
            });
    }

    /// Collects a vec of every possible valid combinations turning every 8
    /// base64 characters into multiple combinations of 3 characters. It will
    /// automatically filter out combinations that are not ascii readable, so
    /// any converted binaries will not be able to be deciphered with this
    /// function.
    pub fn collect_combinations(&mut self, b64_string: &[u8]) {
        self.schema = b64_string
            .iter()
            .chunks(8)
            .into_iter()
            .map(|piece| piece.map(|c| c.to_owned()).collect_vec())
            .map(|piece| {
                Self::get_valid_combinations(piece.as_slice())
                    // Checking that variation is valid UTF-16
                    .filter(|check_ascii| {
                        String::from_utf16(check_ascii.as_slice()).is_ok_and(|s| s.is_ascii())
                    })
                    // Checking that variations do not have control characters in them
                    .filter(|check_control| {
                        String::from_utf16(check_control.as_slice()).is_ok_and(|s| {
                            s.chars()
                                .all(|c| c.is_ascii_graphic() || c.is_whitespace() || c == '\0')
                        })
                    })
                    .collect_vec()
            })
            .map(|replace_empty| {
                if replace_empty.is_empty() {
                    vec![vec![b'?' as u16].repeat(3)]
                } else {
                    replace_empty
                }
            })
            .collect();
    }

    /// Converts utf8 bytes into a rust string. This ends up more helpful when
    /// doing NLP processing on the lines created
    pub fn convert_to_string(&mut self) -> Vec<Vec<String>> {
        return self
            .schema
            .iter()
            .map(|section| {
                section
                    .iter()
                    .map(|variation| String::from_utf16(variation.as_slice()).unwrap())
                    .collect_vec()
            })
            .collect_vec();
    }

    /// Turns the schema into an iterator of every possible combination
    pub fn produce_lines(&mut self) -> impl Iterator<Item = Vec<u16>> {
        return self
            .schema
            .clone()
            .into_iter()
            .multi_cartesian_product()
            .map(|section| section.concat());
    }
}
