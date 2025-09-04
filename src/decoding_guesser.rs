use base64::{Engine, prelude::BASE64_STANDARD};
use itertools::Itertools;

/// Defines ascii characters that are disapproved since theyre control
/// characters and will not be in a real prompt
pub const DISALLOWED_ASCII: [u8; 29] = [
    1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 127,
];

/// The object that generates all combinations
pub struct Base64Bruteforcer<T> {
    pub combinations: Vec<Vec<Vec<T>>>,
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

    return set_list.multi_cartesian_product().filter_map(|combo| {
        match BASE64_STANDARD.decode(combo) {
            Ok(b64) => Some(b64),
            Err(_) => None,
        }
    });
}

    /// Collects a vec of every possible valid combinations turning every 4
    /// base64 characters into multiple combinations of 3 characters. It will
    /// automatically filter out combinations that are not ascii readable, so
    /// any converted binaries will not be able to be deciphered with this
    /// function.
    pub fn collect_combinations(b64_string: &[u8]) -> Vec<Vec<Vec<u8>>> {
        return b64_string
            .iter()
            .chunks(4)
            .into_iter()
            .map(|piece| piece.map(|c| c.to_owned()).collect_vec())
            .map(|piece| {
                Self::get_valid_combinations(piece.as_slice())
                    .filter(|check_ascii| check_ascii.is_ascii())
                    .filter(|check_control_char| {
                        check_control_char
                            .iter()
                            .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace() || *c == 0)
                    })
                    .collect_vec()
            }).map(|replace_empty| {
                if replace_empty.is_empty(){
                    vec![b"???".to_vec()]
                } else {
                    replace_empty
                }
            }).collect();
    }
}
