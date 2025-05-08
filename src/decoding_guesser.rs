use base64::{Engine, prelude::BASE64_STANDARD};
use itertools::Itertools;

/// Defines ascii characters that are disapproved since theyre control
/// characters and will not be in a real prompt
pub const DISALLOWED_ASCII: [u8; 29] = [
    1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 127,
];

/// Takes a base64 slice (4 characters) and creates a vector containing the valid
/// combinations of 3 characters. This can be useful if multiple valid sets
/// appear and you need to compare sets to get the correct values
pub fn get_valid_combinations(b64_slice: &[u8]) -> impl Iterator<Item = Vec<u8>> {
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
