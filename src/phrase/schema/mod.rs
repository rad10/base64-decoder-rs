//! This module contains all the data types provided by this library

use crate::phrase::schema::snippet::Permutation;

pub mod snippet;
pub mod variation;

// Makes sense to handle it on a generic level since its literally just
// calculating the cartesian product
impl<T> Permutation for [Vec<T>] {
    fn permutations(&self) -> f64 {
        self.iter()
            .map(move |section| section.len() as f64)
            .product()
    }
}
