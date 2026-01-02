![minimum rustc 1.88](https://img.shields.io/badge/rustc-1.88+-red.svg)
[![build status](https://github.com/rad10/base64-decoder-rs/workflows/Rust/badge.svg)](https://github.com/rad10/base64-decoder-rs/actions)

# Base64 Decoder

This small project is designed to brute-force a base64 string that has been
lowercased.

This library takes advantage of knowledge understood about the underlying data
inside of the lowercased Base64 string in order to determine accurate results
in as few guesses as possible.

In the sake of text, it can reduce possible values by checking only valid utf8
encoding and throwing out unlikely characters (such as emojis)
