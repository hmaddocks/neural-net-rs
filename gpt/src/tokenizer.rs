//! Character-level tokenizer for text processing
//!
//! This module implements a simple character-level tokenizer that maps each unique
//! character to a token ID. It also supports a special BOS (Beginning of Sequence) token.

use std::collections::HashMap;

/// A simple character-level tokenizer
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Map from character to token ID
    char_to_id: HashMap<char, usize>,
    /// Map from token ID to character
    id_to_char: Vec<char>,
    /// Token ID for the Beginning of Sequence (BOS) token
    pub bos_token: usize,
    /// Total vocabulary size (including BOS)
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer from a collection of documents
    ///
    /// # Arguments
    /// * `docs` - Iterator of string slices representing documents
    ///
    /// # Returns
    /// A new `Tokenizer` instance with a vocabulary built from unique characters
    pub fn from_docs<'a, I>(docs: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        // Collect all unique characters and sort them
        let mut chars: Vec<char> = docs
            .into_iter()
            .flat_map(|doc| doc.chars())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort_unstable();

        // Build character to ID mapping
        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        let id_to_char = chars;
        let bos_token = id_to_char.len();
        let vocab_size = id_to_char.len() + 1; // +1 for BOS token

        Self {
            char_to_id,
            id_to_char,
            bos_token,
            vocab_size,
        }
    }

    /// Encode a string into a sequence of token IDs
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// A vector of token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }

    /// Encode a string with BOS tokens at the start and end
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// A vector of token IDs with BOS tokens surrounding the encoded text
    pub fn encode_with_bos(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![self.bos_token];
        tokens.extend(self.encode(text));
        tokens.push(self.bos_token);
        tokens
    }

    /// Decode a sequence of token IDs back into a string
    ///
    /// # Arguments
    /// * `tokens` - The token IDs to decode
    ///
    /// # Returns
    /// A `String` containing the decoded text (BOS tokens are skipped)
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&id| {
                if id == self.bos_token {
                    None
                } else {
                    self.id_to_char.get(id).copied()
                }
            })
            .collect()
    }

    /// Get the character for a given token ID
    ///
    /// # Arguments
    /// * `id` - The token ID
    ///
    /// # Returns
    /// `Some(char)` if the ID is valid, `None` if it's the BOS token or invalid
    pub fn id_to_char(&self, id: usize) -> Option<char> {
        if id == self.bos_token {
            None
        } else {
            self.id_to_char.get(id).copied()
        }
    }

    /// Get the token ID for a given character
    ///
    /// # Arguments
    /// * `c` - The character to look up
    ///
    /// # Returns
    /// `Some(usize)` if the character is in the vocabulary, `None` otherwise
    pub fn char_to_id(&self, c: char) -> Option<usize> {
        self.char_to_id.get(&c).copied()
    }

    /// Get the vocabulary size (including BOS token)
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the BOS token ID
    pub fn bos_token(&self) -> usize {
        self.bos_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let docs = vec!["hello", "world"];
        let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| *s));

        // Check vocabulary size
        // Unique chars: d, e, h, l, o, r, w (7 chars) + BOS = 8
        assert_eq!(tokenizer.vocab_size(), 8);
        assert_eq!(tokenizer.bos_token(), 7);
    }

    #[test]
    fn test_encode_decode() {
        let docs = vec!["abc", "def"];
        let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| *s));

        let text = "abc";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(text, decoded);
    }

    #[test]
    fn test_encode_with_bos() {
        let docs = vec!["abc"];
        let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| *s));

        let text = "ab";
        let tokens = tokenizer.encode_with_bos(text);

        // Should be [BOS, a_id, b_id, BOS]
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], tokenizer.bos_token());
        assert_eq!(tokens[3], tokenizer.bos_token());
    }

    #[test]
    fn test_decode_skips_bos() {
        let docs = vec!["abc"];
        let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| *s));

        let tokens = vec![
            tokenizer.bos_token(),
            tokenizer.char_to_id('a').unwrap(),
            tokenizer.char_to_id('b').unwrap(),
            tokenizer.bos_token(),
        ];

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn test_char_ordering() {
        let docs = vec!["cba"];
        let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| *s));

        // Characters should be sorted
        let a_id = tokenizer.char_to_id('a').unwrap();
        let b_id = tokenizer.char_to_id('b').unwrap();
        let c_id = tokenizer.char_to_id('c').unwrap();

        assert!(a_id < b_id);
        assert!(b_id < c_id);
    }
}
