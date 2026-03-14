//! Chat formatting and tokenization — convert chat messages to token sequences.

use std::collections::HashMap;
use kerr_engine::data::{Dataset, TokenMode, tokenize_words};

use crate::api_types::ChatMessage;

/// Vocabulary extracted from a Dataset for serving-time encode/decode.
pub struct Vocab {
    pub token_to_idx: HashMap<String, usize>,
    pub idx_to_token: Vec<String>,
    pub mode: TokenMode,
    pub vocab_size: usize,
}

impl Vocab {
    /// Extract vocabulary from a loaded Dataset, discarding training data.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        Self {
            token_to_idx: dataset.token_to_idx.clone(),
            idx_to_token: dataset.idx_to_token.clone(),
            mode: dataset.mode,
            vocab_size: dataset.vocab_size,
        }
    }

    /// Encode text into token indices.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.mode {
            TokenMode::Char => {
                text.chars()
                    .map(|c| *self.token_to_idx.get(&c.to_string()).unwrap_or(&0))
                    .collect()
            }
            TokenMode::Word => {
                let tokens = tokenize_words(text);
                tokens.iter()
                    .map(|t| *self.token_to_idx.get(t).unwrap_or(&0))
                    .collect()
            }
        }
    }

    /// Decode token indices back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let sep = match self.mode {
            TokenMode::Char => "",
            TokenMode::Word => " ",
        };
        tokens.iter()
            .map(|&t| {
                if t < self.idx_to_token.len() {
                    self.idx_to_token[t].as_str()
                } else {
                    "?"
                }
            })
            .collect::<Vec<_>>()
            .join(sep)
    }
}

/// Format chat messages into a token sequence for the model.
///
/// Template (Shakespeare-trained, not instruction-tuned):
///   {system_message}\n\n{user_message}\n\n
pub fn format_chat(messages: &[ChatMessage], vocab: &Vocab) -> Vec<usize> {
    let mut text = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" => {
                text.push_str(&msg.content);
                text.push_str("\n\n");
            }
            _ => {} // ignore unknown roles
        }
    }
    vocab.encode(&text)
}
