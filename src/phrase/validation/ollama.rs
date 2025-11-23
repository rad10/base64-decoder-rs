use std::fmt::Display;

use async_trait::async_trait;
#[cfg(feature = "async")]
use futures::{Stream, StreamExt};
use itertools::Itertools;

#[cfg(feature = "ollama")]
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};
#[cfg(feature = "ollama")]
use url::Url;

use crate::phrase::schema::{Snippet, Variation};

/// Contains all the data necessary for string validation with ollama
pub struct OllamaHandler {
    pub ollama: Ollama,
    pub model: String,
}

const MAX_MESSAGE_SIZE: usize = 5000;
const OLLAMA_PROMPT: &str = "
Your task is to take phrases given to you and return a number between 0 and 1
relating to how confident you are that the given phrase is english text or
powershell code.

Every message will appear in the given form

```
1. `<phrase 1 here>`
2. `<phrase 2 here>`
```

You will need to give a response in the form

```
[
0.123456,
0.893140
]
```
";

#[async_trait]
pub trait AsyncOllama {
    /// Takes a phrase and returns the confidence for each line provided by ollama
    async fn validate_group<T>(
        &self,
        snippet: Snippet<'_, Vec<T>>,
    ) -> impl Stream<Item = (f64, Variation<Vec<T>>)>
    where
        T: Send + Sync,
        Variation<Vec<T>>: Display;

    /// Takes a phrase and returns the confidence for each line provided by ollama
    async fn validate_group_str(
        &self,
        snippet: Snippet<'_, String>,
    ) -> impl Stream<Item = (f64, Variation<String>)>;

    /// Takes a single line and has ollama give a confidence value on the given
    /// line
    async fn validate_line<T>(&self, line: &Variation<T>) -> f64
    where
        T: Send + Sync,
        Variation<T>: Display;
}

impl OllamaHandler {
    pub fn new(address: Url, model: String) -> Self {
        Self {
            ollama: Ollama::from_url(address),
            model,
        }
    }
}

#[async_trait]
impl AsyncOllama for OllamaHandler {
    /// Takes a phrase and returns the confidence for each line provided by ollama
    async fn validate_group<T>(
        &self,
        snippet: Snippet<'_, Vec<T>>,
    ) -> impl Stream<Item = (f64, Variation<Vec<T>>)>
    where
        T: Send + Sync,
        Variation<Vec<T>>: Display,
    {
        let snippet_len = snippet.len();
        futures::stream::iter(snippet.into_iter_var())
            // Separate permutations into groups that will fit into a reasonably sized message
            // max size = x
            // `{y}`\n`{y}`\n... = x
            // z(3 + y) = x
            // z = x/(3 + y)
            .chunks(MAX_MESSAGE_SIZE / (snippet_len + 4) - 3)
            .then(async move |package| {
                let message_data = package
                    .iter()
                    .enumerate()
                    .map(|(num, phrase)| (num + 1, phrase))
                    .map(|(num, phrase)| format!("{num}: `{phrase}`"))
                    .join("\n");

                match self
                    .ollama
                    .generate(
                        GenerationRequest::new(self.model.to_owned(), message_data)
                            .template(OLLAMA_PROMPT),
                    )
                    .await
                {
                    Ok(response) => {
                        // Attempt to collect values from response
                        match serde_json::from_str::<Vec<f64>>(response.response.as_str()) {
                            Ok(values) => {
                                let final_result: Vec<(f64, Variation<Vec<T>>)> =
                                    values.into_iter().zip(package).collect();
                                futures::stream::iter(final_result)
                            }
                            Err(e) => {
                                log::error!(
                                    "Failed to parse response from ollama. Dropping all values"
                                );
                                log::debug!("Received ollama error: {e}");
                                log::debug!(
                                    "Received value from ollama: {}",
                                    response.response.as_str()
                                );
                                futures::stream::iter(Vec::<(f64, Variation<Vec<T>>)>::new())
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to get a response from ollama. Dropping all values");
                        log::debug!("Received ollama error: {e}");
                        futures::stream::iter(Vec::<(f64, Variation<Vec<T>>)>::new())
                    }
                }
            })
            .flatten()
    }

    /// Takes a phrase and returns the confidence for each line provided by ollama
    async fn validate_group_str(
        &self,
        snippet: Snippet<'_, String>,
    ) -> impl Stream<Item = (f64, Variation<String>)> {
        let snippet_len = snippet.len();
        futures::stream::iter(snippet.into_iter_var())
            // Separate permutations into groups that will fit into a reasonably sized message
            // max size = x
            // `{y}`\n`{y}`\n... = x
            // z(3 + y) = x
            // z = x/(3 + y)
            .chunks(MAX_MESSAGE_SIZE / (snippet_len + 4) - 3)
            .then(async move |package| {
                let message_data = package
                    .iter()
                    .enumerate()
                    .map(|(num, phrase)| (num + 1, phrase.to_string().escape_default().to_string()))
                    .map(|(num, phrase)| format!("{num}: `{phrase}`"))
                    .join("\n");

                log::debug!("Sending data to LLM: {message_data}");
                match self
                    .ollama
                    .generate(
                        GenerationRequest::new(self.model.to_owned(), message_data)
                            .template(OLLAMA_PROMPT),
                    )
                    .await
                {
                    Ok(response) => {
                        log::debug!("LLM response: {}", response.response);
                        log::debug!("LLM thinking: {:?}", response.thinking);
                        // Attempt to collect values from response
                        match serde_json::from_str::<Vec<f64>>(&response.response) {
                            Ok(values) => {
                                let final_result: Vec<(f64, Variation<String>)> =
                                    values.into_iter().zip(package).collect();
                                futures::stream::iter(final_result)
                            }
                            Err(e) => {
                                log::error!(
                                    "Failed to parse response from ollama. Dropping all values"
                                );
                                log::debug!("Received ollama error: {e}");
                                log::debug!(
                                    "Received value from ollama: {}",
                                    response.response.as_str()
                                );
                                futures::stream::iter(Vec::<(f64, Variation<String>)>::new())
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to get a response from ollama. Dropping all values");
                        log::debug!("Received ollama error: {e}");
                        futures::stream::iter(Vec::<(f64, Variation<String>)>::new())
                    }
                }
            })
            .flatten()
    }

    /// Takes a single line and has ollama give a confidence value on the given
    /// line
    async fn validate_line<T>(&self, line: &Variation<T>) -> f64
    where
        T: Send + Sync,
        Variation<T>: Display,
    {
        let message_data = format!("1. `{}`", line.to_string().escape_default().to_string());

        match self
            .ollama
            .generate(
                GenerationRequest::new(self.model.to_owned(), message_data).template(OLLAMA_PROMPT),
            )
            .await
        {
            Ok(response) => {
                // Attempt to collect values from response
                match serde_json::from_str::<Vec<f64>>(response.response.as_str()) {
                    Ok(values) => values[0],
                    Err(e) => {
                        log::error!("Failed to parse response from ollama. Dropping all values");
                        log::debug!("Received ollama error: {e}");
                        log::debug!("Received value from ollama: {}", response.response.as_str());
                        0_f64
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to get a response from ollama. Dropping all values");
                log::debug!("Received ollama error: {e}");
                0_f64
            }
        }
    }
    // /// This validator takes the strategy to provide ollama the structural
    // /// skeleton of the phrase and based on the context clues within it, ask
    // /// ollama to determine which variations within each section are correct by
    // /// requiring ollama to apply a confidence on each variation within each
    // /// section. This will then be used to calculate the confidence of each
    // /// permutation
    // pub async fn validate_by_variant_confidences<T>(&self, snippet: &Snippet<'_, Vec<T>>) -> impl Stream<Item = (f64, Variation<Vec<T>>)> {

    // }

    // /// This validation function simply gives ollama the snippets structure
    // /// and asks ollama to create a single variation from this that represents
    // /// ollama's best guess as to what the correct pattern is.
    // pub async fn get_ollama_best_answer<T>(&self, snippet: &Snippet<'_, Vec<T>>) -> impl Stream<Item = (f64, Variation<Vec<T>>);
}
