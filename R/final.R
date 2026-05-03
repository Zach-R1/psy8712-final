# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tm)
library(qdap)
library(textstem)
library(RWeka)
# library() look at ollama stuff
# Lib

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv")

cleaned_data <- import_data %>%
  filter(!is.na(overall_rating)) %>% # Filter to ensure zero missing data in outcome
  select(overall_rating, headline, pros, cons) # In an actual project I'd keep the other data, but sisnce it isnt needed here I only selected necisary variables to reduce processing time.

# Maybe change sections later for the below
emoji_transformer <- function(x) { # Dataset contains significant enough emjoi use to want transformation of at least the more common or more meaningfully used emojis. Built function to replace emojis unicode with the emoji names
  patterns <- c(
    "\\x{1F44D}" = "thumbs up",
    "\\x{ }" = "thumbs down",
    "\\x{ }" = "okay",
    "\\x{1F525}" = "fire",
    "\\x{ }" = "smile",
    "\\x{ }" = "sad",
    "\\x{ }" = "scared",
    "\\x{ }" = " ",
  )
  
  sapply(x, emoji_transformer) # determin if this should actually be a for loop or one of the apply family
}

headline_corpus_original <- VCorpus(VectorSource(cleaned_data$headline))

pros_corpus_original <- VCorpus(VectorSource(cleaned_data$pros))

cons_corpus_original <- VCorpus(VectorSource(cleaned_data$cons))

# maybe do function to abstract out and use on all three?

corpus_transformer <- function(x) { # Making a function instead of writing this three times
  tm_map(content_transformer(emoji_transformer))
  tm_map(content_transformer(str_to_lower)) %>% # shift casing to lower
    tm_map(removePunctuation) %>% # remove punctuation .....
    tm_map(content_transformer(replace_contraction)) %>% # replace contractions with both words
    tm_map(content_transformer(replace_abbreviation)) %>% # replace abbreviations with full words
    tm_map(removeNumbers) %>% # remove numbers as they are irrelevant here
    tm_map(content_transformer(lemmatize_words)) %>% # added content transformer to lemmatize becuase.......
    tm_map(removeWords, stopwords("en")) %>% # Add more stopwords in c() as you look at the comparisons
    tm_map(stripWhitespace)
  
  
} # DO SOMETHING ABOUT BLANK CELLS BECUASE YOU WILL HAVE THEM

headline_corpus <- corpus_transformer(headline_corpus_original)

pros_corpus <- corpus_transformer(pros_corpus_original)

cons_corpus <- corpus_transformer(cons_corpus_original)

# use comparison code maybe delete maybe comment out afterwards

compare_them <- function(x, y) { # make comarison function
  i <- sample(seq_along(x), 1)
  
  cat("Corpus 1", " Row ", i, ": ", content(x[[i]]), "\n\n") # Used cat to make visually neat output for easy comparison
  cat("Corpus 2", " Row ", i, ": ", content(y[[i]])) # Used cat to make visually neat output for easy comparison
}


compare_them(headline_corpus_original, headline_corpus) # apply function



# Analysis