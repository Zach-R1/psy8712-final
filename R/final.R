# Script Settings and Resources
set.seed(777)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tm)
library(qdap)
library(textstem)
library(stringi)
library(RWeka)
library(parallel)
library(doParallel)
#library()
# library() look at ollama stuff
# Lib

n_cores <- detectCores() - 1

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv")

cleaned_data <- import_data %>%
  filter(!is.na(overall_rating)) %>% 
  mutate(
    doc_id = row_number()
    ) %>%
  unite(review, headline, pros, cons, sep = " ", na.rm = TRUE) %>% # Filter to ensure zero missing data in outcome
  select(overall_rating, review, doc_id) # In an actual project I'd keep the other data, but since it isnt needed here I only selected necisary variables to reduce processing time.

# This was a hard decision. Originally I planned to make three seperate corpora for headline, pros, and cons. Since the goal of the assignment is the best model in terms of prediction I chose not to. It'd help with interpretability, but couldn't find any info suggesting it would help prediction. So the additional effort and processing time didnt seem worth it This also deals with the issue later in the process where the transformation process would leave blanks in headline



# Maybe change sections later for the below
# Dataset contains a decent enough emoji use that I didnt think it'd hurt to build a seperte transformation function. Also I wanted to see if I could get it to work.

emoji_transformer <- function(x) {
  x <- stri_replace_all_regex(x, "[\U0001F3FB-\U0001F3FF]", "")
  
  patterns <- c(
    "\U0001F44D", "\U0001F44E", "\U0001F44C", "\U0001F525",
    "\U0001F60A", "\U0001F60C", "\U0001F601", "\U0001F642",
    "\U0001F61E", "\U0001F615", "\U0001F641"
  )
  
  replacements <- c(
    "thumbs up ", "thumbs down ", "okay ", "fire ",
    "smile ", "smile ", "smile ", "smile ",
    "frown ", "frown ", "frown "
  )
  
  x <- stri_replace_all_fixed(x, patterns, replacements, vectorize_all = FALSE)
  
  return(x)
}

sw_pattern <- paste0("\\b(", paste(stopwords("en"), collapse = "|"), ")\\b")
# Test <- cleaned_data %>%
#   mutate(headline = emoji_transformer(cleaned_data$review)) # Tested functionality


cl <- makeCluster(n_cores)
clusterEvalQ(cl, {
  library(stringr)
  library(qdap)
  library(stringi)
  library(textstem)
})
clusterExport(cl, c("emoji_transformer", "sw_pattern"))

chunks <- split(cleaned_data$review, cut(seq_along(cleaned_data$review), n_cores))

results <- parLapply(cl, chunks, function(x) {
  x %>%
    emoji_transformer() %>%
    replace_contraction() %>%
    replace_abbreviation() %>%
    str_to_lower() %>%
    str_remove_all("[[:punct:]]") %>%
    str_remove_all("[[:digit:]]") %>%
    lemmatize_strings() %>%
    str_remove_all(regex(sw_pattern, ignore_case = TRUE)) %>%
    str_squish()
})

stopCluster(cl)

processed_data <- cleaned_data %>%
  mutate(review_processed = unlist(results))



# Containerize
review_corpus_original <- VCorpus(VectorSource(processed_data$review_processed))




# Plan: dtm, dtmslim, see if you should add additional stop words, model with tokens, and topic, then do embeddings POST, do the rest of the modeling, answer questions.



















# use comparison code maybe delete maybe comment out afterwards

# compare_them <- function(x, y) { # make comarison function
#   i <- sample(seq_along(x), 1)
#   
#   cat("Corpus 1", " Row ", i, ": ", x[[i]], "\n\n") # Used cat to make visually neat output for easy comparison
#   cat("Corpus 2", " Row ", i, ": ", y[[i]]) # Used cat to make visually neat output for easy comparison
# } Seems to have done a decent job
# 
# 
# compare_them(processed_data$review, processed_data$review_processed) # apply function



# Analysis