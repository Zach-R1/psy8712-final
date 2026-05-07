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
library(stm)
library(tictoc)

# library() look at ollama stuff I think it was httr
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
  select(overall_rating, review, doc_id )%>%  # In an actual project I'd keep the other data, but since it isnt needed here I only selected necisary variables to reduce processing time.
  sample_n(50000) # Chose in the end to do a sample of the data as the processing time was just too long. Checked rating distribution of full set and the sample and the distrobution is basically the same. 

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

# write_csv(processed_data, "../data/processed_data.csv") # Save processing time in case of R restart during development errors crashed some of my previous runs

# processed_data <- read_csv("../data/processed_data.csv")

# Containerize
review_corpus <- VCorpus(VectorSource(processed_data$review_processed))




# Plan: dtm, dtmslim, see if you should add additional stop words, model with tokens, and topic, then do embeddings POST, do the rest of the modeling, answer questions.


# dtm
my_tokenizer <- function(x) { NGramTokenizer(x, Weka_control(min=1, max=2)) } # make tokenizer That allows for unigram and bigrams

review_dtm <- DocumentTermMatrix( # make dtm
  review_corpus, 
  control = list(tokenize = my_tokenizer))

# saveRDS(review_dtm, "../data/review_dtm.rds") # Save in case of R shut down
#review_dtm <- review_dtm <- readRDS("../data/review_dtm.rds")

# # doc_ids <- sapply(review_corpus, meta, "id") may not need anymore
# tic()
# names(review_corpus) <- processed_data$doc_id
# toc()
#slim_dtm
review_slim_dtm <- removeSparseTerms(review_dtm, .9999) # Used this cutoff so I could actually get the project done, a 3:1 ratio would be around 200k+ but I'd probably do around 30k+ if I were actually doing the project


# FIND A NEW SPARCITY RATIO!!!

N <- nrow(review_dtm)
N

lower_k <- N / 3
upper_k <- N / 2
lower_k
upper_k

k <- ncol(review_slim_dtm)
k
N / k













# Saved not becuase this step 


# OH OH FOR EMBEDDING CALL MAKE A SEPREATE DATA TABLE WITH ONLY A COUPLE REVIEWS TO TEST AND THEN DO IT WITH THE FULL DATA BOOM!!!!



# Analysis

lc2 <- makeCluster(n_cores) # Made cluster using 7/8 processors as I want to do other work while the code runs

registerDoParallel(lc2) # registered parellel backend to alow  loops to execute across the 7 cores of the cluster
tic()
dtm_stm <- readCorpus( # Convert to a format that can be used with stm
  review_slim_dtm, type="slam")
toc()

tic()
kresult <- searchK( # fits models for specified values of k to provided diagnostics that help choose the appropriate number of topics
  dtm_stm$documents,
  dtm_stm$vocab,
  K = seq(2, 20, by = 2)) # It took 1 1/2 hours to get two models to converge. So instead of 10 models i switched to 5. Wanted to strike a balance between actual answer and saving processing time.
toc() # took 11014 seconds

stopCluster(lc2) # Stop cluster to free resuources
registerDoSEQ() # End parallel execution in backend

# saveRDS(kresult, "../data/kresult.rds")
plot(kresult)

topic_model <- stm(dtm_stm$documents, # fit the final model
                   dtm_stm$vocab, 
                   8, # The closest thing to an elbow I can find
                   verbose = FALSE)


####START HERE!!!!!!!!!!!!!!!!
# saveRDS(topic_model, "../data/topic_model.rds")


labelTopics(topic_model, n = 10) # Looking at this if I could do this project fully with best practices I'd add more stop words like I'd need to think about whether to remove good and great. They don't see very informative, but appear useful in bigrams.

docs_kept <- as.integer(names(dtm_stm$documents))

aligned_tbl <- processed_data[docs_kept, ] 

findThoughts(topic_model, text = aligned_tbl$review_processed, n = 3)

plot(topic_model, type = "summary", n = 8) # Plots topic models
topicCorr(topic_model) # Provides intercorrelation of topic models
plot(topicCorr(topic_model)) # graphs relationships between modles

theta <- topic_model$theta # extracts probability 


# Still need to do the rest of this code but for now embeddings over night.


topic_labels <- c("Toxic Management", "") # Topic names...No need to redo stop words


topics_tbl <- tibble(
  doc_id      = docs_kept,  # 
  original    = aligned_tbl$review_processed, # original post titles, already aligned to surviving docs
  topic       = apply(theta, 1, which.max), # for each doc (row), pick the topic with highest probability
  probability = apply(theta, 1, max),  # grab that highest probability value
  upvotes     = aligned_tbl$overall_rating #ratings aligned to other factors
) %>%
  mutate(topic_label = topic_labels[topic])  # Use topic number as index to assign proper topic label to specified topic number








# embeddings DONT FORGET TO OPEN OLLAMA in cmd

embeddings <- lapply(processed_data$review_processed, function(text) {
  response <- httr::POST(
    "http://localhost:11434/api/embeddings",
    body = jsonlite::toJSON(list(
      model = "nomic-embed-text",
      prompt = text
    ), auto_unbox = TRUE),
    httr::content_type_json()
  )
  httr::content(response)$embedding
})
















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