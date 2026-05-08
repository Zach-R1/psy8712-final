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
library(caret)
library(tictoc)

# library() look at ollama stuff I think it was httr
# Lib

n_cores <- detectCores() - 1

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv")

cleaned_data <- import_data %>%
  filter(!is.na(overall_rating)) %>%
  sample_n(50000) %>%  # Chose in the end to do a sample of the data as the processing time was just too long. Checked rating distribution of full set and the sample and the distrobution is basically the same. Given our sample size this is not surprising.
  mutate(
    doc_id = row_number()
    ) %>%
  unite(review, headline, pros, cons, sep = " ", na.rm = TRUE) %>% # Filter to ensure zero missing data in outcome
  select(overall_rating, review, doc_id )  # In an actual project I'd keep the other data, but since it isnt needed here I only selected necisary variables to reduce processing time.


# This was a hard decision. Originally I planned to make three seperate corpora for headline, pros, and cons. Since the goal of the assignment is the best model in terms of prediction I chose not to. It'd help with interpretability, but couldn't find any info suggesting it would help prediction. So the additional effort and processing time didnt seem worth it This also deals with the issue later in the process where the transformation process would leave blanks in headline



# Note: All of the below code from here to the topic_model was made originally to be able to take the full dataset, but timewise it wound up being the unfeasable to continue that way. I maintained the code as is becuase it was more efficient.

# Maybe change sections later for the below
# Dataset contains a decent enough emoji use that I didnt think it'd hurt to build a seperte transformation function. Also I wanted to see if I could get it to work.

emoji_transformer <- function(x) {
  x <- stri_replace_all_regex(x, "[\U0001F3FB-\U0001F3FF]", "") # Remove variants of emojis to only leave base version (thumbs up skin tone tags would lead them to be read as different emojis otherwise). Ignore low base rate emojis as sparsity removal should deal with them
  
  patterns <- c( # Set the emojis I wanted to replace via their unicode representation
    "\U0001F44D", "\U0001F44E", "\U0001F44C", "\U0001F525",
    "\U0001F60A", "\U0001F60C", "\U0001F601", "\U0001F642",
    "\U0001F61E", "\U0001F615", "\U0001F641"
  )
  
  replacements <- c( # Set the names to replace the emojis with additional space to avoid concatenation issues
    "thumbs up ", "thumbs down ", "okay ", "fire ",
    "smile ", "smile ", "smile ", "smile ",
    "frown ", "frown ", "frown "
  )
  
  x <- stri_replace_all_fixed(x, patterns, replacements, vectorize_all = FALSE) # PUT THE FULL SEQUENCE HERE !!!!!!!!!!!!!!!!!!!!!!!!!
  
  return(x) # Return the new emojiless string
}

# Test <- cleaned_data %>%
#   mutate(headline = emoji_transformer(cleaned_data$review)) # Tested functionality


sw_pattern <- paste0("\\b(", paste(c(stopwords("en"), "none", "con", "pro", ""), collapse = "|"), ")\\b") # EXPLANATION OF EACH PART!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Added none, pro, and con as many people put them into their review and it was pretty uniformative. Thought about removing great and good, but when we look at the bigrams they are actually really helpful



cl <- makeCluster(n_cores) # Created cluster with max - 1 cores for parrallelization
clusterEvalQ(cl, { # 
  library(stringr)
  library(qdap)
  library(stringi)
  library(textstem)
})
clusterExport(cl, c("emoji_transformer", "sw_pattern")) # Exported built functions so they could be used in the built cluster

chunks <- split(cleaned_data$review, cut(seq_along(cleaned_data$review), n_cores)) # chunk the dataset sequentially in equal parts to give to each seperate core

results <- parLapply(cl, chunks, function(x) { # Parrallely apply transformations to each chunk
  x %>% # for each string
    emoji_transformer() %>% # Remove emojis
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
  mutate(review_processed = unlist(results)) # WHAT ID DID and did not -select orignial reviews coloumn as i wanted easy access to compare the text pre and post processing

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
review_slim_dtm <- removeSparseTerms(review_dtm, .9999) # Removed sparce terms with a cutoff that gave me a ratio between 3:1 and 2:1 based on the below commented out code 

# REDO THE ABOVE COMMENTS@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

dtm_stm <- readCorpus( # Convert to a format that can be used with stm
  review_slim_dtm, type="slam")

kresult <- searchK( # fits models for specified values of k to provided diagnostics that help choose the appropriate number of topics
  dtm_stm$documents,
  dtm_stm$vocab,
  K = seq(2, 20, by = 2)) # It took 1 1/2 hours to get two models to converge. So instead of 10 models i switched to 5. Wanted to strike a balance between actual answer and saving processing time.


stopCluster(lc2) # Stop cluster to free resuources
registerDoSEQ() # End parallel execution in backend

# saveRDS(kresult, "../data/kresult.rds")
plot(kresult)

topic_model <- stm(dtm_stm$documents, # fit the final model
                   dtm_stm$vocab, 
                   10, # The closest thing to an elbow I can find
                   verbose = FALSE)


####START HERE!!!!!!!!!!!!!!!!
# saveRDS(topic_model, "../data/topic_model.rds")


labelTopics(topic_model, n = 6) # Looking at this if I could do this project fully with best practices I'd add more stop words like I'd need to think about whether to remove good and great. They don't see very informative on, but appear useful in bigrams.

docs_kept <- as.integer(names(dtm_stm$documents))

aligned_tbl <- processed_data[docs_kept, ] 

findThoughts(topic_model, text = aligned_tbl$review_processed, n = 3)

plot(topic_model, type = "summary", n = 8) # Plots topic models
topicCorr(topic_model) # Provides intercorrelation of topic models
plot(topicCorr(topic_model)) # graphs relationships between modles

theta <- topic_model$theta # extracts probability 


# Still need to do the rest of this code but for now embeddings over night.


topic_labels <- c("Organizational resources & processes", "Day-to-day work experience", "Toxic management", "Leadership quality and org dysfunction", "Employee perks", "Positive atmosphere", "Learning opportunities & travel demands", "Work-life balance", "Generic positive comments", "Salary progression & promotion") # Topic names...No need to redo stop words

token_count <- slam::row_sums(review_slim_dtm[docs_kept, ]) # extracted token count

top_words <- review_slim_dtm %>% # I'd take all tokens for an actual project, but for here I'll be using the top 100 most frequent again for easy processing of machine learning models since embeddings will add a lot on their own
  slam::col_sums() %>% # col_sums was chosen over colSums as the base R version can't handle the dtm used to get frequency of each token
  sort(decreasing = TRUE) %>% # sort tokens so they are in order of most to least frequent
  head(100) %>% # Take the top 100
  names() #!!!!!!!!!!!!!!!!!!!!!!!!

top_words_tbl <- review_slim_dtm[as.character(docs_kept), top_words] %>%
  as.matrix() %>%
  as.tibble() %>%
  mutate(doc_id = docs_kept, .before = 1)
  


topics_tbl <- tibble(
  doc_id = docs_kept,  # 
  original = aligned_tbl$review_processed, # original post titles, already aligned to surviving docs # T
  topic = apply(theta, 1, which.max), # for each doc (row), pick the topic with highest probability
  probability = apply(theta, 1, max),  # grab that highest probability value
  overall_rating = aligned_tbl$overall_rating #ratings aligned to other factors
) %>%
  mutate(topic_label = topic_labels[topic],
         token_count = token_count)  # Use topic number as index to assign proper topic label to specified topic number

topics_tokens_tbl <- topics_tbl %>%
  left_join(top_words_tbl, by = "doc_id")






# embeddings DONT FORGET TO OPEN OLLAMA in cmd

clean_data_filtered <- cleaned_data %>% filter(doc_id %in% docs_kept)

embed_cols <- str_c("emb_", 1:768)



embedding_df <- imap(cleaned_data$review, function(text, i) {
  if (i %% 100 == 0) message(sprintf("Processing %d of %d", i, nrow(cleaned_data)))
  resp <- httr::POST(
    "http://localhost:11434/api/embed",
    body = jsonlite::toJSON(list(model = "nomic-embed-text", input = text), auto_unbox = TRUE),
    httr::content_type_json()
  )
  as.numeric(httr::content(resp)$embeddings[[1]])
}) %>%
  unname() %>%
  do.call(cbind, .) %>%
  t() %>%
  as_tibble(.name_repair = ~ embed_cols) %>%
  mutate(doc_id = cleaned_data$doc_id) %>%
  select(doc_id, everything())

embedding_df <- embedding_df %>% filter(doc_id %in% docs_kept) # To fix the above!!!!!!!!!!!!!!!!!!!!!! 50000 to 49998 as with others




full_data_tbl <- topics_tokens_tbl %>% 
  left_join(embedding_df, by = "doc_id") %>%
  select(-original, -doc_id, -probability, -topic_label) # Remove all variables not used in the prediction. I left token count in as a controll for the token models



# Analysis

model_test_data <- full_data_tbl %>% head(100)


train_index  <- createDataPartition(model_test_data$ , p = 0.75, list = FALSE) # Partition data!!!!!!!!!!!!!! CHANGE THE MODEL AFTER TESTING
training_data <- model_tbl[train_index, ] # Create training dataset
test_data  <- model_tbl[-train_index, ] # Create holdout data


cross_val_control <- trainControl( # Create standard CV
  method = "cv",
  number = 10,
  search = "random",
  verboseIter = TRUE
)

# 1 LM token
run_lm <- function(formula) {
  train(
    formula, 
    training_data, 
    na.action = na.pass, 
    method = "lm", 
    preProcess = c("zv", "center", "scale", "medianImpute"), 
    trControl = cross_val_control)}

run_elastic <- run_model2 <- function(formula) { 
  train(
    formula, 
    training_data, 
    na.action = na.pass, 
    method = "glmnet", 
    preProcess = c("zv", "center", "scale", "medianImpute"), 
    tuneGrid = expand.grid( 
      alpha = c(0,1), 
      lambda = seq(0.001, 0.1, length = 10) 
  ),
  trControl = cross_val_control
)}


run_rf <- function(formula, mtry_vals) {
  train(
    formula, 
    training_data, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("zv", "center", "scale", "medianImpute"), 
    tuneGrid = expand.grid( 
      mtry = mtry_vals, 
      splitrule = c("variance", "extratrees"), 
      min.node.size = 5 
    ),
    trControl = cross_val_control 
  )}

# 2 lm topic

# embeddings vs tokens
# topics vs tokens
# 




# LM, ELastice, RF

# Was between Lm and XGboost