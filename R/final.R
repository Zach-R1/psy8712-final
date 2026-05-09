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
#library(tictoc)


n_cores <- detectCores() - 1

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv")

cleaned_data <- import_data %>%
  filter(!is.na(overall_rating)) %>%
  slice_sample(n = 10000) %>%  # Sampled data becuase 800k+ was too much for the laptop. Checked that proportions were similar to full data using prop.table(table(import_data$overall_rating)) and prop.table(table(cleaned$overall_rating)) Also sample_n was more efficient than sample, but slice_sample superceded sample_n
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


# Containerize
review_corpus <- VCorpus(VectorSource(processed_data$review_processed))



# dtm
my_tokenizer <- function(x) { NGramTokenizer(x, Weka_control(min=1, max=2)) } # make tokenizer That allows for unigram and bigrams

review_dtm <- DocumentTermMatrix( # make dtm
  review_corpus, 
  control = list(tokenize = my_tokenizer))


#slim_dtm
review_slim_dtm <- removeSparseTerms(review_dtm, .9991) # Removed sparce terms with a cutoff that gave me a ratio between 3:1 and 2:1 based on the below commented out code 

# REDO THE ABOVE COMMENTS@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# N <- nrow(review_dtm)
# N
# 
# lower_k <- N / 3  # maybe delete and just comment!!!!!!!!!!!!!!!!!!!!!!
# upper_k <- N / 2
# lower_k
# upper_k
# 
# k <- ncol(review_slim_dtm)
# k
# N / k


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
                   6, # The closest thing to an elbow I can find
                   verbose = FALSE)




labelTopics(topic_model, n = 6) # Looking at this if I could do this project fully with best practices I'd add more stop words like I'd need to think about whether to remove good and great. They don't see very informative on, but appear useful in bigrams.

docs_kept <- as.integer(names(dtm_stm$documents))

aligned_tbl <- processed_data[docs_kept, ]



findThoughts(topic_model, text = aligned_tbl$review_processed, n = 10)

plot(topic_model, type = "summary", n = 8) # Plots topic models
topicCorr(topic_model) # Provides intercorrelation of topic models
plot(topicCorr(topic_model)) # graphs relationships between modles

theta <- topic_model$theta # extracts probability 


# Still need to do the rest of this code but for now embeddings over night.


topic_labels <- c("Corporate Culture & Bureaucracy", "Generic positive comments", "Benefits & advancement oppertunities", "Poor management & stressful conditions", "Work-life balance", "Learning opportunities") # Topic names...No need to redo stop words

token_count <- slam::row_sums(review_slim_dtm[docs_kept, ]) # extracted token count

top_words <- review_slim_dtm %>% # I'd take all tokens for an actual project, but for here I'll be using the top 100 most frequent again for easy processing of machine learning models since embeddings will add a lot on their own
  slam::col_sums() %>% # col_sums was chosen over colSums as the base R version can't handle the dtm used to get frequency of each token
  sort(decreasing = TRUE) %>% # sort tokens so they are in order of most to least frequent
  head(200) # Take the top 200 tokens, remove nzv is getting rid of most of these anyways so if i do this now it saves me a lot of processing overhead in every model fitting that includes tokens

top_words_tbl <- review_slim_dtm[as.character(docs_kept), names(top_words)] %>%
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






# embeddings 

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

embedding_df_filtered <- embedding_df %>% filter(doc_id %in% docs_kept) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




full_data_tbl <- topics_tokens_tbl %>% 
  left_join(embedding_df_filtered, by = "doc_id") %>%
  select(-original, -doc_id, -probability, -topic_label) # Remove all variables not used in the prediction. I left token count in as a controll for the token models


# saveRDS(full_data_tbl, "../out/data.RDS")
# full_data_tbl <- readRDS("../out/data.RDS")


# Analysis

#model_test_data <- full_data_tbl %>% head(200)


train_index  <- createDataPartition(full_data_tbl$overall_rating, p = 0.75, list = FALSE) # Partition data!!!!!!!!!!!!!! CHANGE THE MODEL AFTER TESTING
training_data <- full_data_tbl[train_index, ] %>%
  mutate(dummy = 0)# Create training dataset
test_data <- full_data_tbl[-train_index, ] %>%
  mutate(dummy = 0) # Create holdout data


cross_val_control <- trainControl( # Create standard CV
  method = "cv",
  number = 5, # Reduced from 10 to reduce processing time, but would normally do 10 as the standard
  search = "random",
  verboseIter = TRUE
)


# Define predictor sets

token_cols <- c("token_count", names(top_words))
embed_vars <- paste0("emb_", 1:768)
token_data <- training_data %>% select(overall_rating, all_of(token_cols))
# Subsets
token_data <- training_data[, c("overall_rating", token_cols)]
topic_data <- training_data[, c("overall_rating", "topic", "dummy")]
embed_data <- training_data[, c("overall_rating", embed_vars)]
token_topic_data <- training_data[, c("overall_rating", "topic", token_cols)]
token_embed_data <- training_data[, c("overall_rating", token_cols, embed_vars)]
topic_embed_data <- training_data[, c("overall_rating", "topic", embed_vars)]
token_topic_embed_data <- training_data[, c("overall_rating", "topic", token_cols, embed_cols)]


# Define MTRY Values for each model
token_mtry <- c(33, 67, 100) # 
topic_mtry <- 1
embed_mtry <- c(256, 384, 512)
token_topic_mtry <- c(33, 67, 101)
token_embed_mtry <- c(256, 384, 584)
topic_embed_mtry <- c(256, 384, 512)
token_topic_embed_mtry <- c(256, 384, 585)



# Models
run_lm <- function(data) {
  train(
    overall_rating ~ .,
    data = as.data.frame(data),
    na.action = na.pass,
    method = "lm",
    preProcess = c("zv", "center", "scale", "medianImpute"),
    trControl = cross_val_control
  )
}


run_elastic <- function(data) {
  train(
    overall_rating ~ .,
    data = as.data.frame(data),
    na.action = na.pass,
    method = "glmnet",
    preProcess = c("zv", "center", "scale", "medianImpute"),
    tuneGrid = expand.grid(
      alpha = c(0, 1),
      lambda = seq(0.001, 0.1, length = 10)
    ),
    trControl = cross_val_control
  )
}


run_elastic_simple <- function(data) {
  data <- as.data.frame(data)

  # count predictors (exclude outcome)
  predictor_count <- ncol(data) - 1

  # glmnet via caret requires >= 2 predictors so I needed another function to provide a dummy variable
  if (predictor_count == 1) {
    data$.dummy <- 0
  }

  train(
    overall_rating ~ .,
    data = data,
    na.action = na.pass,
    method = "glmnet",
    preProcess = c("center", "scale", "medianImpute"),
    tuneGrid = expand.grid(
      alpha = c(0, 1),
      lambda = seq(0.001, 0.1, length = 3)
    ),
    trControl = cross_val_control
  )
}


run_rf <- function(data, mtry_vals) {
  train(
    overall_rating ~ .,
    data = as.data.frame(data),
    na.action = na.pass,
    method = "ranger",
    preProcess = c("zv", "center", "scale", "medianImpute"),
    tuneGrid = expand.grid(
      mtry = mtry_vals,
      splitrule = c("variance", "extratrees"),
      min.node.size = 20 # i'd normally do 5, but my laptop was hot enough to be concerning so I reduced it. I realize that I am sacrificing
    ),
    trControl = cross_val_control,
    num.trees = 200
  )
}

# 2 lm topic

# embeddings vs tokens
# topics vs tokens

# Linear models
model1 <- run_lm(token_data) # You'll notice these are all their own line of code rather than a loop or apply. just a preference thing. I didnt want to add steps to what I just got working here at great effort.
model2 <- run_lm(topic_data)
model3 <- run_lm(embed_data)
model4 <- run_lm(token_topic_data)
model5 <- run_lm(token_embed_data)
model6 <- run_lm(topic_embed_data)
model7 <- run_lm(token_topic_embed_data)

lc3 <- makeCluster(n_cores)
registerDoParallel(lc3)

# Elastic net models
model8  <- run_elastic(token_data)
model9  <- run_elastic_simple(topic_data) # Run elastic with dummy code so that glmnet allows the use of topic alone
model10 <- run_elastic(embed_data)
model11 <- run_elastic(token_topic_data)
model12 <- run_elastic(token_embed_data)
model13 <- run_elastic(topic_embed_data)
model14 <- run_elastic(token_topic_embed_data)

# Random forest models
model15 <- run_rf(token_data, token_mtry) # redo
model16 <- run_rf(topic_data, topic_mtry) # redo
model17 <- run_rf(embed_data, embed_mtry) # redo
model18 <- run_rf(token_topic_data, token_topic_mtry)
model19 <- run_rf(token_embed_data, token_embed_mtry)
model20 <- run_rf(topic_embed_data, topic_embed_mtry)
model21 <- run_rf(token_topic_embed_data, token_topic_embed_mtry)

stopCluster(lc3)

registerDoSEQ() 



# Check later
models <- list(model1, model2, model3, model4, model5, model6, model7,
               model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21)


cv_results_tbl <- tibble(
  model = c(rep("lm", 7), rep("elastic", 7), rep("rf", 7)),
  predictors = rep(c("token", "topic", "embeddings", "token_topic", "token_embed", "topic_embed", "token_topic_embed"), 3),
  cv_Rsquared = map_dbl(models, ~ max(.x$results$Rsquared, na.rm = TRUE)),
  cv_RMSE = map_dbl(models, ~ min(.x$results$RMSE, na.rm = TRUE))
)


ho_results_tbl <- tibble(
  model = c(rep("lm", 7), rep("elastic", 7), rep("rf", 7)),
  predictors = rep(c("token", "topic", "embeddings", "token_topic", "token_embed", "topic_embed", "token_topic_embed"), 3),
  ho_Rsquared = map_dbl(models, ~ cor(predict(.x, newdata = test_data, na.action = na.pass), test_data$overall_rating)^2),
  ho_RMSE = map_dbl(models, ~ {
    preds <- predict(.x, newdata = test_data, na.action = na.pass)
    sqrt(mean((preds - test_data$overall_rating)^2))
  })
)

final_results_tbl <- cv_results_tbl %>% left_join(ho_results_tbl, by = c("model", "predictors"))

# write.csv(final_results_tbl, "../out/final_results.csv") Saved for my benefit might switch to RDS later since it wont be in the workpace image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?


#   RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?


#   RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?


#   RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?


save.image("../out/workspace.RData")

