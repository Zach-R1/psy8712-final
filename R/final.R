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
#library(tictoc) Used to initially test timing of code


n_cores <- detectCores() - 1 # Set core number by convention for later parallelization

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv") # read_csv chosen because it's faster and cleaner than read.csv

cleaned_data <- import_data %>% # Chose to clean through one long pipe for efficiency
  filter(!is.na(overall_rating)) %>% # Filtered to prevent observations with missing outcome variables as they won't be helpful for this analysis
  slice_sample(n = 10000) %>%  # Sampled data because 800k+ was too much for the laptop. Checked that proportions were similar to full data using prop.table(table(import_data$overall_rating)) and prop.table(table(cleaned$overall_rating)) Also sample_n was more efficient than sample, but slice_sample superseded sample_n
  mutate( 
    doc_id = row_number() # set doc_id as it's own colomn for later alignment and joining
    ) %>%
  unite(review, headline, pros, cons, sep = " ", na.rm = TRUE) %>% # See below
  select(overall_rating, review, doc_id )  # In an actual project I'd keep the other data, but since it isn't needed here I only selected necessary variables for ease and clarity.




# Unite note: This was a hard decision. Originally I planned to make three separate corpora for headline, pros, and cons. Since the goal of the assignment is the best model in terms of prediction I chose not to. It'd help with interpretability, but couldn't find any info suggesting it would help prediction. So the additional effort and processing time didnt seem worth it This also deals with the issue later in the process where the transformation process would leave blanks in headline so one less thing to fix


# Note: All of the below code from here to the topic_model was made originally to be able to take the full dataset, but timewise it wound up being unfeasable to continue that way. I maintained the code as is becuase it was more efficient.

# Dataset contains a decent enough emoji use that I didnt think it'd hurt to build a seperte transformation function. Also I wanted to see if I could get it to work.

emoji_transformer <- function(x) {
  x <- stri_replace_all_regex(x, "[\U0001F3FB-\U0001F3FF]", "") # Remove variants of emojis to only leave base version (thumbs up skin tone tags would lead them to be read as different emojis otherwise). Ignore low base rate emojis as sparsity removal should deal with them. Used regex to search through a range which fixed wouldnt do
  
  patterns <- c( # Set the emojis I wanted to replace via their unicode representation
    "\U0001F44D", "\U0001F44E", "\U0001F44C", "\U0001F525",
    "\U0001F60A", "\U0001F60C", "\U0001F601", "\U0001F642",
    "\U0001F61E", "\U0001F615", "\U0001F641"
  )
  
  replacements <- c( # Set the names to replace the emojis with additional space to avoid concatenation issues. Also chose to simplify labels for variants of smile and frown emojis to just smile and frown
    "thumbs up ", "thumbs down ", "okay ", "fire ",
    "smile ", "smile ", "smile ", "smile ",
    "frown ", "frown ", "frown "
  )
  
  x <- stri_replace_all_fixed(x, patterns, replacements, vectorize_all = FALSE) # Used fixed rather than regex becuase it's more efficient for exactly specified strings like the above. Used it to replace input matching pattern with its replacement.
  
  return(x) # Return the new emojiless string
}

# Test <- cleaned_data %>%
#   mutate(headline = emoji_transformer(cleaned_data$review)) # Tested functionality


sw_pattern <- paste0("\\b(", paste(c(stopwords("en"), "none", "con", "pro", ""), collapse = "|"), ")\\b") # builds pattern for later replacement. builds a single string from the "en" stopwords list and words I added as they were filler. Words seperated by | making it an easy regex pattern, with boundary markers \\b so that regex matches whole words rather than fragments of them. Thought about removing great and good, but when I look at the bigrams they are actually really helpful



cl <- makeCluster(n_cores) # Created cluster with max - 1 cores for parrallelization
clusterEvalQ(cl, { # load necisary libraries to each core
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
    replace_contraction() %>% # Remove contractions
    replace_abbreviation() %>% # remove abvs
    str_to_lower() %>% # set all to lower case. This has to be after the above as the two above this recapitalize the first letter in the string.
    str_remove_all("[[:punct:]]") %>% # regex implementation to remove punctuation was better suited to the processing outside of content transformers
    str_remove_all("[[:digit:]]") %>% # regex implementation to remove numbers was better suited to the processing outside of content transformers
    lemmatize_strings() %>% # althogh it gives me less control than token level lemmatization it is a lot faster than going with lemmatize words
    str_remove_all(regex(sw_pattern, ignore_case = TRUE)) %>% # Remove all strings matched via regex to the pattern established earlier
    str_squish() # basically removeWhitespace but better suited to the current set up.
})

stopCluster(cl)

processed_data <- cleaned_data %>%
  mutate(review_processed = unlist(results)) # WHAT ID DID and did not -select orignial reviews coloumn as i wanted easy access to compare the text pre and post processing


# Containerize
review_corpus <- VCorpus(VectorSource(processed_data$review_processed)) # wrap everything into a Vcorpus object so we can make a DTM


# dtm
my_tokenizer <- function(x) { NGramTokenizer(x, Weka_control(min=1, max=2)) } # make tokenizer That allows for unigram and bigrams

review_dtm <- DocumentTermMatrix( # make dtm
  review_corpus, # of the review cropus
  control = list(tokenize = my_tokenizer)) # with my prebuild tokenizer


#slim_dtm
review_slim_dtm <- removeSparseTerms(review_dtm, .9991) # Removed sparce terms with a cutoff that gave me a ratio between 3:1 and 2:1 based on the below commented out code 

# N <- nrow(review_dtm)
# N
# 
# lower_k <- N / 3  
# upper_k <- N / 2
# lower_k
# upper_k
# 
# k <- ncol(review_slim_dtm)
# k
# N / k


# Analysis

lc2 <- makeCluster(n_cores) # Made cluster

registerDoParallel(lc2) # registered parellel backend to alow  loops to execute across the 11 cores of the cluster

dtm_stm <- readCorpus( # Convert to a format that can be used with stm
  review_slim_dtm, type="slam")

kresult <- searchK( # fits models for specified values of k to provided diagnostics that help choose the appropriate number of topics
  dtm_stm$documents,
  dtm_stm$vocab,
  K = seq(2, 20, by = 2)) # set number of ks to attempt to ten


stopCluster(lc2) # Stop cluster to free resources
registerDoSEQ() # End parallel execution in backend and go back to sequential processing

plot(kresult)

topic_model <- stm(dtm_stm$documents, # fit the final topic model
                   dtm_stm$vocab, 
                   6, # The closest thing to an elbow I can find
                   verbose = FALSE)


labelTopics(topic_model, n = 6) # Looking at this if I could do this project fully with best practices I'd add more stop words like I'd need to think about whether to remove companies and more filler words. They don't see very informative.

docs_kept <- as.integer(names(dtm_stm$documents)) # Set docs_kept to later align different tbls of information based on original doc_ids and the entries that survived scarcity removal.

aligned_tbl <- processed_data[docs_kept, ] # create the actual alignment



findThoughts(topic_model, text = aligned_tbl$review_processed, n = 10) # give examples of each topic which made it esier to interprete what each was and better name them as just lists of words in the FLEX results can be a bit hard to interpret

plot(topic_model, type = "summary", n = 8) # Plots topic models
topicCorr(topic_model) # Provides intercorrelation of topic models
plot(topicCorr(topic_model)) # graphs relationships between modles

theta <- topic_model$theta # extracts probabilities for each topic


topic_labels <- c("Corporate Culture & Bureaucracy", "Generic positive comments", "Benefits & advancement oppertunities", "Poor management & stressful conditions", "Work-life balance", "Learning opportunities") # Set topic names

token_count <- slam::row_sums(review_slim_dtm[docs_kept, ]) # extracted token count

top_words <- review_slim_dtm %>% 
  slam::col_sums() %>% # col_sums was chosen over colSums as the base R version can't handle the dtm used to get frequency of each token
  sort(decreasing = TRUE) %>% # sort tokens so they are in order of most to least frequent
  head(200) # Take the top 200 tokens, remove nzv is getting rid of most of these anyways so if i do this now it saves me a lot of processing overhead in every model fitting that includes tokens. I'd try to find a way to use all of them in an actual project, but my computer was crashing when using that big a dataset.

top_words_tbl <- review_slim_dtm[as.character(docs_kept), names(top_words)] %>% # subset dtm to keep only surviving documents
  as.matrix() %>% # convert to space matrix
  as.tibble() %>% #Convert matrix to tibble
  mutate(doc_id = docs_kept, .before = 1) #add doc id as first colomn for easy joining later
  


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
  left_join(top_words_tbl, by = "doc_id") # joined topic and token data using doc id keeping all data






# embeddings 
embed_cols <- str_c("emb_", 1:768) # set cols for each individual embedding

embedding_df <- imap(cleaned_data$review, function(text, i) { # use element and index
  if (i %% 100 == 0) message(sprintf("Processing %d of %d", i, nrow(cleaned_data))) # consule progress report
  resp <- httr::POST( #send text to local ollama instance
    "http://localhost:11434/api/embed",
    body = jsonlite::toJSON(list(model = "nomic-embed-text", input = text), auto_unbox = TRUE),
    httr::content_type_json()
  )
  as.numeric(httr::content(resp)$embeddings[[1]])
}) %>%
  unname() %>% # strip name from list as a bug fix for later code
  do.call(cbind, .) %>% # bind the embedding vectors as a matrix
  t() %>% # matrix was sideways so transpose
  as_tibble(.name_repair = ~ embed_cols) %>% # convert to tiblle and assign the col names from earlier. .name_repair is an arg from as_tibble and ~ allows me to replace auto generated names with my col names
  mutate(doc_id = cleaned_data$doc_id) %>% # add doc id for later joining
  select(doc_id, everything())

embedding_df_filtered <- embedding_df %>% filter(doc_id %in% docs_kept) #filter so only embeddings from docs that survived sparcity removal are kept




full_data_tbl <- topics_tokens_tbl %>%
  left_join(embedding_df_filtered, by = "doc_id") %>% # join embedding to the rest of the dataset
  select(-original, -doc_id, -probability, -topic_label) # Remove all variables not used in the prediction. I left token count in as a control for the token models. I remember hereing that that practice iproves prediction in token models.


# saveRDS(full_data_tbl, "../out/data.RDS") # Save as requested

# full_data_tbl <- readRDS("../out/data.RDS") # code to import it as needed


# Analysis
train_index  <- createDataPartition(full_data_tbl$overall_rating, p = 0.75, list = FALSE) # partion data set for training/test split

training_data <- full_data_tbl[train_index, ] %>%
  mutate(dummy = 0) # Create training dataset
test_data <- full_data_tbl[-train_index, ] %>%
  mutate(dummy = 0) # Create holdout data


cross_val_control <- trainControl( # Create standard CV
  method = "cv",
  number = 5, # Reduced from 10 to reduce processing time, but would normally do 10 as the standard
  search = "random",
  verboseIter = TRUE
)


# Define predictor sets
token_cols <- c("token_count", names(top_words)) # select col token count and those that share names with top words so that if col order shifts or if i change the number of tokens to keep the code doesn't break
embed_vars <- paste0("emb_", 1:768) # gives me an easy way to reference these cols in a way robust to code changes
# This pattern continues below
token_data <- training_data %>% select(overall_rating, all_of(token_cols))
# Subsets
token_data <- training_data[, c("overall_rating", token_cols)]
topic_data <- training_data[, c("overall_rating", "topic", "dummy")]
embed_data <- training_data[, c("overall_rating", embed_vars)]
token_topic_data <- training_data[, c("overall_rating", "topic", token_cols)]
token_embed_data <- training_data[, c("overall_rating", token_cols, embed_vars)]
topic_embed_data <- training_data[, c("overall_rating", "topic", embed_vars)]
token_topic_embed_data <- training_data[, c("overall_rating", "topic", token_cols, embed_cols)]


# Define MTRY Values for each model to inject into function call
token_mtry <- c(33, 67, 100) # p/3/2, p/3, p/3 * 1.5 to calculate mtry values here and below
topic_mtry <- 1 # only one variable so only one mtry
embed_mtry <- c(256, 384, 512)
token_topic_mtry <- c(33, 67, 101)
token_embed_mtry <- c(256, 384, 584)
topic_embed_mtry <- c(256, 384, 512)
token_topic_embed_mtry <- c(256, 384, 585)



# Models
run_lm <- function(data) { # turn lm model training into function to avoid retyping
  train(
    overall_rating ~ .,
    data = as.data.frame(data),
    na.action = na.pass,
    method = "lm",
    preProcess = c("zv", "center", "scale", "medianImpute"),
    trControl = cross_val_control
  )
}


run_elastic <- function(data) { # turn elstic model training into function to avoid retyping
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

  # glmnet via caret requires >= 2 predictors so I needed another function to provide a dummy variable for the topic only model
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
  train( # turn random forest model training into function to avoid retyping
    overall_rating ~ .,
    data = as.data.frame(data),
    na.action = na.pass,
    method = "ranger",
    preProcess = c("zv", "center", "scale", "medianImpute"),
    tuneGrid = expand.grid(
      mtry = mtry_vals,
      splitrule = c("variance", "extratrees"),
      min.node.size = 20 # i'd normally do 5, but my laptop was hot enough to be concerning so I reduced it. I realize that I am sacrificing some quality here though
    ),
    trControl = cross_val_control,
    num.trees = 200 # reduced trees to ease processing. With a sample this big it probably does not make a huge deal, but would require more testing to be sure
  )
}


# Linear models
model1 <- run_lm(token_data) # You'll notice these are all their own line of code rather than a loop or apply. just a preference thing. I didnt want to add steps to what I just got working here at great effort.
model2 <- run_lm(topic_data)
model3 <- run_lm(embed_data)
model4 <- run_lm(token_topic_data)
model5 <- run_lm(token_embed_data)
model6 <- run_lm(topic_embed_data)
model7 <- run_lm(token_topic_embed_data)

lc3 <- makeCluster(n_cores) # make cluster
registerDoParallel(lc3) # set parralel processing

# Elastic net models
model8  <- run_elastic(token_data)
model9  <- run_elastic_simple(topic_data) # Run elastic with dummy code so that glmnet allows the use of topic alone
model10 <- run_elastic(embed_data)
model11 <- run_elastic(token_topic_data)
model12 <- run_elastic(token_embed_data)
model13 <- run_elastic(topic_embed_data)
model14 <- run_elastic(token_topic_embed_data)

# Random forest models
model15 <- run_rf(token_data, token_mtry)
model16 <- run_rf(topic_data, topic_mtry) 
model17 <- run_rf(embed_data, embed_mtry)
model18 <- run_rf(token_topic_data, token_topic_mtry)
model19 <- run_rf(token_embed_data, token_embed_mtry)
model20 <- run_rf(topic_embed_data, topic_embed_mtry)
model21 <- run_rf(token_topic_embed_data, token_topic_embed_mtry)

stopCluster(lc3) # stop cluster

registerDoSEQ() # go back to sequential processing

# Save all training models to a list
models <- list(model1, model2, model3, model4, model5, model6, model7,
               model8, model9, model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20, model21)


cv_results_tbl <- tibble( # create tbl
  model = c(rep("lm", 7), rep("elastic", 7), rep("rf", 7)), # create col label based on model type
  predictors = rep(c("token", "topic", "embeddings", "token_topic", "token_embed", "topic_embed", "token_topic_embed"), 3), # repeat predictor set once for every model type
  cv_Rsquared = map_dbl(models, ~ max(.x$results$Rsquared, na.rm = TRUE)), # loop over each model in models list, pull R^2 col from CV and take max. map_dbl ensures numeric vector as output rather than a list so as not to break the tibble
  cv_RMSE = map_dbl(models, ~ min(.x$results$RMSE, na.rm = TRUE))
) # Loop over the same but for RMSE and min since lower RMSE is better


ho_results_tbl <- tibble(
  model = c(rep("lm", 7), rep("elastic", 7), rep("rf", 7)), #same as above
  predictors = rep(c("token", "topic", "embeddings", "token_topic", "token_embed", "topic_embed", "token_topic_embed"), 3), # same as above
  # generates predictions on test data and then correlations between predicted and actual. then maps them to a tbl like the above
  ho_Rsquared = map_dbl(models, ~ cor(predict(.x, newdata = test_data, na.action = na.pass),  test_data$overall_rating)^2),
  ho_RMSE = map_dbl(models, ~ {
    preds <- predict(.x, newdata = test_data, na.action = na.pass)
    sqrt(mean((preds - test_data$overall_rating)^2))
  })
) # Same as above but RMSE instead of R^2

final_results_tbl <- cv_results_tbl %>% left_join(ho_results_tbl, by = c("model", "predictors")) # join the tbls for final information

# write.csv(final_results_tbl, "../out/final_results.csv") Saved for my benefit in case something crashed...again



# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?

# Answer: Yes, In all models used adding embeddings provided incremental validity over tokens alone in both the CV and holdout. Increased R^2 and reduced RMSE in both as for every model.

# Used in console: print(final_results_tbl %>% filter(predictors %in% c("token", "token_embed")))





#   RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?

# Answer: It depends on the model. For the linear model and elastic net models there was no noticable improvement in prediction when adding topic to existing tokenization strategy. In fact for the linear model it actually slightly reduced cross-validated R^2 (-0.004) though whether thats a meaningful difference is another question.

# For the Random Forest model, however, adding topic increased cross-validated R^2 by 0.01 and holdout R^2 by 0.021 which is improvement and it slightly reduced holdout RMSE which indicates reduced error both are good signs that topic should be added if using random forest. 

# Used in console: print(final_results_tbl %>% filter(predictors %in% c("token", "token_topic")))





#   RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?

# Answer: Harder to answer, there are some very minute shifts in CV R^2 and RMSE, but if we focus on the holdout we see no real meaningful shifts in R^2 or RMSE over the best individual model. Embeddings offer better prediction than topic which makes sense given how much info they provide. Topic doesnt seem to provide enough unique information to provide incremental validity over embeddings alone. Adding embeddings to topic doesnt make sense becuase embeddings alone gets you the same benefit while reducing variables by 1.

# Used in console: print(final_results_tbl %>% filter(predictors %in% c("topic", "embeddings", "topic_embed")))





#   RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?

# Answer: The best prediction of overall job satisfaction using the provided information is an elastic net model using tokens and embeddings. It has the largest overall holdout R^2 and lowest holdout RMSE, slightly out performing the elastic net model using all available variables.

# Used in console: final_results_tbl %>% arrange(desc(ho_Rsquared)) %>% head(6) 
# Used in console: and to be safe: final_results_tbl %>% arrange(ho_RMSE) %>% head(6)


save.image("../out/workspace.RData")

