# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
# Lib
# Lib

# Data Import and Cleaning
import_data <- read_csv("../data/glassdoor_reviews.csv")

cleaned_data <- import_data %>%
  filter(!is.na(overall_rating)) %>% # Filter to ensure zero missing data in outcome
  select(overall_rating, headline, pros, cons) # In an actual project I'd keep the other data, but sisnce it isnt needed here I only selected necisary variables to reduce processing time.

# Analysis