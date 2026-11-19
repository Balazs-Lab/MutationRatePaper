################################################################################
# Data Analysis for Figure 3 Plots:
# MRA Analysis
################################################################################
# Load libraries
library(readr)
library(ggplot2)
library(readxl)
library(tidyr)
library(tidyverse)
library(dplyr)

# Set Directory
# setwd("~/")
setwd("~/Bioinformatics/ERR-Seq_2/MutRatePaper/scripts/figures")
################################################################################
## Load MRA Data
################################################################################

path_to_MRA = "../../data/models/mra/production/FINAL_mutation_supply_dataset.csv"
df <- read_csv(path_to_MRA)


################################################################################
## Extended Figure 3B - observed vs. posterior predictive check (PPC) predicted mutation counts by mutation type
################################################################################

ppc_data <- read_csv("../../data/models/mra/production/postfit_ppc_by_mutation_type.csv")

# select mut_type, obs_wt, pred_wt, and wt_ppc_ratio 
ppc_data <- ppc_data %>%
  dplyr::select(mut_type, obs_wt, pred_wt, wt_ppc_ratio)

# write to .csv 
write_csv(ppc_data, "../../figures/extended3/3B-ppc_by_mutation_type.csv")

# print the mean wt_ppc_ratio across mutaiton type
mean(ppc_data$wt_ppc_ratio, na.rm = TRUE)

################################################################################
## Figure 3A - Sliding Window Analysis - Observed Counts vs Full MRA
################################################################################

window_size = 250 
step_size = 25

# sliding window 
min_pos <- min(df$ref_position, na.rm = TRUE)
max_pos <- max(df$ref_position, na.rm = TRUE)

windows <- tibble(
  window_start = seq(min_pos, max_pos - window_size + 1, by = step_size)
) %>%
  mutate(
    window_end = window_start + window_size - 1,
    window_mid = (window_start + window_end) / 2
  )

summarize_one_window <- function(start, end, mid, data) {
  data %>%
    filter(ref_position >= start, ref_position <= end) %>%
    summarise(
      window_start = start,
      window_end = end,
      window_mid = mid,
     MRA_per_position = mean(bio_rate_median, na.rm = TRUE),
     Observed_per_position = sum(wt_count) / sum(wt_coverage),
      
      .groups = "drop"
    )
}

sliding_window_results <- pmap_dfr(
  list(
    windows$window_start,
    windows$window_end,
    windows$window_mid
  ),
  summarize_one_window,
  data = df
)

cor.test(sliding_window_results$MRA_per_position, sliding_window_results$Observed_per_position, use = "complete.obs", method = "pearson")
# write to .csv 
write_csv(sliding_window_results, "../../figures/figure3/3-sliding_window_results-pooled.csv")


################################################################################
## Figure 3A/B - Sliding Window Analysis - MRA mutation types
################################################################################
# perform sliding window analysis for G->A, nonG->A, Insertions, and Deletions
# 1 - assign mutation classes 
df_classed <- df %>%
  mutate(
    mutation_class = case_when(
      mut_type == "G->A" ~ "G_to_A",
      mut_type %in% c("insertion") ~ "insertion",
      mut_type %in% c("deletion") ~ "deletion",
      str_detect(mut_type, "->") & mut_type != "G->A" ~ "nonG_to_A_substitution",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(mutation_class))

rate_column <- "bio_rate_median"
# take the mean of the various substitions at each site
position_class_rates <- df_classed %>%
  group_by(ref_position, ref_nucleotide, mutation_class) %>%
  summarise(
    estimated_class_rate = mean(.data[[rate_column]], na.rm = TRUE),
    wt_count = sum(wt_count, na.rm = TRUE),
    wt_coverage = sum(wt_coverage, na.rm = TRUE),
    n_mutation_types_in_class = n(),
    .groups = "drop"
  )

summarize_one_window_type <- function(start, end, mid, data) {
  
  data %>%
    filter(ref_position >= start, ref_position <= end) %>%
    group_by(mutation_class) %>%
    summarise(
      window_start = start,
      window_end = end,
      window_mid = mid,
      MRA_estiamte = mean(estimated_class_rate, na.rm = TRUE),
      Observed_per_position = sum(wt_count) / sum(wt_coverage),
      
      
      .groups = "drop"
    )
}

summarize_one_window_type(1,250,150, data = position_class_rates)
sliding_window_results_type <- pmap_dfr(
  list(
    windows$window_start,
    windows$window_end,
    windows$window_mid
  ),
  summarize_one_window_type,
  data = position_class_rates
)

# melt the data for plotting - only keep summed_estimated_rate_per_position data
sliding_window_melted <- sliding_window_results_type %>%
  dplyr::select(window_mid, mutation_class, MRA_estiamte,Observed_per_position) %>%
  pivot_wider(
    names_from = mutation_class,
    values_from = c(MRA_estiamte,Observed_per_position)
  )

# save to csv
write_csv(sliding_window_melted, "../../figures/figure3/3-sliding_window_results-subclass.csv")

# For correlation table, only return non-overlapping positions (i.e. every 250th position)
sliding_window_melted <- sliding_window_results_type %>%
  filter(window_end %% 250 == 0) %>%
  dplyr::select(window_mid, mutation_class, MRA_estiamte,Observed_per_position) %>%
  pivot_wider(
    names_from = mutation_class,
    values_from = c(MRA_estiamte,Observed_per_position)
  )

sliding_window_results <- sliding_window_results %>%
  filter(window_end %% 250 == 0)

write_csv(sliding_window_results %>%
  filter(window_end %% 250 == 0), "../../figures/extended3/3C-sliding_window_results-subclass-nonoverlapping.csv")

# create a table and save the correlation output data
all <- cor.test(sliding_window_results$MRA_per_position, sliding_window_results$Observed_per_position, use = "complete.obs", method = "pearson")
GA <- cor.test(sliding_window_melted$MRA_estiamte_G_to_A, sliding_window_melted$Observed_per_position_G_to_A, use = "complete.obs", method = "pearson")
nGA <- cor.test(sliding_window_melted$MRA_estiamte_nonG_to_A_substitution, sliding_window_melted$Observed_per_position_nonG_to_A_substitution, use = "complete.obs", method = "pearson")
IN <- cor.test(sliding_window_melted$MRA_estiamte_insertion, sliding_window_melted$Observed_per_position_insertion, use = "complete.obs", method = "pearson")
DEL <- cor.test(sliding_window_melted$MRA_estiamte_deletion, sliding_window_melted$Observed_per_position_deletion, use = "complete.obs", method = "pearson")


# add data to table
cor_test_table <- tibble(
  mutation_class = character(),
  estimate = numeric(),
  p.value = numeric()
)
cor_test_table <- cor_test_table %>%
  add_row(mutation_class = "all", estimate = all$estimate, p.value = all$p.value) %>%
  add_row(mutation_class = "G->A", estimate = GA$estimate, p.value = GA$p.value) %>%
  add_row(mutation_class = "nonG->A", estimate = nGA$estimate, p.value = nGA$p.value) %>%
  add_row(mutation_class = "insertion", estimate = IN$estimate, p.value = IN$p.value) %>%
  add_row(mutation_class = "deletion", estimate = DEL$estimate, p.value = DEL$p.value)
cor_test_table
# write to csv
write_csv(cor_test_table, "../../figures/figure3/3-correlation_data.csv")



