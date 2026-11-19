################################################################################
# Data Analysis for Figure 1 Plots:
# MRA Analysis
################################################################################
library(dplyr)
library(tidyr)
library(readxl)
library(readr)
set.seed(153)

################################################################################  
## Load Raw ERRSeq Data
################################################################################
all_mutations_df <- read_csv("../../data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/all_mutations.csv")
consensus_df <- read_csv("../../data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/consensus_sequence_data.csv")
annotated_muts_df <- read_csv("../../data/pipeline/11-analysis/coding_mutations.csv")

################################################################################  
## Label and Annotate Data
################################################################################

# label samples and controls
consensus_df %>% 
  group_by(Sample) %>% 
  mutate(condition = ifelse(grepl("ePCR", Sample), "control", "sample")) %>% 
  ungroup() -> consensus_df

all_mutations_df %>% 
  group_by(sample_id) %>% 
  mutate(condition = ifelse(grepl("ePCR", sample_id), "control", "sample")) %>% 
  ungroup() -> all_mutations_df

# Count total UMI_ID by condition 
consensus_df %>% 
  group_by(condition) %>% 
  summarise(total_UMI_ID = n_distinct(UMI_ID)) -> total_UMI_ID_df

# split sample_df by "-"
consensus_df %>% 
  separate(Sample, into = c("Strain", "library","group","value","rep"), sep = "-", remove = FALSE) -> consensus_df

# create gagpol and env groups if #library is LP252 
consensus_df %>% 
  mutate(group = ifelse(grepl("LP252", library), "gagpol", "env_plus")) -> consensus_df

consensus_df %>% 
  dplyr::mutate(rep_UMI = paste0(rep, "_", UMI_ID)) -> consensus_df

################################################################################  
## Figure 1D - Aggregate REC data 
################################################################################

consensus_df %>% 
  group_by(condition,group) %>% 
  summarise(total_UMI_ID = n_distinct(rep_UMI)) -> total_UMI_ID_df

total_UMI_ID_df
# write to csv 
write.csv(total_UMI_ID_df, "../../figures/figure1/1D-total_UMI_ID_by_condition_and_group.csv", row.names = FALSE)


################################################################################  
## Figure 1E - Raw Mutation Rates - using bootsrapped mutation rates
################################################################################
total_UMI_ID_df

# annotate groups 
all_mutations_df %>% 
  dplyr::mutate(group = ifelse(grepl("LP252", library_name), "gagpol", "env_plus")) -> all_mutations_df
# annotate condition
all_mutations_df %>%
  dplyr::mutate(condition = ifelse(grepl("ePCR", sample_id), "control", "sample")) -> all_mutations_df

# sample n rows from consensus_df and compute mutation rate
bootstrap_sample <- function (df, n) {
  df %>% 
    dplyr::group_by(condition,group) %>%
    dplyr::sample_n(n) %>%
    dplyr::summarise(total_RECs = n_distinct(UMI_ID), 
              total_muts = sum(Total_Mutations),
              total_bases = sum(Total_Bases),
              mutation_rate = total_muts/total_bases,
              .groups = "drop")
}

# perform 500 bootstraps of size 100 and store the results in a data frame
output_mutatio_data <- data.frame()
samples = 500 
sample_size = 100
for (i in 1:samples) {
  bootstrap_result <- bootstrap_sample(consensus_df, sample_size)
  output_mutatio_data <- rbind(output_mutatio_data, bootstrap_result)
}  
# summary data from output_mutatio_data
output_mutatio_data %>% 
  dplyr::group_by(condition,group) %>%
  dplyr::summarise(n = n(),
            mean_mutation_rate = mean(mutation_rate),
            sd_mutation_rate = sd(mutation_rate),
            se = sd_mutation_rate / sqrt(n),
            lower_ci = mean_mutation_rate - qt(0.975, df = n - 1)*se,
            upper_ci = mean_mutation_rate + qt(0.975, df = n - 1)* se ,
            .groups = "drop") -> summary_mutation_data

summary_mutation_data
# reorder columns
summary_mutation_data <- summary_mutation_data %>% 
  select(condition, group, n, mean_mutation_rate, upper_ci, lower_ci, sd_mutation_rate, se)                                                          
write.csv(summary_mutation_data, "../../figures/figure1/1E-mutation_rate_by_condition_and_group.csv", row.names = FALSE)



################################################################################  
## Figure 1F / G - Distribution of Mutations per REC 
################################################################################
samples = 100 
sample_size = 1000
# create a function to sample n rows from consensus_df and compute mutations per REC

bootstrap_distribution <- function(df, n) {
  df %>% 
    sample_n(n) %>%
    group_by(Total_Mutations) %>%
    summarise(count = n(),total_muts = sum(Total_Mutations)) %>%
    ungroup() %>%
    dplyr::mutate(Total_Mutation = ifelse(Total_Mutations >= 10, 10, Total_Mutations)) %>%
    group_by(Total_Mutation) %>%
    summarise(count = sum(count),total_muts = sum(total_muts)) %>% 
    ungroup() %>%
    mutate(percent_recs = count / sum(count) * 100,
           percent_muts = total_muts / sum(total_muts) * 100, 
          .groups = "drop") 
  
}

distribution_table = data.frame()
for (i in 1:samples) {
  bootstrap_result <- bootstrap_distribution(consensus_df %>% filter(condition == "sample"), sample_size)
  distribution_table <- rbind(distribution_table, bootstrap_result)
}

# summarize distribution table 
distribution_table %>% 
  dplyr::group_by(Total_Mutation) %>%
  dplyr::summarise(n = n(),
            mean_percent_recs = mean(percent_recs),
            sd_percent_recs = sd(percent_recs),
            se_percent_recs = sd_percent_recs / sqrt(n),
            lower_ci_percent_recs = mean_percent_recs - qt(0.975, df = n - 1)*se_percent_recs,
            upper_ci_percent_recs = mean_percent_recs + qt(0.975, df = n - 1)* se_percent_recs,
            mean_percent_muts = mean(percent_muts),
            sd_percent_muts = sd(percent_muts),
            se_percent_muts = sd_percent_muts / sqrt(n),
            lower_ci_percent_muts = mean_percent_muts - qt(0.975, df = n - 1)*se_percent_muts,
            upper_ci_percent_muts = mean_percent_muts + qt(0.975, df = n - 1)* se_percent_muts,
            .groups = "drop") -> summary_distribution_table
summary_distribution_table$Total_Mutation_Index <- c(0:9, "10+")

# order columns
summary_distribution_table <- summary_distribution_table %>% 
  dplyr::select(Total_Mutation, Total_Mutation_Index, n, 
         mean_percent_recs, upper_ci_percent_recs, lower_ci_percent_recs,
         mean_percent_muts, upper_ci_percent_muts, lower_ci_percent_muts)
summary_distribution_table
# write to .csv
write.csv(summary_distribution_table, "../../figures/figure1/1F-G-mutations_per_REC.csv", row.names = FALSE)



################################################################################  
## Extended Figure 1 B - Distribution of Mutations per REC - Control Data
################################################################################
# perform the same analysis for the control data

distribution_table_control = data.frame()
for (i in 1:samples) {
  bootstrap_result <- bootstrap_distribution(consensus_df %>% filter(condition == "control"), sample_size)
  distribution_table_control <- rbind(distribution_table_control, bootstrap_result)
}

# summarize distribution table 
distribution_table_control %>% 
  group_by(Total_Mutation) %>%
  summarise(n = n(),
            mean_percent_recs = mean(percent_recs),
            sd_percent_recs = sd(percent_recs),
            se_percent_recs = sd_percent_recs / sqrt(n),
            lower_ci_percent_recs = mean_percent_recs - qt(0.975, df = n - 1)*se_percent_recs,
            upper_ci_percent_recs = mean_percent_recs + qt(0.975, df = n - 1)* se_percent_recs,
            mean_percent_muts = mean(percent_muts),
            sd_percent_muts = sd(percent_muts),
            se_percent_muts = sd_percent_muts / sqrt(n),
            lower_ci_percent_muts = mean_percent_muts - qt(0.975, df = n - 1)*se_percent_muts,
            upper_ci_percent_muts = mean_percent_muts + qt(0.975, df = n - 1)* se_percent_muts,
            .groups = "drop") -> distribution_table_control_control
distribution_table_control_control

# order columns
distribution_table_control_control <- distribution_table_control_control %>% 
  select(Total_Mutation, n, 
         mean_percent_recs, upper_ci_percent_recs, lower_ci_percent_recs,
         mean_percent_muts, upper_ci_percent_muts, lower_ci_percent_muts)
distribution_table_control_control
# write to .csv
write.csv(distribution_table_control_control, "../../figures/extended1/1B-control-mutations_per_REC.csv", row.names = FALSE)



################################################################################
## Figure 1H - Aggregate Mutation Types
################################################################################

annotated_muts_df$classification
annotated_muts_df %>%
  filter(mutation_type == "substitution") %>%
  mutate(condition = ifelse(grepl("ePCR", sample_id), "control", "sample")) %>%
  filter(condition == "sample") %>%
  group_by(classification) %>%
  summarise(total_muts = n()) %>%
  ungroup() %>%
  mutate(percent_muts = total_muts / sum(total_muts) * 100) -> mutation_type_df

mutation_type_df

# write csv
write.csv(mutation_type_df, "../../figures/figure1/1H-mutation_types.csv", row.names = FALSE)

################################################################################
## Figure 1I - Mutation Type Analysis - Fraction of Nonsense Mutations
################################################################################

# create a bootstrap function to sample n rows from consensus_df and compute fraction of lethal mutations
sample_size = 10
samples = 100

# update is_lethal to be tru if calssification is nonsense
annotated_muts_df$is_lethal <- ifelse(annotated_muts_df$classification == "nonsense", TRUE, FALSE)

lethal_bootstrap_sample <- function(df, n, total_muts) {
  
  if (total_muts >= 10) {
    df %>% 
      filter(condition == "sample") %>%
      filter(Total_Mutations >= total_muts) %>% 
      sample_n(n,replace = TRUE) %>%
      select(Sample,UMI_ID) -> single_mutation_df
    
  }
  else {
    
    df %>% 
    filter(condition == "sample") %>%
    filter(Total_Mutations == total_muts) %>% 
    sample_n(n,replace = TRUE) %>%
    select(Sample,UMI_ID) -> single_mutation_df
  
  }

  single_mutation_df %>% 
    inner_join(annotated_muts_df %>% filter(mutation_type == "substitution"), by = c("Sample" = "sample_id", "UMI_ID" = "umi_id")) %>% 
    group_by(UMI_ID) %>%
    summarise(lethal = sum(is_lethal)) %>%
    group_by(lethal) %>%
    summarise(fraction_lethal = n()) %>% 
    mutate(percent_lethal = fraction_lethal / sum(fraction_lethal) * 100) %>% 
    mutate(lethal_group = ifelse(lethal > 0, "lethal", "non-lethal")) %>%
    group_by(lethal_group) %>%
    summarise(total_muts = sum(fraction_lethal)) %>%
    mutate(percent_muts = total_muts / sum(total_muts) * 100)
}

# perform 100 bootstraps and store the results in a data frame
# do this for total_muts = 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10+
output_lethal <- data.frame()
total_mut_list <- c(1,2,3,4,5,6,7,8,9,10)
for (total_muts in total_mut_list) {
  for (i in 1:samples) {
    bootstrap_result <- lethal_bootstrap_sample(consensus_df, sample_size, total_muts)
    bootstrap_result$total_muts_group <- total_muts
    output_lethal <- rbind(output_lethal, bootstrap_result)
  }
}  
output_lethal

# summary for output_lethal
output_lethal %>% 
  group_by(lethal_group,total_muts_group) %>%
  filter(lethal_group == "lethal") %>%
  summarise(n = n(),
            mean_percent_muts = mean(percent_muts),
            sd_percent_muts = sd(percent_muts),
            se_percent_muts = sd_percent_muts / sqrt(n),
            lower_ci_percent_muts = mean_percent_muts - qt(0.975, df = n - 1)*se_percent_muts,
            upper_ci_percent_muts = mean_percent_muts + qt(0.975, df = n - 1)* se_percent_muts, 
            .groups = "drop") -> summary_lethal_df

# reorder columns
summary_lethal_df <- summary_lethal_df %>% 
  select(lethal_group, total_muts_group, n, mean_percent_muts, upper_ci_percent_muts, lower_ci_percent_muts, sd_percent_muts, se_percent_muts)


summary_lethal_df
# write csv
write.csv(summary_lethal_df, "../../figures/figure1/1I-fraction_lethal_mutations.csv", row.names = FALSE)

