################################################################################
# Data Analysis for Figure 4 Plots:
# HIV Mutation Supply Constrains Amino Acid (AA) Mutability, Limiting AA Substitution Type Rate in vivo
################################################################################
# Load libraries
library(tidyr)
library(mgcv)
library(tibble)
library(readr)
library(readxl)
library(dplyr)
library(MASS)

set.seed(153)

setwd("~/Bioinformatics/ERR-Seq_2/MutRatePaper/scripts/figures")

################################################################################
## Figure 4A - MRA Distribution of mutation-type rates
################################################################################

# Load in full MRA dataset 
mra_path <- "../../data/models/mra/production/FINAL_mutation_supply_dataset.csv"
mra_df <- read.csv(mra_path)

# pivot wider using mut_type
mra_df_mut_type <- mra_df %>%
  dplyr::select(mut_type, bio_rate_median) %>%
  filter(!is.na(bio_rate_median)) %>%
  group_by(mut_type) %>%
  mutate(row_id = row_number()) %>%
  ungroup() %>%
  pivot_wider(
    id_cols = row_id,
    names_from = mut_type,
    values_from = bio_rate_median
  ) %>%
  dplyr::select(-row_id)


# sort by median value
mra_df_mut_type <- mra_df_mut_type %>% dplyr::select(all_of(names(sort(sapply(., median, na.rm = TRUE)))))

# remove NA
mra_df_mut_type <- mra_df_mut_type %>% dplyr::select(where(~ !all(is.na(.))))


# write to .csv
write_csv(mra_df_mut_type, "../../figures/figure4/4A-MRA_Distribution_Mutation_Type_Rates.csv")


################################################################################
## Extended Figure 4A - Mutation Type Variation
################################################################################
mra_df_mut_type_sd <- mra_df %>%
  dplyr::select(mut_type, bio_rate_median) %>%
  filter(!is.na(bio_rate_median)) %>%
  group_by(mut_type) %>% 
  summarise(sd = sd(log10(bio_rate_median))) %>%
  arrange(-sd)

write_csv(mra_df_mut_type_sd, "../../figures/extended4/4A-MRA_Distribution_Mutation_Type_Rates_SD.csv")


################################################################################
## Figure 4B Analysis - MRA Distribution of Nucleotide Context for G->A Muts
################################################################################

# pivot wider using mut_type
mra_df_3p <- mra_df %>% 
  filter(mut_type == "G->A") %>%
  dplyr::select(context_3p, bio_rate_median) %>%
  filter(!is.na(bio_rate_median)) %>%
  group_by(context_3p) %>%
  mutate(row_id = row_number()) %>%
  ungroup() %>%
  pivot_wider(
    id_cols = row_id,
    names_from = context_3p,
    values_from = bio_rate_median
  ) %>%
  dplyr::select(-row_id)



# sort by median value
mra_df_3p <- mra_df_3p %>% dplyr::select(all_of(names(sort(sapply(., median, na.rm = TRUE)))))

# write to .csv
write_csv(mra_df_3p, "../../figures/figure4/4B-MRA_Distribution_GA_3p.csv")


################################################################################
## Figure 4C Analysis - HIV AA Mutability Plot
################################################################################

# Load in full MRA Mutation Dataset 
aa_path <- "../../data/models/mra/production/comprehensive_analysis_mra_product/aa_level_supply_table.csv"
aa_df <- read.csv(aa_path)

# create a matrix for each aa mutation type, with the value represting the mean mutation rate for that aa chang

aa_matrix <- aa_df %>%
  dplyr::select(ref_aa, alt_aa, mu_aa) %>%
  filter(!is.na(mu_aa)) %>%
  group_by(ref_aa, alt_aa) %>%
  summarise(mean_mutation_rate = log10(mean(mu_aa, na.rm = TRUE)), .groups = "drop") %>%
  pivot_wider(
    id_cols = ref_aa,
    names_from = alt_aa,
    values_from = mean_mutation_rate
  ) 
# write to .csv 
write_csv(aa_matrix, "../../figures/figure4/4C-AA_mutability.csv")

################################################################################
## Extended Figure 4B - AA Changes by Number of Mutations
################################################################################

aa_df %>% 
  group_by(protein,min_nt_changes) %>%
  summarise(nt_change = n()) %>%
  mutate(fraction_of_group = nt_change / sum(nt_change)) %>%
  dplyr::select(-nt_change) %>%
  group_by(min_nt_changes) %>%
  summarise(mean = mean(fraction_of_group)*100,sd = sd(fraction_of_group)*100) -> nt_changes_df 

write_csv(nt_changes_df, "../../figures/extended4/4B-MutationNTchanges.csv")


################################################################################  
## Figure 4E Analysis - Load MRA Related Data
################################################################################

# read in mutation supply data - correcetly aligned to Variant Data
supply_path <- "../../data/mutation_supply/mra_variant_supply.csv"
supply_data <- read_csv(supply_path)
# concatenate ref_aa position and alt_aa to be the row id 
supply_data$id <- paste(supply_data$protein,"-",supply_data$ref_aa, supply_data$position, supply_data$alt_aa, sep = "")
supply_data <- supply_data %>%
  dplyr::select(-c(protein, -ref_aa, -position, -alt_aa))

### SDR-DB Variant Frequencies ###
# read in mutation frequency
prevalence_path <- "../../data/mutation_supply/variant_prevalence.xlsx"
prevalence_data <- read_xlsx(prevalence_path)
# concatenate Cons Pos and Mut to be the row id 
prevalence_data$id <- paste(prevalence_data$protein,"-",prevalence_data$Cons, prevalence_data$Pos, prevalence_data$Mut, sep = "")

### Protease Fitness Data ###
## add fitness data
protease_fitess_path <-  "../../data/mutation_supply/ProteaseFitness.xls"
protease_fitness_data <- read_xls(protease_fitess_path)
# melt all cols except position 
protease_fitness_long <- protease_fitness_data %>%
  pivot_longer(cols = -position, names_to = "aa", values_to = "fitness") 

### AA Metadata ###
metadata_file_path <- "../../data/mutation_supply/amino_acid_table.csv"
aa_data <- read.csv(metadata_file_path) %>% dplyr::select(aa,type_code)
# create a matrix of amino acid properties for each possible aa change (concat the type_codes)
aa_matrix <- expand.grid(ref_aa = aa_data$aa, alt_aa = aa_data$aa) %>%
  left_join(aa_data, by = c("ref_aa" = "aa")) %>%
  rename(ref_type_code = type_code) %>%
  left_join(aa_data, by = c("alt_aa" = "aa")) %>%
  rename(alt_type_code = type_code) %>%
  mutate(mutation_type = ifelse(ref_type_code == alt_type_code, "similar", "different")) %>%
  mutate(change_class = paste(ref_type_code, alt_type_code, sep = "->")) 

### Merge Data - Protease Only ###
# inner join the first two 
merged_data <- left_join(supply_data, prevalence_data, by = "id") %>%
  filter(protein == "Pro")

merged_data <- merged_data %>%
  left_join(protease_fitness_long, by = c("position" = "position", "alt_aa" = "aa"))
head(merged_data)
merged_data$AA_mut <- paste(merged_data$ref_aa, merged_data$alt_aa, sep = "->")

# add aa_matix data
merged_data <- merged_data %>%
  left_join(aa_matrix, by = c("ref_aa" = "ref_aa", "alt_aa" = "alt_aa"))

head(merged_data)

# convert counts to frequency
merged_data$freq <- (merged_data$all_muts + 0.1) / merged_data$all_seqs




################################################################################
## Mutation Type Model dataframe 
################################################################################

# group data by AA_mut, return the mean mutation rate and mean frequency
merged_data %>% 
  group_by(AA_mut,min_nt_changes) %>%
  filter(Rx == 'naive') %>%
  filter(min_nt_changes == 1) %>%
  summarise(mean_mu_aa = mean(mu_aa, na.rm = TRUE),
            mean_freq = sum(all_muts, na.rm = TRUE)/sum(all_seqs, na.rm = TRUE),
            total_counts= sum(all_muts, na.rm = TRUE),
            total_sequences = sum(all_seqs, na.rm = TRUE),
            avg_fitness = mean(fitness, na.rm = TRUE),
            change_class = first(mutation_type),
            n_muts = n()) -> mut_type_df

################################################################################
## Extended Figure 4D - Distribution of Mutation Type Frequencies
################################################################################


n_bins <- 15   
log_freq <- log10(mut_type_df$mean_freq)
h <- hist(log_freq, breaks = n_bins, plot = FALSE)
mut_type_hist <- data.frame(bin_center = h$mids, Freq = h$counts)
write_csv(mut_type_hist, "../../figures/extended4/4D-MutationTypeHist.csv")


################################################################################
## Extended Figure 4E - Distribution of Mutation Type Fitness Costs
################################################################################


n_bins <- 15   
fitness_cost <- mut_type_df$avg_fitness
h <- hist(fitness_cost, breaks = n_bins, plot = FALSE)
mut_type_hist <- data.frame(bin_center = h$mids, Freq = h$counts)
write_csv(mut_type_hist, "../../figures/extended4/4E-MutationTypeFitnessCost.csv")

################################################################################
## Extended Figure 4F - Distribution of Mutation Type Mutability
################################################################################


n_bins <- 25
mutability <- log10(mut_type_df$mean_mu_aa)
h <- hist(mutability, breaks = n_bins, plot = FALSE)
mutability_hist <- data.frame(bin_center = h$mids, Freq = h$counts)
write_csv(mutability_hist, "../../figures/extended4/4F-MutationTypeMutability.csv")


################################################################################
## Extended Figure 4G - Distribution of Mutation Type Chemistry Changes
################################################################################


change_class <- mut_type_df$change_class
chemchange_df <- data.frame(table(change_class))
write_csv(chemchange_df, "../../figures/extended4/4G-MutationTypeChemistryClassChange.csv")


################################################################################
## Figure 4 Model - Mutation Type Rates of supply data -  1 step
################################################################################
mut_type_df$log_mu_aa <- log10(mut_type_df$mean_mu_aa)
model_primary <- glm.nb(
  total_counts ~ log_mu_aa + avg_fitness + change_class + offset(log(total_sequences)),
  data = mut_type_df
)
summary(model_primary)



summarise_propagation <- function(fit, label, mu_var = "log_mu_aa") {
  s  <- summary(fit)$coefficients[mu_var, ]
  b  <- s[[1]] / log(10)      # natural-log response -> log10 per log10 mutability
  se <- s[[2]] / log(10)
  ## null.deviance already carries the offset and the fitted theta, so this is
  ## the coherent comparison. Do NOT refit the null with glm.nb(): it
  ## re-estimates theta and the deviances stop being on the same scale.
  data.frame(
    label              = label,
    n                  = stats::nobs(fit),
    beta_log10         = b,
    se_log10           = se,
    ci_lower           = b - 1.96 * se,
    ci_upper           = b + 1.96 * se,
    fold_per_10x       = 10^b,
    p_vs_0             = 2 * pnorm(-abs(b / se)),
    p_vs_1             = 2 * pnorm(-abs((b - 1) / se)),
    deviance_explained = 1 - deviance(fit) / fit$null.deviance,
    theta              = fit$theta,
    row.names = NULL
  )
}

stats_tbl <- summarise_propagation(model_primary, "AA change types, protease, naive")

out_dir = "../../figures/extended4"
capture.output(summary(model_primary),
               file = file.path(out_dir, "4H-MutationTypeModelOutput.txt"))
write_csv(stats_tbl, file.path(out_dir, "4H-MutationTypeModelStats.csv"))

dev_expl <- 1 - deviance(model_primary) / model_primary$null.deviance


################################################################################
# Figure 4E Analysis - variance decomposition of model 
################################################################################

## LMG / Shapley decomposition for a GLM (incl. negative binomial) ------------
## Averages the incremental DEVIANCE explained by each term over all orderings
## of the predictors. Equivalent to relaimpo::calc.relimp(type="lmg", rela=TRUE)
## when family = gaussian() (verified to machine precision).
##
## Two requirements specific to negative binomial:
##  * theta is held FIXED at the full-model value for every subset fit. Letting
##    glm.nb() re-estimate it per subset puts the deviances on different scales.
##  * the offset is carried into every subset including the null model.
##
## Shares are of EXPLAINED deviance (they sum to 1), not of total deviance.
## They are sample-dependent descriptive quantities; at n ~ 130 the bootstrap
## CIs overlap heavily and they cannot be used to rank predictors.
lmg_glm <- function(fit, data, terms, response, offset_expr = NULL, theta = NULL) {
  fam <- if (inherits(fit, "negbin")) {
    MASS::negative.binomial(theta = if (is.null(theta)) fit$theta else theta)
  } else family(fit)
  rhs <- function(tt) { p <- c(tt, offset_expr); if (!length(p)) "1" else paste(p, collapse = " + ") }
  cache <- new.env()
  D <- function(tt) {
    key <- if (!length(tt)) "__null__" else paste(sort(tt), collapse = "|")
    if (!is.null(cache[[key]])) return(cache[[key]])
    v <- deviance(glm(as.formula(paste(response, "~", rhs(tt))), family = fam, data = data))
    assign(key, v, envir = cache); v
  }
  perms <- function(v) if (length(v) <= 1) list(v) else
    do.call(c, lapply(seq_along(v), function(i) lapply(perms(v[-i]), function(p) c(v[i], p))))
  P <- perms(terms); inc <- setNames(numeric(length(terms)), terms)
  for (p in P) { cur <- character(0); prev <- D(cur)
  for (tm in p) { cur <- c(cur, tm); now <- D(cur); inc[tm] <- inc[tm] + (prev - now); prev <- now } }
  inc <- inc / length(P); total <- D(character(0)) - D(terms)
  tibble::tibble(predictor = terms,
                 share_of_explained_deviance = as.numeric(inc / total),
                 percent_of_explained = 100 * as.numeric(inc / total),
                 model_deviance_explained = 1 - D(terms) / D(character(0))) |>
    dplyr::arrange(dplyr::desc(share_of_explained_deviance))
}

nb_lmg <- lmg_glm(model_primary, mut_type_df,
                  terms = c("log_mu_aa", "avg_fitness", "change_class"),
                  response = "total_counts",
                  offset_expr = "offset(log(total_sequences))")


# write to .csv
write_csv(nb_lmg, "../../figures/figure4/4E-AA_Mutability-Fitness-Freq_model_decomposition.csv")



################################################################################
## Figure 4F Analysis - Scatter Plot and Propagation Band
################################################################################

adjusted_response <- function(fit, data, mu_var = "log_mu_aa", count_var= "all_muts",exposure_var = "all_seqs", pseudo = 0.5) {
  stopifnot(mu_var %in% names(coef(fit)))
  b <- coef(fit)[[mu_var]]
  x <- data[[mu_var]]; y <- data[[count_var]]; N <- data[[exposure_var]]
  stopifnot(all(N > 0), length(x) == nrow(data))
  
  lp_other <- predict(fit, type = "link") - b * x - log(N)
  
  ## pseudo is for DISPLAY ONLY: the model was fitted on raw counts, so
  ## zero-count rows are genuine observations rather than dropped rows.
  adj <- log10((y + pseudo) / N) - (lp_other - mean(lp_other)) / log(10)
  data.frame(x = x, y = adj, zero = (y == 0))
}

propagation_band <- function(fit, pts, mu_var = "log_mu_aa", n = 100) {
  b  <- coef(fit)[[mu_var]] / log(10)
  se <- summary(fit)$coefficients[mu_var, 2] / log(10)
  xb <- mean(pts$x); yb <- mean(pts$y)
  x  <- seq(min(pts$x), max(pts$x), length.out = n)
  data.frame(x        = x,
             y_hat    = yb + b * (x - xb),
             y_lower  = yb + b * (x - xb) - 1.96 * se * abs(x - xb),
             y_upper  = yb + b * (x - xb) + 1.96 * se * abs(x - xb),
             y_slope1 = yb + 1.0 * (x - xb))
}

pts  <- adjusted_response(model_primary, mut_type_df, count_var = "total_counts", exposure_var = "total_sequences")
line <- propagation_band(model_primary, pts)

mut_type_df$adj_log_freq  <- pts$y
mut_type_df$log_mean_freq <- log10(mut_type_df$mean_freq)
write_csv(mut_type_df, "../../figures/figure4/4F-AA_Mutability-Freq-adjusted.csv")
write_csv(line,        "../../figures/figure4/4F-AA_Mutability-Freq-regression-line.csv")
