################################################################################
# Data Analysis for Figure 5 Plots:
# MRA Analysis
################################################################################
library(dplyr)
library(tidyr)
library(readxl)
library(readr)
library(broom)
library(mgcv)
library(MASS)
set.seed(153)

################################################################################  
## Helper Functions
################################################################################
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

################################################################################  
## Load MRA Data
################################################################################

# read in mutation supply data
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
protease_fitness_long$protein <- "Pro"

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

### Protease Resistance Scores ###
protease_resistance_paths <- "../../data/mutation_supply/Proteaseresistance_score.xlsx"
protease_resistance_df <- read_xlsx(protease_resistance_paths) %>% dplyr::select(c(id,ScoreSum))
protease_resistance_df$id <- paste("Pro-",protease_resistance_df$id,sep = "")

### Merge Data ###

# inner join the first two 
# join left into prevalence
merged_data <- left_join(prevalence_data,supply_data, by = "id")

merged_data <- merged_data %>%
  left_join(protease_fitness_long, by = c("position" = "position", "alt_aa" = "aa","protein"))
head(merged_data)
merged_data$AA_mut <- paste(merged_data$ref_aa, merged_data$alt_aa, sep = "->")

# add aa_matix data
merged_data <- merged_data %>%
  left_join(aa_matrix, by = c("ref_aa" = "ref_aa", "alt_aa" = "alt_aa"))
protease_resistance_df$protein <- "Pro"
head(merged_data)

# add protease_resistance_df
#merged_data <- merged_data %>% left_join(protease_resistance_df, by = c("id" = "id", "protein" = "protein"))

# convert counts to frequency
merged_data$freq <- (merged_data$all_muts + 0.5) / merged_data$all_seqs

# drop any row with AA = NA
merged_data <- merged_data %>% filter(!is.na(mu_aa))

################################################################################
## Extended Figure 5A  - Number of Possible DRM Paths per Position for HIV Protease
################################################################################
# group data by AA_mut, return the mean mutation rate and mean frequency
merged_data %>% 
  filter(min_nt_changes == 1) %>%
  filter(protein == "Pro") %>%
  filter(Rx == 'naive') %>% 
  filter(DRM == 1) -> mut_type_df

# filter data for positions / proteins combinations with more than one Cons value 
choice_df <- merged_data %>%
  filter(Rx == 'naive') %>%
  filter(protein == "Pro") %>%
  filter(DRM == 1) %>%
  group_by(protein, Pos) %>%
  summarise(path_count = n()) 

# count the number of DRMs with 1, 2, 3, etc path counts
count_table <- data.frame(table(choice_df$path_count))

# group 2+ into one category (multiple)
count_table <- count_table %>%
  mutate(Var1 = as.numeric(as.character(Var1))) %>%
  mutate(Var1 = ifelse(Var1 >= 2, "2+", as.character(Var1))) %>%
  group_by(Var1) %>%
  summarise(Freq = sum(Freq)) %>%
  ungroup() %>%
  rename(path_count = Var1)

# write to .csv
write_csv(count_table, "../../figures/extended5/5A-DRM-Mutability-PathCount.csv")

################################################################################
## Figure 5A  - DRM Paths per Position for HIV Protease
################################################################################
# filter data for positions / proteins combinations with more than one Cons value 
choice_df <- merged_data %>%
  filter(Rx == 'naive') %>%
  filter(min_nt_changes == 1) %>%
  filter(protein == "Pro") %>%
  filter(DRM == 1) %>%
  group_by(protein, Pos) %>%
  summarise(path_count = n()) 

# filter mut_type_df to only include those positions / proteins with more than one Cons value
merged_data %>%
  filter(DRM == 1) %>%
  filter(min_nt_changes == 1) %>%
  full_join(choice_df, by = c("protein", "Pos")) %>%
  filter(protein == "Pro") %>%
  filter(Rx != 'naive') %>%
  filter(!is.na(mu_aa)) %>%
  filter(!is.na(Rx)) %>%
  group_by(Pos,aa_change) %>%
  summarize(
    Avearge_frequencey = mean(freq),
    Average_log_Mu = mean(log10_mu_aa)
  ) -> mut_type_df

# write .csv
write_csv(mut_type_df, "../../figures/figure5/5A-HIVProtease-DRM-byPos.csv")

################################################################################
## Extended Figure 5C - Model of DRM Frequency
################################################################################

# filter data for positions / proteins combinations with more than one Cons value 
choice_df <- merged_data %>%
  filter(Rx == 'naive') %>%
  filter(min_nt_changes == 1) %>%
  filter(protein == "Pro") %>%
  filter(DRM == 1) %>%
  group_by(protein, Pos) %>%
  summarise(path_count = n()) 

# filter mut_type_df to only include those positions / proteins with more than one Cons value
merged_data %>%
  filter(DRM == 1) %>%
  filter(min_nt_changes == 1) %>%
  full_join(choice_df, by = c("protein", "Pos")) %>%
  filter(path_count > 1) %>%
  filter(protein == "Pro") %>%
  filter(Rx != 'naive') %>%
  filter(!is.na(mu_aa)) %>%
  filter(!is.na(Rx)) -> mut_type_df
mut_type_df$Pos <- as.factor(mut_type_df$Pos)
mut_type_df$site = factor(paste(mut_type_df$protein, mut_type_df$Pos, mut_type_df$Cons, sep = "_"))
mut_type_df <- mut_type_df %>%
  left_join(protease_resistance_df, by = c("id" = "id"))

# model
mut_type_df$log_mu_aa <- log10(mut_type_df$mu_aa)
model_primary <- glm.nb(
  all_muts ~ log_mu_aa + fitness + Pos + Rx + offset(log(all_seqs)),
  data = mut_type_df
)
summary(model_primary)

stats_tbl <- summarise_propagation(model_primary, "DRM Frequency, protease, treated")

out_dir = "../../figures/extended5"
capture.output(summary(model_primary),
               file = file.path(out_dir, "5C-HIVProteaseDRMModelOutput.txt"))
write_csv(stats_tbl, file.path(out_dir, "5C-HIVProteaseDRMModelStats.csv"))

pts  <- adjusted_response(model_primary, mut_type_df, count_var = "all_muts", exposure_var = "all_seqs")
line <- propagation_band(model_primary, pts)

mut_type_df$adj_log_freq  <- pts$y
write_csv(mut_type_df, "../../figures/figure5/5C-Protease_DRM_Mutability-Freq-adjusted.csv")
write_csv(line,        "../../figures/figure5/5C-Protease_DRM_Mutability-Freq-regression-line.csv")


################################################################################
## Figure 5D - Variance Decomposition
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
                  terms = c("log_mu_aa", "fitness", "Pos", "Rx"),
                  response = "all_muts",
                  offset_expr = "offset(log(all_seqs))")
nb_lmg

# write to .csv
write_csv(nb_lmg, "../../figures/figure5/5D-HIVProteaseDRM-Model-VarianceDecomp.csv")


################################################################################
## Extended Figure 5D - DMR paths per site, extended
################################################################################

# group data by AA_mut, return the mean mutation rate and mean frequency
merged_data %>% 
  filter(min_nt_changes == 1) %>%
  filter(Rx == 'naive') %>% 
  filter(DRM == 1) -> mut_type_df

# filter data for positions / proteins combinations with more than one Cons value 
choice_df <- merged_data %>%
  filter(Rx == 'naive') %>%
  filter(DRM == 1) %>%
  group_by(protein, Pos) %>%
  summarise(path_count = n()) 

# count the number of DRMs with 1, 2, 3, etc path counts
count_table <- data.frame(table(choice_df$path_count))

# group 2+ into one category (multiple)
count_table <- count_table %>%
  mutate(Var1 = as.numeric(as.character(Var1))) %>%
  mutate(Var1 = ifelse(Var1 >= 2, "2+", as.character(Var1))) %>%
  group_by(Var1) %>%
  summarise(Freq = sum(Freq)) %>%
  ungroup() %>%
  rename(path_count = Var1)

# write to .csv
write_csv(count_table, "../../figures/extended5/5D-DRM-Mutability-PathCount-AllSites.csv")


################################################################################
## Figure 5E - Extended DRM Model
################################################################################
# filter data for positions / proteins combinations with more than one Cons value 
choice_df <- merged_data %>%
  filter(Rx == 'naive') %>%
  filter(min_nt_changes == 1) %>%
  filter(DRM == 1) %>%
  group_by(protein, Pos) %>%
  summarise(path_count = n()) 

# filter mut_type_df to only include those positions / proteins with more than one Cons value
merged_data %>%
  filter(DRM == 1) %>%
  filter(min_nt_changes == 1) %>%
  full_join(choice_df, by = c("protein", "Pos")) %>%
  filter(path_count > 1) %>%
  filter(Rx != 'naive') %>%
  filter(!is.na(mu_aa)) %>%
  filter(!is.na(Rx)) -> mut_type_df


mut_type_df$Pos <- as.factor(mut_type_df$Pos)
mut_type_df$site = factor(paste(mut_type_df$protein, mut_type_df$Pos, mut_type_df$Cons, sep = "_"))

# model
mut_type_df$log_mu_aa <- log10(mut_type_df$mu_aa)
model_primary <- glm.nb(
  all_muts ~ log_mu_aa + site + offset(log(all_seqs)),
  data = mut_type_df
)
summary(model_primary)

stats_tbl <- summarise_propagation(model_primary, "DRM Frequency, ALL, treated")

out_dir = "../../figures/extended5"
capture.output(summary(model_primary),
               file = file.path(out_dir, "5F-ALL-DRMModelOutput.txt"))
write_csv(stats_tbl, file.path(out_dir, "5F-ALL-DRMModelStats.csv"))

pts  <- adjusted_response(model_primary, mut_type_df, count_var = "all_muts", exposure_var = "all_seqs")
line <- propagation_band(model_primary, pts)

mut_type_df$adj_log_freq  <- pts$y
write_csv(mut_type_df, "../../figures/figure5/5E-ALL_DRM_Mutability-Freq-adjusted.csv")
write_csv(line,        "../../figures/figure5/5E-ALL_DRM_Mutability-Freq-regression-line.csv")






































