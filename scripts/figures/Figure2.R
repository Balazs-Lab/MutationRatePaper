################################################################################
# Data Analysis for Figure 2 Plots: Site Specific Mutation Accumulation
################################################################################
library(dplyr)
library(tidyr)
library(readxl)
library(readr)
set.seed(6174)

################################################################################  
## Load Raw ERRSeq Data
################################################################################

all_mutations_df <- read_csv("../../data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/all_mutations.csv")
consensus_df <- read_csv("../../data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/consensus_sequence_data.csv")

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
  separate(Sample, into = c("Strain", "library","group","value","rep","other"), sep = "-", remove = FALSE) -> consensus_df

consensus_df %>% 
  dplyr::mutate(rep_UMI = paste0(rep, "_", UMI_ID)) -> consensus_df

consensus_df %>% group_by(condition,library) %>% 
  summarise(total_UMI_ID = n_distinct(UMI_ID)) -> total_UMI_ID_df

# Keep the same mutation rows mra_build_model_input.py writes into the model table:
# NL4-3 libraries, WT/control groups, RECs with <=1000 mutations, the library-position
# trim, a covered reference position, and a mutation type on the model grid.
edge_trim <- 20
model_groups <- c("WT", "NEG", "Sparsomycin", "pUC19", "ePCR")

read_library_map <- function(lib) {
  pos_col <- paste0(lib, "_pos")
  base_col <- paste0(lib, "_base")
  read_csv(
    sprintf("../../data/pipeline/00-reference-map/%s.csv", lib),
    show_col_types = FALSE
  ) %>%
    transmute(
      ref_position = as.integer(ref_pos),
      ref_nucleotide = ref_base,
      library_position = as.integer(.data[[pos_col]]),
      library_base = .data[[base_col]]
    ) %>%
    filter(!is.na(library_position), ref_nucleotide == library_base)
}

library_maps <- lapply(c("LP249", "LP252", "LP253"), read_library_map)
names(library_maps) <- c("LP249", "LP252", "LP253")

# Reference positions that survive the reference-map trim. This is the position set
# in mutation_supply_data.csv.
covered_refs <- bind_rows(lapply(library_maps, function(map) {
  max_pos <- max(map$library_position)
  map %>%
    filter(
      library_position > edge_trim,
      library_position < (max_pos - edge_trim)
    ) %>%
    dplyr::select(ref_position, ref_nucleotide)
})) %>%
  distinct()

consensus_df$sample_id <- consensus_df$Sample
consensus_df$umi_id <- consensus_df$UMI_ID
all_mutations_df <- all_mutations_df %>%
  inner_join(
    consensus_df %>% dplyr::select(sample_id, umi_id, group, Total_Mutations),
    by = c("sample_id", "umi_id")
  ) %>%
  filter(
    library_name %in% names(library_maps),
    group %in% model_groups,
    !is.na(Total_Mutations),
    Total_Mutations <= 1000
  )

lib_max <- all_mutations_df %>%
  group_by(library_name) %>%
  summarise(max_library_position = max(library_position, na.rm = TRUE), .groups = "drop")

all_mutations_df <- all_mutations_df %>%
  inner_join(lib_max, by = "library_name") %>%
  filter(
    library_position > edge_trim,
    library_position < (max_library_position - edge_trim)
  ) %>%
  inner_join(covered_refs, by = "ref_position") %>%
  filter(
    type %in% c("insertion", "deletion") |
      (type == "substitution" &
         ref_base == ref_nucleotide &
         cons_base %in% c("A", "C", "G", "T"))
  ) %>%
  mutate(condition = ifelse(group == "ePCR", "control", "sample"))

# for each position, count the total number of mutations (by library and by condition)
site_mut_df <- all_mutations_df %>%
  group_by(library_name, condition, ref_position) %>%
  summarise(total_mutations = n(), .groups = "drop")

### Env Plus - Merge 249 and 253
# Env coverage trim. LP249 and LP253 share this reference span.
env_refs <- library_maps[["LP249"]] %>%
  filter(
    library_position > edge_trim,
    library_position < (max(library_position) - edge_trim)
  )
env_start <- min(env_refs$ref_position)
env_end <- max(env_refs$ref_position)
env_sample <- site_mut_df %>% 
  filter(library_name %in% c("LP249","LP253"),condition == "sample") %>%
  complete(ref_position = seq(env_start, env_end), fill = list(total_mutations = 0)) %>%
  arrange(ref_position) %>% 
  mutate(library_name = "EnvPlus", condition = "sample")  %>%
  filter(ref_position >= env_start & ref_position <= env_end) %>%
  group_by(ref_position,library_name,condition) %>%
  summarise(total_mutations = sum(total_mutations)) 
# if total_mutations = 0, add 0.5 to avoid log(0) issues 
env_sample %>% 
  mutate(adjusted_total_mutations = ifelse(total_mutations == 0, 0.5, total_mutations)) -> env_sample
# write to .csv 
write.csv(env_sample, "../../figures/figure2/2B-env_plus_site_specific_muts.csv", row.names = FALSE)
plot(env_sample$total_mutations)

### Env Plus Control - Merge 249 and 253 for control data
env_control <- site_mut_df %>% 
  filter(library_name %in% c("LP249","LP253"),condition == "control") %>%
  complete(ref_position = seq(env_start, env_end), fill = list(total_mutations = 0)) %>%
  arrange(ref_position) %>% 
  mutate(library_name = "EnvPlus", condition = "control")  %>%
  filter(ref_position >= env_start & ref_position <= env_end) %>%
  group_by(ref_position,library_name,condition) %>%
  summarise(total_mutations = sum(total_mutations)) 
# if total_mutations = 0, add 0.5 to avoid log(0) issues 
env_control %>% 
  mutate(adjusted_total_mutations = ifelse(total_mutations == 0, 0.5, total_mutations)) -> env_control
# write to .csv 
write.csv(env_control, "../../figures/extended2/2B-env_plus_control_site_specific_muts.csv", row.names = FALSE)
plot(env_control$total_mutations)



## GagPol
# LP252 calls are trimmed with the mutation-table end (4810), which is past the
# reference-map end (4790). Calls through reference 4769 still enter the model
# because Env covers those positions. The panel includes that contribution.
lp252_max <- lib_max$max_library_position[lib_max$library_name == "LP252"]
gag_refs <- library_maps[["LP252"]] %>%
  filter(
    library_position > edge_trim,
    library_position < (lp252_max - edge_trim),
    ref_position %in% covered_refs$ref_position
  )
gagpol_start <- min(gag_refs$ref_position)
gagpol_end <- max(gag_refs$ref_position)
gagpol_sample <- site_mut_df %>% 
  filter(library_name %in% c("LP252"),condition == "sample") %>%
  complete(ref_position = seq(gagpol_start, gagpol_end), fill = list(total_mutations = 0)) %>%
  arrange(ref_position) %>%
  mutate(library_name = "GagPol", condition = "sample") %>% 
  filter(ref_position >= gagpol_start & ref_position <= gagpol_end)
gagpol_sample %>% 
  mutate(adjusted_total_mutations = ifelse(total_mutations == 0, 0.5, total_mutations)) -> gagpol_sample
write.csv(gagpol_sample, "../../figures/figure2/2A-gagpol_site_specific_muts.csv", row.names = FALSE)
plot(gagpol_sample$total_mutations)

### GagPol Control - LP252 for control data
gagpol_control <- site_mut_df %>% 
  filter(library_name == "LP252", condition == "control") %>%
  complete(ref_position = seq(gagpol_start, gagpol_end), fill = list(total_mutations = 0)) %>%
  arrange(ref_position) %>%
  mutate(library_name = "GagPol", condition = "control") %>% 
  filter(ref_position >= gagpol_start & ref_position <= gagpol_end) %>% 
  group_by(ref_position,library_name,condition) %>%
  summarise(total_mutations = sum(total_mutations)) 
gagpol_control %>% 
  mutate(adjusted_total_mutations = ifelse(total_mutations == 0, 0.5, total_mutations)) -> gagpol_control
write.csv(gagpol_control, "../../figures/extended2/2A-gagpol_control_site_specific_muts.csv", row.names = FALSE)
plot(gagpol_control$total_mutations)


# overlap - compare the overlap mutation accumulation in env_sample and gagpol_sample
# join by position - keep only the overlap 
joint_df <- env_sample %>% 
  inner_join(gagpol_sample, by = "ref_position", suffix = c("_env", "_gagpol")) %>% 
  dplyr::select(ref_position, total_mutations_env, total_mutations_gagpol)

# compute total RECs for each group 
total_env <- sum(joint_df$total_mutations_env)
total_gagpol <- sum(joint_df$total_mutations_gagpol)

total_UMI_ID_df %>% filter(library == "LP249" | library == "LP253", condition == "sample") %>% 
  summarise(total_UMI_ID = sum(total_UMI_ID)) -> total_env_UMI_ID_df
total_UMI_ID_df %>% filter(library == "LP252", condition == "sample") %>% 
  summarise(total_UMI_ID = sum(total_UMI_ID)) -> total_gagpol_UMI_ID_df

total_env / (total_env_UMI_ID_df[2][1]* env_end - env_start)
total_gagpol / (total_gagpol_UMI_ID_df[2][1] * gagpol_end - gagpol_start)






################################################################################
## Variance Analysis
######################################################################################
library(AER)

site_variance <- function(df) {
  site_summary <- df %>%
    group_by(ref_position) %>%
    summarise(
      total_mutations = sum(total_mutations),
      .groups = "drop"
    )
  
  poisson_model <- glm(
    total_mutations ~ 1,
    family = poisson(),
    data = site_summary
  )
  
  dispersiontest(poisson_model)
}
gagpol_sample_disp <- site_variance(gagpol_sample)
gagpol_control_disp <- site_variance(gagpol_control)
env_sample_disp <- site_variance(env_sample)
env_control_disp <- site_variance(env_control)

# create a summary table of the results
output_table <- tibble(
  library = c("GagPol", "GagPol", "EnvPlus", "EnvPlus"),
  condition = c("sample", "control", "sample", "control"),
  dispersion_statistic = c(gagpol_sample_disp$estimate, 
                           gagpol_control_disp$estimate, 
                           env_sample_disp$estimate, 
                           env_control_disp$estimate),
  p_value = c(gagpol_sample_disp$p.value, 
              gagpol_control_disp$p.value, 
              env_sample_disp$p.value, 
              env_control_disp$p.value)
)
output_table

# write to .csv
write_csv(output_table, "../../figures/extended2/2C-variance_analysis_results.csv")




################################################################################
### Extended Figure  2D  - Nucleotide Context Model Evaluations 
################################################################################
# data was aggregated using the mra data preparation function, which parsed 
# the five_prime and three_prime data 
library(mgcv)
library(stringr)
mutation_supply_data <- read_csv("../../data/models/mra/mutation_supply_data.csv")

head(mutation_supply_data,10)

# Number of context lengths to generate
n_context <- 5

for (i in seq_len(n_context)) {
  
  # Five-prime context:
  # Extract N nucleotides from the RIGHT side of five_prime
  mutation_supply_data[[paste0("p5p", i)]] <- str_sub(
    mutation_supply_data$five_prime,
    start = -i,
    end   = -1
  )
  
  # Three-prime context:
  # Extract N nucleotides from the LEFT side of three_prime
  mutation_supply_data[[paste0("p3p", i)]] <- str_sub(
    mutation_supply_data$three_prime,
    start = 1,
    end   = i
  )
}

# Replace empty strings with NA
mutation_supply_data <- mutation_supply_data %>%
  mutate(
    across(
      matches("^(5p|3p)[0-9]+$"),
      ~ na_if(.x, "")
    )
  )

head(mutation_supply_data)
mutation_supply_data$log_wt_coverage <- log10(mutation_supply_data$WT_coverage)


################################################################################
# Compare nucleotide-context models across all mutation types
################################################################################

library(dplyr)
library(tidyr)
library(purrr)
library(mgcv)


################################################################################
# Fit one mutation type and one context type
################################################################################

fit_context_model <- function(
    mutation_supply_data,
    mut_type,
    context_type
) {
  
  # Check that the requested context column exists
  if (!context_type %in% names(mutation_supply_data)) {
    stop(
      paste0(
        "Context column '",
        context_type,
        "' was not found."
      )
    )
  }
  
  
  ##########################################################################
  # Create long-format WT and control data
  ##########################################################################
  
  wt_data <- mutation_supply_data %>%
    filter(Mut_Type == mut_type) %>%
    transmute(
      ref_position,
      condition = "WT",
      mutation_count = WT_mutation_count,
      coverage = WT_coverage,
      context = .data[[context_type]]
    )
  
  
  control_data <- mutation_supply_data %>%
    filter(Mut_Type == mut_type) %>%
    transmute(
      ref_position,
      condition = "Control",
      mutation_count = Control_mutation_count,
      coverage = Control_coverage,
      context = .data[[context_type]]
    )
  
  
  mut_type_data <- bind_rows(
    wt_data,
    control_data
  ) %>%
    filter(
      !is.na(mutation_count),
      !is.na(coverage),
      coverage > 0,
      !is.na(context)
    ) %>%
    mutate(
      condition = factor(
        condition,
        levels = c("Control", "WT")
      ),
      context = droplevels(factor(context))
    )
  
  
  ##########################################################################
  # Check that the model can be fitted
  ##########################################################################
  
  if (nrow(mut_type_data) == 0) {
    stop("No usable observations remain after filtering.")
  }
  
  if (nlevels(droplevels(mut_type_data$condition)) < 2) {
    stop("Condition has fewer than two observed levels.")
  }
  
  if (nlevels(droplevels(mut_type_data$context)) < 2) {
    stop(
      paste0(
        "Context '",
        context_type,
        "' has fewer than two observed levels for ",
        mut_type,
        "."
      )
    )
  }
  
  
  ##########################################################################
  # Baseline model: no nucleotide context
  ##########################################################################
  
  base_model <- bam(
    mutation_count ~
      condition +
      offset(log(coverage)),
    family = nb(),
    method = "fREML",
    data = mut_type_data
  )
  
  
  ##########################################################################
  # Context model
  ##########################################################################
  
  context_model <- bam(
    mutation_count ~
      condition +
      context +
      offset(log(coverage)),
    family = nb(),
    method = "fREML",
    data = mut_type_data
  )
  
  
  ##########################################################################
  # Extract model statistics
  ##########################################################################
  
  base_summary <- summary(base_model)
  context_summary <- summary(context_model)
  
  model_comparison <- tibble(
    mutation_type = mut_type,
    context_type = context_type,
    observations = nrow(mut_type_data),
    genomic_positions = n_distinct(mut_type_data$ref_position),
    context_levels = nlevels(mut_type_data$context),
    
    base_adj_r_sq = base_summary$r.sq,
    context_adj_r_sq = context_summary$r.sq,
    delta_adj_r_sq =
      context_summary$r.sq - base_summary$r.sq,
    
    base_deviance_explained = base_summary$dev.expl,
    context_deviance_explained = context_summary$dev.expl,
    delta_deviance_explained =
      context_summary$dev.expl - base_summary$dev.expl,
    
    base_AIC = AIC(base_model),
    context_AIC = AIC(context_model),
    delta_AIC =
      AIC(context_model) - AIC(base_model),
    
    status = "success",
    error_message = NA_character_
  )
  
  
  return(
    list(
      data = mut_type_data,
      base_model = base_model,
      context_model = context_model,
      comparison = model_comparison
    )
  )
}


################################################################################
# Run all mutation type × context type combinations
################################################################################

run_all_context_models <- function(
    mutation_supply_data,
    mutation_types = NULL,
    context_types
) {
  
  ##########################################################################
  # Use every observed mutation type unless a subset is supplied
  ##########################################################################
  
  if (is.null(mutation_types)) {
    mutation_types <- mutation_supply_data %>%
      filter(!is.na(Mut_Type)) %>%
      distinct(Mut_Type) %>%
      arrange(Mut_Type) %>%
      pull(Mut_Type)
  }
  
  
  ##########################################################################
  # Check that all context columns exist
  ##########################################################################
  
  missing_contexts <- setdiff(
    context_types,
    names(mutation_supply_data)
  )
  
  if (length(missing_contexts) > 0) {
    stop(
      paste(
        "The following context columns were not found:",
        paste(missing_contexts, collapse = ", ")
      )
    )
  }
  
  
  ##########################################################################
  # Restrict all models to positions with every context available
  #
  # This ensures that context models are compared using the same positions.
  ##########################################################################
  
  comparison_data <- mutation_supply_data %>%
    filter(
      if_all(
        all_of(context_types),
        ~ !is.na(.x)
      )
    )
  
  
  ##########################################################################
  # Create all mutation type × context type combinations
  ##########################################################################
  
  model_grid <- crossing(
    mutation_type = mutation_types,
    context_type = context_types
  )
  
  
  ##########################################################################
  # Fit every model
  ##########################################################################
  
  model_results <- map2(
    model_grid$mutation_type,
    model_grid$context_type,
    function(current_mutation, current_context) {
      
      tryCatch(
        
        fit_context_model(
          mutation_supply_data = comparison_data,
          mut_type = current_mutation,
          context_type = current_context
        ),
        
        error = function(e) {
          
          # Return an informative row instead of stopping the loop
          list(
            data = NULL,
            base_model = NULL,
            context_model = NULL,
            
            comparison = tibble(
              mutation_type = current_mutation,
              context_type = current_context,
              observations = NA_integer_,
              genomic_positions = NA_integer_,
              context_levels = NA_integer_,
              
              base_adj_r_sq = NA_real_,
              context_adj_r_sq = NA_real_,
              delta_adj_r_sq = NA_real_,
              
              base_deviance_explained = NA_real_,
              context_deviance_explained = NA_real_,
              delta_deviance_explained = NA_real_,
              
              base_AIC = NA_real_,
              context_AIC = NA_real_,
              delta_AIC = NA_real_,
              
              status = "failed",
              error_message = conditionMessage(e)
            )
          )
        }
      )
    }
  )
  
  
  ##########################################################################
  # Give each result a descriptive name
  ##########################################################################
  
  names(model_results) <- paste(
    model_grid$mutation_type,
    model_grid$context_type,
    sep = "__"
  )
  
  
  ##########################################################################
  # Combine results into one table
  ##########################################################################
  
  comparison_table <- model_results %>%
    map("comparison") %>%
    bind_rows() %>%
    mutate(
      delta_adj_r_sq_percent =
        100 * delta_adj_r_sq,
      
      base_deviance_percent =
        100 * base_deviance_explained,
      
      context_deviance_percent =
        100 * context_deviance_explained,
      
      delta_deviance_percent =
        100 * delta_deviance_explained
    ) %>%
    arrange(
      mutation_type,
      desc(delta_deviance_explained)
    )
  
  
  return(
    list(
      comparison = comparison_table,
      models = model_results,
      model_grid = model_grid,
      analysis_data = comparison_data
    )
  )
}


context_types <- c(
  "p5p1",
  "p5p2",
  "p5p3",
  "p3p1",
  "p3p2",
  "p3p3"
)

all_context_results <- run_all_context_models(
  mutation_supply_data = mutation_supply_data,
  context_types = context_types
)

context_comparison <- all_context_results$comparison

context_comparison %>%
  filter(mutation_type == "G->A") %>%
  group_by(context_type) %>%
  summarise(base_deviance_explained = mean(base_deviance_explained),
            context_deviance_explained = mean(context_deviance_explained),
            delta_deviance_explained = mean(delta_deviance_explained))-> out_GA
out_GA$mut_type <- "G->A"

context_comparison %>%
  filter(mutation_type != "G->A") %>%
  group_by(context_type) %>%
  summarise(base_deviance_explained = mean(base_deviance_explained),
            context_deviance_explained = mean(context_deviance_explained),
            delta_deviance_explained = mean(delta_deviance_explained)) -> out_non_GA
out_non_GA$mut_type <- "nonG->A"

out <- rbind(out_GA, out_non_GA)

# write to .csv
write_csv(out, "../../figures/extended2/2D-nucleotide_context_analysis_results.csv")

