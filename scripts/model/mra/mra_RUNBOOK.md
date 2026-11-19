# mra Mutation-Supply Model — Runbook

`mra` is the paper's **primary** mutation-supply model: the Bayesian hierarchical
model that estimates the per-site, per-mutation-type **biological mutation rate**
(`bio_rate`) across the HIV-1 (NL4-3) genome while separating it from the
technical error measured in matched error-control (ePCR) libraries. Its
amino-acid-level output is the Mutation Rate Atlas (MRA).

This runbook documents the final production model and the exact, reproducible
commands that produce every output used in the paper. It is written to stand
alone as an extended-data methods supplement.

## Model at a glance

> | | |
> |---|---|
> | **Purpose** | Per-site, per-mutation-type biological mutation rate (`bio_rate`) across the HIV-1 NL4-3 genome, separated from technical error. |
> | **Identifier** | `mra_joint_control_wt_nb_ga2_sub1_ga3_sum0_bio_splines` |
> | **Likelihood** | Negative binomial; experimental (WT) and control (ePCR) arms fit jointly, with REC coverage as exposure. |
> | **Rate decomposition** | `WT = error + biological`; `control = error`. Released quantity: `bio_rate`. |
> | **Overdispersion** | Separate per mutation type × arm (28 total). |
> | **Sequence context** | G→A: 3′ dinucleotide → 13 levels (sparse `CC/CG/TC/TT` pooled to `OTHER`). Other substitutions: 3′ single base (cells with <20 WT counts pooled to type-only). Indels: none. |
> | **Position effect** | 6 cubic B-spline (df 30), **sum-to-zero**: G→A AA, G→A GA, G→A SHARED, non-G→A shared, insertion, deletion. On the biological arm only. |
> | **WT-only (error = 0)** | `C→A`, `G→T` (oxidative-damage control artefact), `deletion` (sparse control) — biological rates are upper bounds. |
> | **Priors** | Explicit, weakly informative (not transferred): bio log-rates `Normal(log 1e-5, 2.0)`, error `Normal(log 1e-6, 1.5)`, HalfNormal spreads/amplitudes, Exponential dispersion. Full set in `mra_priors.csv`. |
> | **Sampling** | PyMC NUTS, 4 chains × 2000 draws (6000 tune), `target_accept = 0.98`, seed 42. |
> | **Data** | Libraries LP249/252/253 (NL4-3); RECs ≤1000 mutations; 20-bp edge trim; 8,213 positions × 5 = 41,065 observations. |
> | **Configuration** | `data/models/mra/mra_run_production.csv` (settings) + `mra_priors.csv` (priors). |
> | **Final outputs** | `data/models/mra/production/` (fit) and `.../comprehensive_analysis_mra_product/` (amino-acid atlas). |

Final model identifier (recorded in every fit's `model_metadata.json`):

```
mra_joint_control_wt_nb_ga2_sub1_ga3_sum0_bio_splines
```

The `sum0` token marks the key feature of the final model: every biological
position spline is **sum-to-zero constrained** (see
[Model specification](#model-specification)).

All paths below are relative to the repository root `MutRatePaper/`. Scripts
resolve this automatically via `PROJECT_ROOT = Path(__file__).resolve().parents[3]`,
so no machine-specific paths are hard-coded. Commands are shown run from
`scripts/model/mra/`.

---

## 1. Reproducibility model: everything is driven by two CSV files

The entire fit is defined by two plain-text configuration files under
`data/models/mra/`. Nothing about the model is hard-coded in the scripts; to
reproduce or alter the fit you edit these files, not the Python.

| File | Role |
|------|------|
| `mra_run_production.csv` | All **non-prior** run settings (sampling, model structure, spline regularisers, data filters). Used by both the input builder and the fit. |
| `mra_run_test.csv` | Same schema, reduced sampling for a fast smoke test. |
| `mra_priors.csv` | The **complete prior specification**: one row per sampled parameter, with family, location, and scale. |

Both loaders are tiny and auditable:

- `mra_run.py` → `RunConfig`: reads `mra_run_*.csv` (columns `parameter,value,type`),
  type-casts each value, and refuses duplicate keys.
- `mra_priors.py` → `FilePriors`: reads `mra_priors.csv` (columns
  `parameter,family,location,scale`), and serves `Normal`, `HalfNormal`, and
  `Exponential` priors to the model by exact parameter name. A mismatch between
  the family the model requests and the family in the file is a hard error, so
  the fitted model can never silently diverge from the documented priors.

A fixed `random_seed` (42) makes the fit deterministic given the same inputs and
configuration.

### 1.1 Production run settings (`mra_run_production.csv`)

These are the exact values used for the paper fit.

| parameter | value | group | meaning |
|-----------|-------|-------|---------|
| `random_seed` | `42` | sampling | Fixed seed for a reproducible fit. |
| `chains` | `4` | sampling | NUTS chains. |
| `cores` | `4` | sampling | Parallel sampling workers. |
| `draws` | `2000` | sampling | Retained draws per chain (8,000 total). |
| `tune` | `6000` | sampling | Warm-up iterations per chain. |
| `target_accept` | `0.98` | sampling | NUTS target acceptance probability. |
| `context_mode` | `mixed_ga2_else1` | structure | G→A uses the 3′ dinucleotide; other substitutions use the 3′ single base. |
| `context_n` | `3` | structure | Length of the flanking-context string read from the input table. |
| `min_context_count` | `0` | structure | No rarity drop at this stage (G→A pooling handled separately). |
| `min_sub_context_wt_count` | `20` | structure | Non-G→A type×base cells with `< 20` WT counts are pooled to the type-level effect. |
| `drop_indels` | `0` | structure | Keep insertions and deletions. |
| `include_position_bio` | `1` | structure | Fit the six biological position splines. |
| `include_position_err` | `0` | structure | No positional spline on the error arm. |
| `spline_df` | `30` | structure | Cubic B-spline basis (31 functions; 30 free under sum-to-zero). |
| `ga_pool_2mers` | `CC,CG,TC,TT` | structure | Sparse G→A 3′ dinucleotides pooled into `OTHER`. |
| `wt_only_mut_types` | `C->A,G->T,deletion` | structure | Types fit WT-only (no control arm). |
| `spline_sigma_floor` | `0.1` | spline | Additive floor on each spline amplitude (`σ = floor + HalfNormal`). |
| `spline_sum_to_zero` | `1` | spline | Hard sum-to-zero on each bio P-spline. |
| `spline_smoothness_scale` | `0.25` | spline | Roughness penalty for non-G→A substitution and indel splines. |
| `spline_smoothness_scale_ga_aa` | `0.08` | spline | Weaker penalty for the G→A **AA** spline. |
| `spline_smoothness_scale_ga_non_aa` | `0.18` | spline | Stronger penalty for the G→A **GA** and **SHARED** splines. |
| `max_muts_per_rec` | `1000` | data | Keep consensus molecules (RECs) with ≤ this many mutations. |
| `edge_trim` | `20` | data | Trim this many bases from each end of each library amplicon. |
| `wt_groups` | `WT,NEG,Sparsomycin,pUC19` | data | Sample-name groups pooled into the mutation-supply (experimental) class. |
| `control_groups` | `ePCR` | data | Sample-name groups pooled into the error-control class. |
| `context_k` | `5` | data | Flanking bases stored in `five_prime`/`three_prime` when the input table is built. |

Each row also carries an `env_var` and `rationale` column; the `env_var` names
are legacy overrides and are **not** consulted when a `--run` file is supplied.

### 1.2 Prior specification (`mra_priors.csv`)

The final model uses explicit, weakly-informative priors anchored on round
order-of-magnitude rates — **not** empirically-centred or transfer priors
(`use_transfer_priors = false`). Centres are deliberately generic so the data,
not the prior, determines each rate. The file contains one row per sampled
scalar; the distinct priors are:

| Prior group | n params | Family | Location | Scale | Applies to |
|-------------|---------:|--------|----------|-------|------------|
| `bio_log_rate` | 25 | Normal | `log(1e-5) = -11.513` | `2.0` | Global bio intercept, the 11 per-type substitution effects, and all 13 G→A context intercepts (same prior for every biological log-rate). |
| `error_intercept` | 1 | Normal | `log(1e-6) = -13.816` | `1.5` | Technical error-rate intercept. |
| `bio_type_spread` | 1 | HalfNormal | — | `1.0` | SD of the indel type effects around the shared intercept. |
| `error_type_spread` | 1 | HalfNormal | — | `1.0` | SD of per-type error effects. |
| `context_spread` | 1 | HalfNormal | — | `0.5` | SD of 3′ single-base context deviations (half the type spread, so context modulates but does not redefine a type). |
| `spline_amplitude` | 6 | HalfNormal | — | `0.75` | Amplitude (σ) of each of the six bio splines. |
| `spline_shape` | 180 | Normal | `0` | `1` | Standardised spline coefficients (6 splines × 30 free). |
| `unit_normal` | 72 | Normal | `0` | `1` | Non-centred auxiliaries for type/context effects (`effect = z·σ`). |
| `dispersion` | 28 | Exponential | — | `1.0` | Negative-binomial overdispersion, one per mutation type × arm (WT/control). |

Choosing and sanity-checking these priors is itself scripted and reproducible:

- `mra_prior_predictive.py` — simulates data from the prior **without fitting**
  and checks that the prior predictive brackets the observed data (contains it,
  is wider than it by orders of magnitude, and is not absurd). This is how the
  prior scales were set before running PyMC.
- `mra_prior_audit.py` — after a fit, reconstructs each parameter's prior and
  compares it to the posterior (prior/posterior contraction and z-shift), to
  confirm the posterior is data-driven rather than prior-driven.

---

## 2. Model specification

For every genome position `i` and mutation type `t(i)`, the observed control and
experimental (WT) mutation counts are modelled as negative-binomial draws with
the number of consensus molecules (RECs) as exposure:

```
c_i ~ NB( E_i^c · λ_i^err ,            α_{t(i)}^c )      # control arm
e_i ~ NB( E_i^e · (λ_i^err + λ_i^bio), α_{t(i)}^e )      # experimental arm
```

- `E_i^c`, `E_i^e` — control / experimental REC coverage at the site (exposure).
- `λ_i^err` — technical error rate; `λ_i^bio` — biological mutation rate (the
  released quantity, `bio_rate`).
- `α^c`, `α^e` — separate overdispersion parameters per mutation type and arm
  (28 total; `dispersion` prior).

**Rate structure (log scale).**

- `log λ^bio = type effect + 3′-context effect + biological position spline`
- `log λ^err = intercept + per-type effect` (no context, no positional spline)

**Sequence context (`mixed_ga2_else1`).**

- **G→A**: 3′ **dinucleotide** (the two bases immediately 3′ of the mutated G,
  i.e. the `NN` in a `G·NN` motif). The four sparsest dinucleotides
  (`CC, CG, TC, TT`) are pooled into a single `OTHER` level, giving **13** G→A
  context levels, each with its own intercept.
- **All other substitutions**: 3′ **single base** (`A/C/G/T`). A type×base cell
  with `< 20` WT observations is pooled back to a type-only effect.
- **Insertions and deletions**: no sequence context.

**Six biological position splines (cubic B-spline, df = 30, sum-to-zero).**
Positions are standardised before basis construction. Each spline samples
`n_basis − 1 = 30` free coefficients and sets the last to `−Σ(free)`, so the
spline is mean-zero across the genome and the intercept owns the overall level
("amplitude owns only shape"). Smoothness is imposed by a roughness penalty on
second differences, with context-specific weights.

| Spline | Applies to | Smoothness weight |
|--------|------------|-------------------|
| G→A AA | G→A, `AA` 2-mer | `0.08` |
| G→A GA | G→A, `GA` 2-mer | `0.18` |
| G→A SHARED | all other G→A 2-mers (incl. the pooled `OTHER`) | `0.18` |
| non-G→A shared | all non-G→A substitutions | `0.25` |
| insertion | insertions | `0.25` |
| deletion | deletions | `0.25` |

**WT-only correction.** For `C→A`, `G→T`, and `deletion` the control arm is
omitted and `λ^err = 0`:

- `C→A` / `G→T`: oxidative-damage (8-oxo-G) artefacts inflate these in the
  directly-sequenced plasmid control but are repaired after transfection.
- `deletion`: control counts are too sparse to be informative.

Consequently the biological estimates for these three types are an **upper
bound** on the true rate, and the inflated control counts cannot contaminate the
error term.

---

## 3. Data flow

```
data/pipeline/10-mutation_analysis_.../all_mutations.csv            ┐
data/pipeline/10-mutation_analysis_.../consensus_sequence_data.csv  ├─▶ mra_build_model_input.py --run
data/pipeline/00-reference-map/{LP###.csv, *.yaml}                  ┘   (applies per-REC filter + edge trim)
                                                                         │
                                       data/models/mra/mutation_supply_data.csv
                                       data/models/mra/mutation_supply_data.build_metadata.json
                                                                         │
            mra_MutationSupplyPymc_production_joint_WT_control.py --priors --run --outdir
                                                                         ▼
                      <OUTDIR>/  (idata.nc, ppc.nc, model_metadata.json, posterior_summary.csv, …)
                                                                         │
         postfit ─▶ prior-audit ─▶ diagnostics ─▶ build-table ─▶ comprehensive (AA/MRA) ─▶ sliding-window ─▶ variant-supply
```

The paper's final fit lives in `data/models/mra/production/`, which is also the
scripts' default output directory, so `--outdir` / `MUT_SUPPLY_OUTDIR` can be
omitted to reproduce into it (they are shown explicitly below for clarity).

---

## 4. Prerequisites

- Conda environment `pymc_env` with PyMC (paper used **PyMC 5.26.1**), ArviZ,
  PyTensor, NumPy, pandas, patsy, PyYAML, matplotlib. The exact environment is
  exported under `env/` — recreate it with
  `conda env create -n pymc_env -f env/pymc_env.from-history.yml` (portable) or
  `env/pymc_env.no-builds.yml` / `env/pymc_env.full.yml` for stricter pinning;
  see `env/README.md`.
- Pipeline step-10 outputs present under
  `data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/`
  (`all_mutations.csv`, `consensus_sequence_data.csv`).
- Reference map under `data/pipeline/00-reference-map/` (per-library
  `LP###.csv` and `library_to_biological_reference.yaml`); reference strain
  `NL43`; libraries `LP249`, `LP252`, `LP253`.
- Gene annotation `data/mutation_supply/annotation/NL43/NL43_protein_codons.csv`
  (9 HIV-1 proteins) for the amino-acid atlas (Step 8).

```bash
conda activate pymc_env
cd scripts/model/mra
```

---

## 5. Step-by-step reproducible pipeline

Every step below is a single command. Steps 0, 2, and 3 are required to
reproduce the fit; the rest reproduce the published analyses and tables.

### Step 0 — Build the model input (applies the per-REC filter)

The only place per-REC quality filtering happens. Reads the `--run` file for
`max_muts_per_rec`, `edge_trim`, `context_k`, `wt_groups`, and `control_groups`.

```bash
python3 mra_build_model_input.py --run ../../../data/models/mra/mra_run_production.csv
```

Outputs:
- `data/models/mra/mutation_supply_data.csv` — one row per (position × mutation
  type): `ref_position, Mut_Type, WT_mutation_count, Control_mutation_count,
  WT_coverage, Control_coverage, ref_nucleotide, five_prime, three_prime`.
- `data/models/mra/mutation_supply_data.build_metadata.json` — filter stats
  (RECs considered/kept/dropped, covered positions, output rows).

RECs with `Total_Mutations > max_muts_per_rec` are dropped from **both** the
mutation numerator and the coverage denominator so per-site rates stay
internally consistent. (At the paper threshold of 1000, 0 of 60,677 RECs were
dropped.)

### Step 1 — (Optional) Prior predictive check

Confirms the prior brackets reality before any fitting.

```bash
python3 mra_prior_predictive.py                 # default candidate spec
python3 mra_prior_predictive.py --dump-spec      # print the spec as JSON
```

### Step 2 — Test fit (fast smoke test)

Verifies the model compiles and samples using `mra_run_test.csv`
(1 chain, 50 draws/tune).

```bash
python3 mra_MutationSupplyPymc_production_joint_WT_control.py \
  --priors ../../../data/models/mra/mra_priors.csv \
  --run    ../../../data/models/mra/mra_run_test.csv \
  --outdir ../../../data/models/mra/production_test
```

### Step 3 — Production fit (the paper model)

```bash
python3 mra_MutationSupplyPymc_production_joint_WT_control.py \
  --priors ../../../data/models/mra/mra_priors.csv \
  --run    ../../../data/models/mra/mra_run_production.csv \
  --outdir ../../../data/models/mra/production
```

`--priors` and `--run` are **required**; `--data` and `--outdir` default to
`data/models/mra/mutation_supply_data.csv` and `data/models/mra/production`
respectively (or set `MUT_SUPPLY_DATA_PATH` / `MUT_SUPPLY_OUTDIR`).

Key outputs in `<OUTDIR>`:
- `idata.nc` — full posterior (InferenceData).
- `ppc.nc` — posterior predictive samples.
- `model_metadata.json` — complete, self-describing record of the fit: resolved
  `run_settings`, the `prior_unique` summary, context levels, spline rules,
  base rates, and observation counts.
- `posterior_summary.csv`, `posterior_diagnostics.csv` — parameter summaries,
  R-hat, ESS.

For the rest of the pipeline, point the environment at this directory once:

```bash
export MUT_SUPPLY_OUTDIR="$(cd ../../../data/models/mra/production && pwd)"
```

### Step 4 — Post-fit analysis (tables, PPC, position profiles, figures)

```bash
python3 mra_MutationSupplyPymc_postfit_analysis.py
```

Writes to `$MUT_SUPPLY_OUTDIR`:
- `FINAL_mutation_supply_dataset.csv` — per-site posterior `bio_rate`
  (median + 95% interval), error rate, and inputs; the atlas's nucleotide-level
  source table.
- `postfit_ppc_wt_by_mutation_type.csv`, `postfit_ppc_control_corrected_types.csv`,
  `postfit_ppc_by_mutation_type.csv` — observed vs predicted counts per type.
- `postfit_r2_by_mutation_type.csv`, `postfit_mutation_spectrum.csv`,
  `postfit_context_summary.csv`.
- `postfit_position_profile_{GtoA_AA,GtoA_GA,GtoA_SHARED,shared_sub,insertion,deletion}.csv`
  — fitted spline profiles.
- `postfit_model_checklist.txt`, `loo_wt.txt`, `waic_wt.txt`.
- `figures/` — `ppc_*`, `mutation_spectrum_*`, `context_bio_supply_GtoA`,
  `position_bio_rate_*`.

### Step 5 — Prior audit (prior vs posterior sensitivity)

```bash
python3 mra_prior_audit.py "$MUT_SUPPLY_OUTDIR" -o "$MUT_SUPPLY_OUTDIR/prior_audit.csv"
```

Reports per-parameter contraction `1 − (σ_post/σ_prior)²` and
`z = (μ_post − μ_prior)/σ_prior`. High contraction / large |z| ⇒ data-driven.

### Step 6 — Calibration diagnostics (optional)

```bash
python3 mra-PostFitDiagnostics.py --model-output-dir "$MUT_SUPPLY_OUTDIR"
```

### Step 7 — Build the bio_rate table (optional)

```bash
python3 mra-BuildMutationSupplyTable.py \
  --input-data ../../../data/models/mra/mutation_supply_data.csv \
  --supply-csv "$MUT_SUPPLY_OUTDIR/mutation_supply_site_mutation_table.csv" \
  --out-csv    "$MUT_SUPPLY_OUTDIR/mutation_supply_bio_rate_table.csv"
```

### Step 8 — Amino-acid supply (the Mutation Rate Atlas)

Converts the per-site posterior into amino-acid-coded supply. Reads
`FINAL_mutation_supply_dataset.csv`, enumerates every mutant-codon path per
codon, translates via the genetic code, and sums path rates (product of
per-site `bio_rate_median` along a path) to `(protein, AA position, alt_aa)`.

```bash
python3 mra_comprehensive_mutation_supply_analysis.py
```

Outputs to `$MUT_SUPPLY_OUTDIR/comprehensive_analysis_mra_product/`:
- `site_level_supply_table.csv`, `aa_mutation_path_table.csv`,
  `aa_level_supply_table.csv` (with `mu_aa`, `mu_aa_lo95`, `mu_aa_hi95`),
  `aa_mutation_entropy.csv`.
- `module1`–`module5` tables (variance decomposition, within/between-AA, etc.),
  `robustness_bootstrap_site_summary.csv`, `analysis_run_metadata.csv`,
  `aa_level_supply_ci_methodology.txt`, and figures.

**Gene-aware annotation is required** for the published atlas: the enumeration
uses `data/mutation_supply/annotation/NL43/NL43_protein_codons.csv` (9 proteins).
Confirm `codon_annotation_mode = gene_annotated` and `n_proteins_annotated = 9`
in `analysis_run_metadata.csv`; otherwise it silently falls back to a single
`GENOME` frame (`frame_inferred`), which does **not** match the paper. Override
with `--protein-codons-file` if needed.

### Step 9 — Sliding-window observed vs modeled rate (Figure 3)

```bash
python3 mra_sliding_window_observed_vs_bio.py
```

Outputs to `$MUT_SUPPLY_OUTDIR/sliding_window_analysis/`:
`sliding_window_observed_vs_bio_rate_{100bp_step10,150bp_step10,250bp_step25}.csv`,
`sliding_window_correlation_stats_*.csv` (Pearson r of observed vs modeled),
`sliding_window_analysis_metadata.json`, and linear/log10 figures. Observed rate
= `Σwt/Σwt_cov − Σctrl/Σctrl_cov`; modeled rate = `bio_rate_median −
error_rate_median` (net biological supply).

### Step 10 — Variant supply (mature-protein extraction, optional)

```bash
python3 mra_build_variant_supply.py
```

Converts the polyprotein-keyed `aa_level_supply_table.csv` into a mature-protein
table (Pro, RT, IN, p24) renumbered into each protein's own coordinates. Mature
boundaries come from the GenBank `misc_feature` records in
`data/pipeline/00-bed_files/NL43.gb`. Defaults: `--aa-supply`
`.../comprehensive_analysis_mra_product/aa_level_supply_table.csv`, `--genbank`
`.../00-bed_files/NL43.gb`, `--out`
`data/mutation_supply/mra_variant_supply.csv` (`--no-verify` to skip the
cross-check against the legacy workbook).

---

## 6. Script inventory

```
scripts/model/mra/
  # configuration loaders
  mra_run.py                                            RunConfig: load mra_run_*.csv (sampling/structure/data settings)
  mra_priors.py                                         FilePriors: load mra_priors.csv (Normal/HalfNormal/Exponential)

  # build input
  mra_build_model_input.py                              pipeline → mutation_supply_data.csv (per-REC filter, edge trim)

  # prior tooling
  mra_prior_predictive.py                               prior predictive simulation + acceptance checks (pre-fit)
  mra_prior_audit.py                                    prior vs posterior contraction / z-shift (post-fit)
  mra_prior_transfer.py                                 legacy transfer-prior loader (NOT used; final fit uses explicit priors)

  # fit
  mra_MutationSupplyPymc_production_joint_WT_control.py the production model (requires --priors, --run)

  # post-fit + analyses
  mra_MutationSupplyPymc_postfit_analysis.py            FINAL dataset, PPC, R², position profiles, figures
  mra-PostFitDiagnostics.py                             calibration diagnostics
  mra-BuildMutationSupplyTable.py                       per-site bio_rate table with CIs
  mra_comprehensive_mutation_supply_analysis.py         AA-level supply / Mutation Rate Atlas (wrapper)
  comprehensive_mutation_supply_analysis.py             shared module for the above
  mra_sliding_window_observed_vs_bio.py                 sliding-window observed vs modeled (wrapper)
  sliding_window_observed_vs_bio.py                     shared module for the above
  mra_build_variant_supply.py                           mature-protein variant-supply table

  mra_RUNBOOK.md / mra_RUNBOOK.html                     this document
```

Configuration files (under `data/models/mra/`): `mra_run_production.csv`,
`mra_run_test.csv`, `mra_priors.csv`.

---

## 7. Extended-data mapping

| Extended-data item | File(s) |
|--------------------|---------|
| Model fit (convergence + posterior) | `<OUTDIR>/posterior_summary.csv`, `posterior_diagnostics.csv`, `model_metadata.json`, `idata.nc` |
| Mutation-supply input table | `data/models/mra/mutation_supply_data.csv` |
| Prior specification | `data/models/mra/mra_priors.csv` |
| Run settings | `data/models/mra/mra_run_production.csv` |
| Per-site atlas (nucleotide) | `<OUTDIR>/FINAL_mutation_supply_dataset.csv` |
| Amino-acid atlas (MRA) | `<OUTDIR>/comprehensive_analysis_mra_product/aa_level_supply_table.csv` |
| Sliding-window validation (Fig. 3) | `<OUTDIR>/sliding_window_analysis/` |

---

## 8. Reproducibility checklist

1. Activate `pymc_env` (PyMC 5.26.1) from the repo root.
2. `mra_build_model_input.py --run mra_run_production.csv` → `mutation_supply_data.csv`.
3. (Optional) `mra_prior_predictive.py` to confirm the prior.
4. `mra_MutationSupplyPymc_production_joint_WT_control.py --priors mra_priors.csv --run mra_run_production.csv --outdir production`.
5. Confirm convergence: no parameters with R-hat > 1.01 or ESS < 400
   (`posterior_diagnostics.csv` / `postfit_model_checklist.txt`).
6. `mra_prior_audit.py production` to confirm the posterior is data-driven.
7. Run post-fit (Step 4) and the analyses (Steps 6–10).
8. The fixed seed (42) plus the two CSVs fully determine the fit; archive the
   CSVs and `model_metadata.json` alongside the outputs.
