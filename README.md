# HIV-1 Mutational Bias Shapes the Supply of Drug Resistance Mutations

**Adam D. Nitido<sup>1</sup>, Nartşav Ömür<sup>1</sup>, and Alejandro B. Balazs<sup>1,\*</sup>**

<sup>1</sup>Ragon Institute of MGH, MIT, and Harvard, Cambridge, MA, 02139, USA  
<sup>\*</sup>Lead contact; correspondence: [abalazs@mgh.harvard.edu](mailto:abalazs@mgh.harvard.edu)

This repository contains the data, analysis code, and figure-generation scripts accompanying the HIV-1 Mutation Rate Atlas (MRA) paper.

---

## Summary

Durable HIV-1 therapeutics require understanding not only which mutations confer resistance, but also how readily those mutations arise. This work reports an **HIV-1 Mutation Rate Atlas (MRA)** built from empirical measurements of single-cycle HIV-1 mutation accumulation in the absence of selection, using error-corrected long-read sequencing (ERR-Seq). The MRA estimates the mutation rate at every genomic position to each possible nucleotide, enabling evaluation of amino acid mutability across the genome. Using the MRA, mutation-rate biases predict HIV-1 drug resistance variant frequencies across patients, showing that mutational biases shape clinically relevant evolutionary outcomes.

---

## What is the Mutation Rate Atlas (MRA)?

The MRA is a genome-wide resource of estimated per-site, per-mutation-type **biological** mutation rates for HIV-1<sub>NL4-3</sub>, inferred with a hierarchical Bayesian model that separates biological mutation accumulation from technical error measured in matched library controls.

With the MRA you can:

- Query nucleotide mutation rates across nearly the full HIV-1<sub>NL4-3</sub> coding genome
- Translate those rates into **amino acid mutability** (including multi-nucleotide paths)
- Evaluate how mutational supply relates to variant frequencies observed in people living with HIV (PLWH), including drug resistance mutations (DRMs)

**Interactive WebApp:** [https://balazslab.partners.org/HIV-MRA](https://balazslab.partners.org/HIV-MRA/)

The WebApp supports position/amino-acid queries (HIV-1<sub>HXB2</sub> coordinates), curated DRM and bNAb-escape tables, browser-side custom-sequence MRA generation, and precomputed group M LANL compendium analyses. Submitted sequences are processed in the browser and are not sent to a third-party server.

---

## Repository contents

| Path | Contents |
|------|----------|
| `configs/` | Pipeline configuration (`pipeline.yaml`) listing paper samples and processing parameters |
| `data/pipeline/` | ERR-Seq references, maps, and aggregated mutation/consensus outputs used for analyses |
| `data/mra/` | MRA model inputs, priors/run settings, production fit summaries, and amino-acid mutability tables |
| `data/supply/` | Downstream supply analyses: Stanford Drug Resistance Database extracts, protease fitness data, and variant-supply tables |
| `figures/` | Tabular source data underlying main Figures 1–5 and Extended Figures 1–5 |
| `scripts/pipeline/` | Helper scripts called by the Snakemake workflow |
| `scripts/analysis/` | Coding-effect annotation and analysis data preparation |
| `scripts/model/mra/` | Hierarchical Bayesian MRA model (PyMC), runbook, and conda environment specs |
| `scripts/figures/` | R scripts that regenerate figure source tables from analysis outputs |
| `snakefiles/Pipeline.Snakefile` | ERR-Seq analysis Snakemake workflow |

Related software (external):

- **BELT** (barcode/UMI mapping, consensus calling, mutation analysis): [https://github.com/adamn102/belt](https://github.com/adamn102/belt) (MIT License)

---

## Data availability (as stated in the paper)

| Resource | Location |
|----------|----------|
| Raw sequencing data | NCBI SRA BioProject **PRJNA1560037** (see [NCBI BioProject](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1560037); release status may lag manuscript submission) |
| Processed data, model, and analysis code | This repository |
| Interactive MRA WebApp | [https://balazslab.partners.org/HIV-MRA](https://balazslab.partners.org/HIV-MRA/) |
| Genetic constructs / unique biological materials | Available from the lead contact upon request (no restriction) |

External datasets used in analyses (not generated here):

- Stanford HIV Drug Resistance Database genotype–treatment amino-acid prevalence tables
- HIV-1<sub>NL4-3</sub> protease fitness measurements from Boucher et al., 2019
- LANL group M compendium sequences (WebApp analyses)

---

## Reproducing analyses (high level)

### 1. ERR-Seq processing pipeline

Requires [Snakemake](https://snakemake.readthedocs.io/) (≥ v9.13.4 as used in the paper), [BELT](https://github.com/adamn102/belt), and the paper’s tool stack (fastplong, minimap2, samtools, bcftools, UMI-tools, Dorado for basecalling upstream of this repo).

```bash
# Dry-run
snakemake -s snakefiles/Pipeline.Snakefile \
  --configfile configs/pipeline.yaml -n

# Full run (after placing sample FASTQs under the paths listed in configs/pipeline.yaml)
snakemake -s snakefiles/Pipeline.Snakefile \
  --configfile configs/pipeline.yaml --cores 24
```

Paper analyses start from aggregated outputs under `data/pipeline/10-mutation_analysis_ERRSeq-PAPER-SAMPLES/`. Raw FASTQs are expected from SRA rather than being stored in this repository.

### 2. Mutation Rate Atlas model

See the detailed runbook:

- [`scripts/model/mra/mra_RUNBOOK.md`](scripts/model/mra/mra_RUNBOOK.md)
- Conda environment specs: [`scripts/model/mra/env/`](scripts/model/mra/env/)

Production outputs used in the paper live under `data/mra/production/`, including:

- `FINAL_mutation_supply_dataset.csv` — per-site nucleotide mutation rates (MRA core table)
- `comprehensive_analysis_mra_product/aa_level_supply_table.csv` — amino acid mutability

### 3. Figure source tables

R scripts in `scripts/figures/` (`Figure1.R`–`Figure5.R`) regenerate the CSV tables under `figures/`. Precomputed outputs for the published figures are already present in `figures/figure1`–`figure5` and `figures/extended1`–`extended5`.

> **Note:** Some figure scripts still reference older path aliases (`data/models/mra/`, `data/mutation_supply/`). The current layout uses `data/mra/` and `data/supply/`. Prefer the checked-in `figures/` tables unless you update those paths locally.

---

## Software versions (paper Methods)

Key versions reported in the manuscript Methods:

| Tool | Version |
|------|---------|
| Snakemake | 9.13.4 |
| fastplong | 0.4.1 |
| minimap2 | 2.30 |
| samtools | 1.22.1 |
| bcftools | 1.22 |
| UMI-tools | 1.1.6 |
| PyMC | 5.26.1 |
| MASS (`glm.nb`, R) | 7.3-65 |
| ONT Dorado (sup model) | `dna_r10.4.1_e8.2_400bps_sup@v5.2.0` |

---

## Citation

If you use the MRA, ERR-Seq pipeline, or analysis code, please cite the manuscript:

> Nitido, A. D., Ömür, N., & Balazs, A. B. *A Mutation Rate Atlas of HIV-1 Reveals How Mutational Bias Shapes the Supply of Drug Resistance Mutations.* (manuscript).

A formal citation with journal, year, and DOI will be added upon publication.

---

## License

CC-BY-NC 4.0 International license. The companion BELT toolkit is separately released under the MIT License.

---

## Contact

**Alejandro B. Balazs** — [abalazs@mgh.harvard.edu](mailto:abalazs@mgh.harvard.edu)  
Ragon Institute of MGH, MIT, and Harvard
