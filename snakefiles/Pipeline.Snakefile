# =============================================================================
# BELT Pipeline — Snakefile
# =============================================================================
#
# Pipeline steps (in order):
#   1. Quality filter (fastp-long)     → filter_sequences
#   2. Map reads (minimap2)            → map_reads
#   3. Sort & index BAM                → samtools_sort, samtools_index
#   4. UMI map                         → umi_map
#   5. UMI group                       → umi_group
#   6. Consensus call                 → consensus_call
#   7. Compile consensus              → compile_consensus
#   8. Mutation calling               → mutation_calling
#   9. Mutation analysis              → mutation_analysis
#  10. Coding mutational effects    → coding_analysis (scripts/analysis/)
#  11. Prepare analysis data        → prepare_rec_data (scripts/analysis/)
#
# Before steps 7-8, the trimmed reference is edge-trimmed (edge_trim.bp bp removed
# from each end) via rule edge_trim_reference to exclude PCR-induced edge effects
# from both the mutation-rate numerator and denominator. See configs/pipeline.yaml.
#
# Large intermediates (BAMs, UMI CSVs) are marked temp() and removed automatically
# when keep_intermediates is false (see configs/pipeline.yaml).
# Compiled consensus FASTAs are kept until step 10 completes, then removed.
#
# Usage:
#   snakemake -s snakefiles/Pipeline.Snakefile --configfile configs/pipeline.yaml
# =============================================================================

import os
import glob
import json

configfile: "configs/pipeline.yaml"


# -----------------------------------------------------------------------------
# Configuration — paths and parameters from config
# -----------------------------------------------------------------------------

# Samples and references
SAMPLES = list(config["samples"].keys())
REFS = config.get("references", {})
REFERENCE_DIR = REFS["reference_dir"]
CONSENSUS_REFERENCE_DIR = REFS.get("consensus_reference_dir", REFERENCE_DIR)
TRIMMED_REFERENCE_DIR = REFS.get("trimmed_reference_dir", REFERENCE_DIR)
REFERENCE_MAP_DIR = REFS.get("reference_map_dir", REFERENCE_DIR)
BIOLOGICAL_REFERENCE_DIR = REFS.get("biological_reference_dir", REFERENCE_DIR)

# Edge trimming — remove EDGE_TRIM_BP bp from each end of the trimmed reference
# before consensus compilation and mutation calling (excludes PCR edge effects).
EDGE_TRIM_CFG = config.get("edge_trim", {})
EDGE_TRIM_BP = int(EDGE_TRIM_CFG.get("bp", 0))
EDGE_TRIMMED_REFERENCE_DIR = EDGE_TRIM_CFG.get("output_dir", "data/pipeline/00-edge_trimmed_reference")

# Step 1 — FastQ filter (fastp-long)
FILTERED_DIR = config["step1_fastq_filter"]["output_dir"]
FASTP_Q = str(config["step1_fastq_filter"]["qualified_quality_phred"])
FASTP_U = str(config["step1_fastq_filter"]["unqualified_percent_limit"])
FASTP_DISABLE_ADAPTER = bool(config["step1_fastq_filter"]["disable_adapter_trimming"])
FASTP_LOG_HTML_DIR = config["step1_fastq_filter"]["logs_html_dir"]
FASTP_LOG_JSON_DIR = config["step1_fastq_filter"]["logs_json_dir"]

# Step 2/3 — Mapping, sort, index
MAPPED_DIR = config["step2_mapping"]["output_dir"]
MAP_THREADS = int(config["step2_mapping"]["threads"])
MAP_LOG_DIR = config["step2_mapping"]["log_dir"]
SORTED_DIR = config["step3_sort_index"]["sorted_dir"]

# Step 5 — UMI map (motif/min-length from step5_umi_map; step 4 umi_extract removed)
UMI_MOTIF = config["step5_umi_map"]["motif"]
UMI_MIN_READ_LENGTH = int(config["step5_umi_map"]["min_read_length"])
UMI_MAPS_DIR = config["step5_umi_map"]["output_dir"]
UMI_MAP_MAX_DISTANCE = int(config["step5_umi_map"]["max_distance"])
UMI_MAP_LOG_DIR = config["step5_umi_map"]["log_dir"]

# Step 6 — UMI group
UMI_GROUPING = config["step6_umi_group"]["grouping"]
UMI_THRESHOLD = int(config["step6_umi_group"]["threshold"])
UMI2_MAJORITY_THRESHOLD = float(config["step6_umi_group"].get("umi2_majority_threshold", 0.5))
UMI_GROUP_DIR = config["step6_umi_group"]["output_dir"]
UMI_GROUP_LOG_DIR = config["step6_umi_group"].get("log_dir", "logs/6-umi_group")
# Pre-filter corrected UMI CSVs before grouping: drop UMIs with total count < this value.
# Default to the umi_group threshold so only UMIs that could form valid groups are kept.
UMI_GROUP_MIN_COUNT = int(config["step6_umi_group"].get("min_umi_count_for_grouping", UMI_THRESHOLD))

# Step 7 — Consensus
CONSENSUS_BASE_DIR = config["step7_consensus"]["output_base_dir"]
CONSENSUS_THREADS = int(config["step7_consensus"]["threads"])
CONSENSUS_LOG_DIR = config["step7_consensus"]["log_dir"]
CONSENSUS_METHOD = config["step7_consensus"].get("method", "bcftools")
# Minimum fraction of reads supporting a variant to include in consensus (e.g. 0.8 = 80%). None = bcftools default (majority).
CONSENSUS_MIN_ALLELE_FRACTION = config["step7_consensus"].get("min_allele_fraction")

# Step 8 — Compile consensus
COMPILED_CONSENSUS_DIR = config["step8_compile_consensus"]["output_dir"]

# Step 9 — Mutation calling
MUT_CALL_DIR = config["step9_mutation_calling"]["output_dir"]
MUT_CALL_THREADS = int(config["step9_mutation_calling"]["threads"])
MUT_CALL_LOG_DIR = config["step9_mutation_calling"]["log_dir"]

# Step 10 — Mutation analysis
MUT_ANALYSIS_INPUT_DIR = config["step10_mutation_analysis"]["input_dir"]
MUT_ANALYSIS_OUTPUT_DIR = config["step10_mutation_analysis"]["output_dir"]
MUT_ANALYSIS_LOG_FILE = config["step10_mutation_analysis"]["log_file"]

# Step 11 — Coding-region mutational effects (scripts/analysis/01_coding_analysis.py)
STEP11_CODING_CFG = config.get("step11_coding_analysis", {})
CODING_SCRIPT = STEP11_CODING_CFG.get("script", "scripts/analysis/01_coding_analysis.py")
CODING_OUTPUT_DIR = STEP11_CODING_CFG.get("output_dir", "data/pipeline/11-analysis")
CODING_GENBANK_FILE = STEP11_CODING_CFG.get(
    "genbank_file", "data/pipeline/00-bed_files/NL43.gb"
)
CODING_REFERENCE_FA = STEP11_CODING_CFG.get(
    "reference_fa", "data/pipeline/00-consensus_reference/NL43.fa"
)
CODING_LOG_FILE = STEP11_CODING_CFG.get("log_file", "logs/analysis/coding_analysis.log")
CODING_MUTATIONS_CSV = f"{CODING_OUTPUT_DIR}/coding_mutations.csv"

# Step 12 — Prepare analysis data (scripts/analysis/00_prepare_data.py)
STEP12_CFG = config.get("step12_prepare_analysis", config.get("step11_prepare_analysis", {}))
PREPARE_OUTPUT_DIR = STEP12_CFG.get("output_dir", "data/pipeline/11-analysis")
SAMPLE_GROUPS_CONFIG = STEP12_CFG.get("sample_groups_config", "configs/analysis.yaml")
PREPARE_INCLUDE_GROUPS = STEP12_CFG.get("include_groups", ["samples", "controls"])
PREPARE_LOG_FILE = STEP12_CFG.get("log_file", "logs/analysis/prepare_rec_data.log")
PREPARE_SCRIPT = STEP12_CFG.get("script", "scripts/analysis/00_prepare_data.py")
PREPARED_CONSENSUS_CSV = f"{PREPARE_OUTPUT_DIR}/consensus_sequence_data.csv"

# Keep large intermediate files (BAMs, filtered FASTQs, consensus dirs) for faster reruns.
# When false, outputs use Snakemake temp() and explicit cleanup steps remove siblings.
KEEP_INTERMEDIATES = bool(config.get("keep_intermediates", True))

def pipeline_temp(path):
    """Apply Snakemake temp() unless keep_intermediates is enabled."""
    return path if KEEP_INTERMEDIATES else temp(path)

def pipeline_temp_dir(path):
    """Apply Snakemake temp(directory()) unless keep_intermediates is enabled."""
    d = directory(path)
    return d if KEEP_INTERMEDIATES else temp(d)

# Wildcard constraints (sample: no path slashes; library_id unconstrained, used via expand)
wildcard_constraints:
    sample="[^/]+",
    ref_base="[^/]+"

###########################################################
# Rule: all — target outputs for the main pipeline
###########################################################

def list_mutation_calling_outputs():
    return [f"{MUT_CALL_DIR}/{sample}_mutations.json" for sample in SAMPLES]

def list_mutation_analysis_outputs():
    return [
        f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_summary.csv",
        f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_analysis_report.json",
        f"{MUT_ANALYSIS_OUTPUT_DIR}/all_mutations.csv",
        f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_position_stats.csv",
        f"{MUT_ANALYSIS_OUTPUT_DIR}/consensus_sequence_data.csv"
    ]

rule all:
    input:
        CODING_MUTATIONS_CSV,
    message: "BELT Pipeline complete (coding mutational effects only)."

###########################################################
# Helper functions — reference paths and inputs
###########################################################

def get_reference(wildcards):
    """Get reference file for a sample"""
    base_name = wildcards.sample.split('-')[0]
    return f"{REFERENCE_DIR}/{base_name}.fa"

def get_consensus_reference(wildcards):
    """Get consensus reference file for a sample"""
    # Extract the base name (e.g., B145 from B145-XTMMF9-12)
    base_name = wildcards.sample.split('-')[0]
    return f"{CONSENSUS_REFERENCE_DIR}/{base_name}.fa"

def get_trimmed_reference(wildcards):
    """Get trimmed reference file for a sample"""
    base_name = wildcards.sample.split('-')[0]
    return f"{TRIMMED_REFERENCE_DIR}/{base_name}.fa"

def get_edge_trimmed_reference(wildcards):
    """Get edge-trimmed reference (EDGE_TRIM_BP removed from each end) for a sample.

    Used by compile_consensus and mutation_calling so both the mutation-rate
    numerator and denominator exclude the reference edges. When EDGE_TRIM_BP == 0
    this file is a copy of the trimmed reference.
    """
    base_name = wildcards.sample.split('-')[0]
    return f"{EDGE_TRIMMED_REFERENCE_DIR}/{base_name}.fa"

def get_reference_map(wildcards):
    """Get reference position mapping file for a sample"""
    base_name = wildcards.sample.split('-')[0]
    return f"{REFERENCE_MAP_DIR}/{base_name}.csv"

def get_nucleotide_context_file(wildcards):
    """Get nucleotide context CSV file for a reference"""
    base_name = wildcards.reference_base
    return f"{CONSENSUS_REFERENCE_DIR}/{base_name}_nucleotide_context.csv"

def list_nucleotide_context_files():
    """List nucleotide context CSV files for every reference in the biological reference dir, so belt mutation-summary has matching context for each reference it loads."""
    ref_dir = BIOLOGICAL_REFERENCE_DIR
    if not os.path.isdir(ref_dir):
        return []
    fastas = glob.glob(os.path.join(ref_dir, "*.fa"))
    ref_bases = sorted({os.path.splitext(os.path.basename(p))[0] for p in fastas})
    return [f"{CONSENSUS_REFERENCE_DIR}/{ref}_nucleotide_context.csv" for ref in ref_bases]

def get_barcode(wildcards):
    """Get barcode motif for a sample"""
    return UMI_MOTIF

def get_min_length(wildcards):
    """Get minimum read length for a sample"""
    return UMI_MIN_READ_LENGTH

def get_sample_fastq(wildcards):
    """Get FASTQ path from config for a sample"""
    return config["samples"][wildcards.sample]

###########################################################
# Step 1 — filter_sequences (quality filter raw FASTQ)
###########################################################

rule filter_sequences:
    input:
        R = get_sample_fastq
    params:
        l = get_min_length,
        q = FASTP_Q,
        u = FASTP_U,
        disable_flag = ('--disable_adapter_trimming' if FASTP_DISABLE_ADAPTER else '')
    log:
        h = f"{FASTP_LOG_HTML_DIR}/{{sample}}.html",
        j = f"{FASTP_LOG_JSON_DIR}/{{sample}}.json"
    output:
        R = pipeline_temp(f"{FILTERED_DIR}/{{sample}}.fastq")
    message: "Quality filtering {wildcards.sample}"
    shell:
        """
        fastplong --in {input.R} --out {output.R} \
        --length_required {params.l} \
        --qualified_quality_phred {params.q} \
        --unqualified_percent_limit {params.u} \
        {params.disable_flag} \
        --html {log.h} --json {log.j}
        """

###########################################################
# Step 2 — map_reads (minimap2)
###########################################################

rule map_reads:
    input:
        ref = get_reference,
        R = f"{FILTERED_DIR}/{{sample}}.fastq"
    log:
        L = f"{MAP_LOG_DIR}/{{sample}}.log"
    params:
        threads = MAP_THREADS
    output:
        O = pipeline_temp(f"{MAPPED_DIR}/{{sample}}.bam")
    message: "Mapping reads for {wildcards.sample}"
    shell:
        """
        minimap2 -t {params.threads} -a {input.ref} {input.R} > {output.O} 2> {log.L}
        """

###########################################################
# Step 3 — samtools_sort, samtools_index
###########################################################

rule samtools_sort:
    input:
        f"{MAPPED_DIR}/{{sample}}.bam"
    output:
        pipeline_temp(f"{SORTED_DIR}/{{sample}}.bam")
    message: "Sorting BAM for {wildcards.sample}"
    shell:
        "samtools sort {input} -o {output}"

###########################################################
# Step 3 (cont.) — samtools_index
###########################################################

rule samtools_index:
    input:
        f"{SORTED_DIR}/{{sample}}.bam"
    output:
        pipeline_temp(f"{SORTED_DIR}/{{sample}}.bam.bai")
    message: "Indexing BAM for {wildcards.sample}"
    shell:
        "samtools index {input}"

###########################################################
# Step 5 — umi_map (map/correct UMIs; step 4 umi_extract removed)
###########################################################

rule umi_map:
    input:
        reference = get_reference,
        alignment = f"{SORTED_DIR}/{{sample}}.bam",
        bai = f"{SORTED_DIR}/{{sample}}.bam.bai"
    params:
        min_read_length = get_min_length,
        motif = get_barcode,
        max_distance = UMI_MAP_MAX_DISTANCE,
        log_dir = UMI_MAP_LOG_DIR
    output:
        outfile = pipeline_temp(f"{UMI_MAPS_DIR}/{{sample}}_Paired_UMIs_read_names.csv")
    message: "Mapping UMIs for {wildcards.sample}"
    shell:
        """
        belt umi-map \
            --reference "{input.reference}" \
            --alignment "{input.alignment}" \
            --min_read_length {params.min_read_length} \
            --motif "{params.motif}" \
            --whitelist cluster \
            --max_distance {params.max_distance} \
            --output_dir "{UMI_MAPS_DIR}" \
            --log_dir "{params.log_dir}" \
            --write_csv
        """

###########################################################
# Step 6 — umi_group (group reads by corrected UMIs)
###########################################################

rule umi_group_prefilter:
    input:
        correctedUMIs = f"{UMI_MAPS_DIR}/{{sample}}_Paired_UMIs_read_names.csv"
    params:
        min_count = UMI_GROUP_MIN_COUNT,
        umi_column = UMI_GROUPING,
        script = "scripts/pipeline/03_filter_umis_for_grouping.py"
    output:
        filteredUMIs = pipeline_temp(f"{UMI_GROUP_DIR}/{{sample}}_Paired_UMIs_for_grouping.csv"),
        log_json = f"{UMI_GROUP_LOG_DIR}/{{sample}}_umi_group_prefilter.log.json"
    message: "Prefiltering UMIs for grouping (dropping low-count UMIs) for {wildcards.sample}"
    shell:
        """
        mkdir -p {UMI_GROUP_DIR} "{UMI_GROUP_LOG_DIR}"
        python3 {params.script} \
            --input {input.correctedUMIs} \
            --output {output.filteredUMIs} \
            --umi-column {params.umi_column} \
            --min-count {params.min_count} \
            --log-json {output.log_json}
        """

rule umi_group:
    input:
        alignment = f"{SORTED_DIR}/{{sample}}.bam",
        correctedUMIs = f"{UMI_GROUP_DIR}/{{sample}}_Paired_UMIs_for_grouping.csv"
    params:
        grouping = UMI_GROUPING,
        threshold = UMI_THRESHOLD,
        umi2_majority_threshold = UMI2_MAJORITY_THRESHOLD
    output:
        outfile = pipeline_temp(f"{UMI_GROUP_DIR}/{{sample}}.grouped.bam"),
        log_json = f"{UMI_GROUP_LOG_DIR}/{{sample}}_umi_group.log.json"
    message: "Grouping UMIs for {wildcards.sample}"
    shell:
        """
        belt umi-group \
        --alignment "{input.alignment}" \
        --corrected_umis "{input.correctedUMIs}" \
        --log_dir "{UMI_GROUP_LOG_DIR}" \
        --grouping {params.grouping} \
        --threshold {params.threshold} \
        --umi2_majority_threshold {params.umi2_majority_threshold} \
        --output_dir {UMI_GROUP_DIR}
        # Create empty BAM file if no output was generated (no UMIs pass filters)
        if [ ! -f {output.outfile} ]; then
            samtools view -H {input.alignment} | samtools view -Sb - > {output.outfile}
            samtools index {output.outfile}
        fi
        """

###########################################################
# Step 7 — consensus_call
###########################################################

rule consensus_call:
    input:
        grouped_bam = f"{UMI_GROUP_DIR}/{{sample}}.grouped.bam",
        ref = get_reference
    output:
        fasta_dir = pipeline_temp_dir(f"{CONSENSUS_BASE_DIR}/{{sample}}/{{sample}}/fasta"),
        bam_dir = pipeline_temp_dir(f"{CONSENSUS_BASE_DIR}/{{sample}}/{{sample}}/bam")
    threads: CONSENSUS_THREADS
    log:
        f"{CONSENSUS_LOG_DIR}/{{sample}}.log"
    message: "Calling consensus for {wildcards.sample}"
    params:
        consensus_method = CONSENSUS_METHOD,
        ref_path = lambda wildcards, input: str(input.ref),
        min_allele_fraction_arg = f'--min-allele-fraction {CONSENSUS_MIN_ALLELE_FRACTION}' if CONSENSUS_MIN_ALLELE_FRACTION is not None else '',
        cleanup_intermediates = (not KEEP_INTERMEDIATES),
    shell:
        """
        # Check if BAM file exists and has reads
        if [ ! -f {input.grouped_bam} ]; then
            echo "Error: Grouped BAM file not found: {input.grouped_bam}" >> {log}
            exit 1
        fi
        read_count=$(samtools view -c {input.grouped_bam} 2>/dev/null || echo "0")
        if [ "$read_count" -eq "0" ]; then
            echo "Warning: Grouped BAM file is empty (no reads). Creating empty output directories." >> {log}
            mkdir -p {output.fasta_dir} {output.bam_dir}
            touch {output.fasta_dir}/.empty
            if [ "{params.cleanup_intermediates}" = "True" ]; then
                rm -f "{input.grouped_bam}.bai" "{UMI_MAPS_DIR}/{wildcards.sample}_umi_tagged.bam" || true
            fi
            exit 0
        fi
        # Copy reference to /tmp if path contains spaces (bcftools path issue)
        ORIG_REF="{params.ref_path}"
        FINAL_REF="$ORIG_REF"
        TEMP_REF_FILE=""
        if [[ "$ORIG_REF" = /* ]] && [[ "$ORIG_REF" == *" "* ]]; then
            TEMP_REF_BASE=$(basename "$ORIG_REF")
            TEMP_REF_FILE="/tmp/belt_ref_$$_$TEMP_REF_BASE"
            if cp "$ORIG_REF" "$TEMP_REF_FILE" 2>&1; then
                if [[ -f "$TEMP_REF_FILE" ]]; then
                    [[ -f "$ORIG_REF.fai" ]] && cp "$ORIG_REF.fai" "$TEMP_REF_FILE.fai" 2>&1
                    FINAL_REF="$TEMP_REF_FILE"
                    echo "DEBUG: Successfully copied reference to $FINAL_REF" >> {log}
                else
                    echo "ERROR: Copy succeeded but file not found: $TEMP_REF_FILE" >> {log}
                    exit 1
                fi
            else
                echo "ERROR: Failed to copy reference from $ORIG_REF to $TEMP_REF_FILE" >> {log}
                exit 1
            fi
        fi
        echo "DEBUG: Final reference path: $FINAL_REF" >> {log}
        # Run belt consensus
        belt consensus \
        --grouped-bam "{input.grouped_bam}" \
        --reference "$FINAL_REF" \
        --output-dir "{CONSENSUS_BASE_DIR}/{wildcards.sample}" \
        --sample {wildcards.sample} \
        --threads {threads} \
        --consensus-method {params.consensus_method} \
        {params.min_allele_fraction_arg} \
        >> {log} 2>&1
        EXIT_CODE=$?
        [[ -n "$TEMP_REF_FILE" ]] && [[ -f "$TEMP_REF_FILE" ]] && rm -f "$TEMP_REF_FILE" "$TEMP_REF_FILE.fai"
        rm -rf "{CONSENSUS_BASE_DIR}/{wildcards.sample}/{wildcards.sample}/tmp" || true
        if [ "{params.cleanup_intermediates}" = "True" ]; then
            rm -f "{input.grouped_bam}.bai" "{UMI_MAPS_DIR}/{wildcards.sample}_umi_tagged.bam" || true
        fi
        exit $EXIT_CODE
        """

###########################################################
# Step 8 — compile_consensus (concat, map to trimmed ref, trim)
###########################################################

rule edge_trim_reference:
    input:
        trimmed_ref = f"{TRIMMED_REFERENCE_DIR}/{{ref_base}}.fa"
    output:
        edge_trimmed_ref = f"{EDGE_TRIMMED_REFERENCE_DIR}/{{ref_base}}.fa"
    params:
        trim = EDGE_TRIM_BP
    message: "Edge-trimming {params.trim} bp from each end of reference {wildcards.ref_base}"
    shell:
        """
        mkdir -p {EDGE_TRIMMED_REFERENCE_DIR}
        python3 scripts/pipeline/edge_trim_reference.py \
            --input {input.trimmed_ref} \
            --output {output.edge_trimmed_ref} \
            --trim {params.trim}
        """

rule compile_consensus:
    input:
        fasta_dir = f"{CONSENSUS_BASE_DIR}/{{sample}}/{{sample}}/fasta",
        trimmed_ref = get_edge_trimmed_reference
    output:
        raw_consensus = pipeline_temp(f"{COMPILED_CONSENSUS_DIR}/{{sample}}-raw-consensus.fa"),
        trimmed_consensus = f"{COMPILED_CONSENSUS_DIR}/{{sample}}-compiled-consensus.fa",
        alignment = pipeline_temp(f"{COMPILED_CONSENSUS_DIR}/{{sample}}-to-trimmed-ref.bam")
    params:
        cleanup_intermediates = (not KEEP_INTERMEDIATES),
    message: "Compiling and trimming consensus for {wildcards.sample}"
    shell:
        """
        # Step 1: Concatenate all consensus sequences (avoid ARG_MAX)
        rm -f {output.raw_consensus}

        # Handle samples with no consensus sequences (empty grouped BAM upstream).
        shopt -s nullglob
        fasta_files=({input.fasta_dir}/*.fasta)
        shopt -u nullglob
        if [ ${{#fasta_files[@]}} -eq 0 ]; then
            touch {output.raw_consensus}
            touch {output.trimmed_consensus}
            : > {output.alignment}
            if [ "{params.cleanup_intermediates}" = "True" ]; then
                rm -rf "{CONSENSUS_BASE_DIR}/{wildcards.sample}" || true
            fi
            exit 0
        fi

        for f in "${{fasta_files[@]}}"; do
            cat "$f" >> {output.raw_consensus}
        done
        
        # Step 2: Map to trimmed reference
        minimap2 -ax map-ont {input.trimmed_ref} {output.raw_consensus} | \
            samtools view -bS - | samtools sort -o {output.alignment}
        
        # Step 3: Extract aligned regions, removing soft-clipped bases
        # Parse CIGAR to trim 5' and 3' overhangs using Python script
        samtools view -F 4 {output.alignment} | \
            python3 scripts/pipeline/trim_consensus_to_reference.py > {output.trimmed_consensus}
        
        # If no reads aligned, create empty file
        if [ ! -s {output.trimmed_consensus} ]; then
            touch {output.trimmed_consensus}
        fi

        if [ "{params.cleanup_intermediates}" = "True" ]; then
            rm -rf "{CONSENSUS_BASE_DIR}/{wildcards.sample}" || true
        fi
        """

###########################################################
# Step 9 — mutation_calling
###########################################################

rule mutation_calling:
    input:
        consensus_file = f"{COMPILED_CONSENSUS_DIR}/{{sample}}-compiled-consensus.fa",
        reference_file = get_edge_trimmed_reference,
        reference_map = get_reference_map
    output:
        json_file = f"{MUT_CALL_DIR}/{{sample}}_mutations.json",
        csv_file = f"{MUT_CALL_DIR}/samples/{{sample}}/{{sample}}_per_read.csv"
    params:
        consensus_file = f"{COMPILED_CONSENSUS_DIR}/{{sample}}-compiled-consensus.fa",
        reference_file = get_edge_trimmed_reference,
        edge_trim = EDGE_TRIM_BP,
        sample_name = "{sample}",
        output_dir = MUT_CALL_DIR,
        temp_output_dir = f"{MUT_CALL_DIR}/temp"
    threads: MUT_CALL_THREADS
    log:
        f"{MUT_CALL_LOG_DIR}/{{sample}}.log"
    message: "Calling mutations and remapping positions for {wildcards.sample}"
    shell:
        """
        mkdir -p {MUT_CALL_DIR}
        mkdir -p {MUT_CALL_LOG_DIR}
        mkdir -p {params.temp_output_dir}
        mkdir -p "{MUT_CALL_DIR}/samples/{params.sample_name}"

        # Handle samples with no compiled consensus sequences (empty grouped BAM upstream).
        if [ ! -s "{params.consensus_file}" ]; then
            echo "Warning: Compiled consensus is empty. Writing empty mutation outputs." >> {log}
            python3 - <<'PY'
import json
from pathlib import Path

sample = "{params.sample_name}"
temp_dir = Path("{params.temp_output_dir}")
sample_dir = temp_dir / "samples" / sample
sample_dir.mkdir(parents=True, exist_ok=True)

json_path = temp_dir / (sample + "_mutations.json")
csv_path = sample_dir / (sample + "_per_read.csv")

payload = {{
    "sample_name": sample,
    "consensus_count": 0,
    "total_length": 0,
    "total_mutations": 0,
    "average_mutation_rate": 0.0,
    "mutation_types": {{"substitution": 0, "insertion": 0, "deletion": 0}},
    "mutations_by_position": [],
}}
json_path.write_text(json.dumps(payload, indent=2))
csv_path.write_text("Sample_ID,Consensus_ID,Sequence_Length,Mutation_Count,Mutation_Rate\\n")
PY
        else
            # Step 1: Call mutations using trimmed reference (positions in library coordinates)
            # Convert absolute reference path to relative path to avoid space issues
            REF_FILE="{params.reference_file}"
            if [[ "$REF_FILE" = /* ]] && [[ "$REF_FILE" == *" "* ]]; then
                REF_FILE=$(python3 -c "import os; print(os.path.relpath('{params.reference_file}', os.getcwd()))")
            fi
            belt mutation-call \
                --consensus-file "{params.consensus_file}" \
                --reference-file "$REF_FILE" \
                --sample-name {params.sample_name} \
                --output-dir "{params.temp_output_dir}" \
                >> {log} 2>&1
        fi
        
        # Step 2: Remap positions from library reference to biological reference (e.g., LP249 -> NL43)
        # Extract library name from sample (e.g., LP249 from LP249-NL43-WT-1-P1)
        LIBRARY_NAME=$(echo "{params.sample_name}" | cut -d'-' -f1)
        
        python3 scripts/pipeline/remap_mutation_positions.py \
            --map-file "{input.reference_map}" \
            --json-file "{params.temp_output_dir}/{params.sample_name}_mutations.json" \
            --csv-file "{params.temp_output_dir}/samples/{params.sample_name}/{params.sample_name}_per_read.csv" \
            --output-json "{output.json_file}" \
            --output-csv "{output.csv_file}" \
            --library-name "$LIBRARY_NAME" \
            --library-mapping-file "{REFERENCE_MAP_DIR}/library_to_biological_reference.yaml" \
            --offset {params.edge_trim} \
            >> {log} 2>&1
        
        # Clean up temporary files
        rm -rf "{params.temp_output_dir}/{params.sample_name}_mutations.json"
        rm -rf "{params.temp_output_dir}/samples/{params.sample_name}"
        """

###########################################################
# nucleotide_context — generate context CSV for references
###########################################################

def get_reference_for_context(wildcards):
    """Get reference FASTA for nucleotide context generation (biological reference so context length matches mutation-summary validation)."""
    return f"{BIOLOGICAL_REFERENCE_DIR}/{wildcards.reference_base}.fa"

rule generate_nucleotide_context:
    input:
        reference = get_reference_for_context
    output:
        context_csv = f"{CONSENSUS_REFERENCE_DIR}/{{reference_base}}_nucleotide_context.csv"
    log:
        f"{CONSENSUS_REFERENCE_DIR}/{{reference_base}}_nucleotide_context.log"
    message: "Generating nucleotide context for {wildcards.reference_base}"
    shell:
        """
        mkdir -p {CONSENSUS_REFERENCE_DIR}
        python3 scripts/pipeline/generate_nucleotide_context.py {input.reference} {output.context_csv} >> {log} 2>&1
        """

###########################################################
# Step 10 — mutation_analysis (summary from per-sample mutation files)
###########################################################

rule mutation_analysis:
    input:
        mutation_files = list_mutation_calling_outputs(),
        nucleotide_context_files = list_nucleotide_context_files()
    output:
        summary_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_summary.csv",
        report_json = f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_analysis_report.json",
        all_mutations_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/all_mutations.csv",
        position_stats_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_position_stats.csv",
        consensus_data_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/consensus_sequence_data.csv"
    params:
        output_dir = MUT_ANALYSIS_OUTPUT_DIR,
        input_dir = MUT_ANALYSIS_INPUT_DIR,
        context_dir = CONSENSUS_REFERENCE_DIR,
        reference_dir = BIOLOGICAL_REFERENCE_DIR
    log:
        MUT_ANALYSIS_LOG_FILE
    message: "Creating mutation analysis summary"
    shell:
        """
        mkdir -p {params.output_dir}
        mkdir -p $(dirname {log})
        belt mutation-summary \
            --output-dir {params.output_dir} \
            --input-dir {params.input_dir} \
            --context-dir {params.context_dir} \
            --reference-dir {params.reference_dir} \
            >> {log} 2>&1
        """

###########################################################
# Step 11 — coding_analysis (scripts/analysis/01_coding_analysis.py)
# Classify mutations as synonymous / missense / nonsense in coding regions.
###########################################################

rule coding_analysis:
    input:
        mutations_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/all_mutations.csv",
        genbank_file = CODING_GENBANK_FILE,
        reference_fa = CODING_REFERENCE_FA,
    output:
        coding_mutations_csv = CODING_MUTATIONS_CSV,
    params:
        script = CODING_SCRIPT,
        output_dir = CODING_OUTPUT_DIR,
        genbank_file = CODING_GENBANK_FILE,
        reference_fa = CODING_REFERENCE_FA,
        mutations_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/all_mutations.csv",
    log:
        CODING_LOG_FILE
    message: "Classifying coding-region mutational effects"
    run:
        import importlib.util
        import os
        import contextlib

        os.makedirs(os.path.dirname(log[0]), exist_ok=True)

        spec = importlib.util.spec_from_file_location("coding_analysis", params.script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.GENBANK_FILE = params.genbank_file
        mod.MUTATIONS_CSV = params.mutations_csv
        mod.REFERENCE_FA = params.reference_fa
        mod.OUTPUT_DIR = params.output_dir

        with open(log[0], "w") as lf:
            with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                mod.main()

###########################################################
# Step 12 — prepare_rec_data (scripts/analysis/00_prepare_data.py)
# Enriched consensus_sequence_data.csv for manual figure scripts (scripts/figures/).
###########################################################

rule prepare_rec_data:
    input:
        context_csv = f"{CONSENSUS_REFERENCE_DIR}/NL43_nucleotide_context.csv",
        counts_csv = f"{MUT_ANALYSIS_OUTPUT_DIR}/consensus_sequence_data.csv",
        report_json = f"{MUT_ANALYSIS_OUTPUT_DIR}/mutation_analysis_report.json",
        sample_groups = SAMPLE_GROUPS_CONFIG,
        compiled_dir = COMPILED_CONSENSUS_DIR,
    output:
        consensus_data = PREPARED_CONSENSUS_CSV,
    params:
        output_dir = PREPARE_OUTPUT_DIR,
        include_groups = PREPARE_INCLUDE_GROUPS,
        script = PREPARE_SCRIPT,
        compiled_dir = COMPILED_CONSENSUS_DIR,
    log:
        PREPARE_LOG_FILE
    message: "Preparing analysis consensus_sequence_data.csv"
    run:
        import importlib.util
        import os
        import contextlib

        os.makedirs(params.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log[0]), exist_ok=True)

        spec = importlib.util.spec_from_file_location(
            "prepare_data",
            params.script,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.CONSENSUS_SEQUENCE_DATA_CSV = input.context_csv
        mod.MUTATION_COUNT_DATA_CSV = input.counts_csv
        mod.MUTATION_SPECIFIC_DATA_CSV = input.report_json
        mod.SAMPLE_GROUPS_CONFIG_YAML = input.sample_groups
        mod.COMPILED_CONSENSUS_DIR = input.compiled_dir
        mod.INCLUDE_GROUPS = list(params.include_groups)
        mod.OUTPUT_DIR = params.output_dir

        with open(log[0], "w") as lf:
            with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                mod.main()

        # Compiled consensus FASTAs are only needed through this step.
        if not KEEP_INTERMEDIATES:
            for fa in os.listdir(params.compiled_dir):
                if fa.endswith("-compiled-consensus.fa"):
                    os.remove(os.path.join(params.compiled_dir, fa))
