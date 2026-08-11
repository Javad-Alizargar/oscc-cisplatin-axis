# OSCC cisplatin-resistance-associated transcriptional axis

Reproducibility package for **“Cross-dataset validation of a cisplatin-resistance-associated transcriptional axis in oral squamous cell carcinoma.”**

This repository contains the frozen 150-up/150-down signature, analysis and orthogonal-validation scripts, compact result tables, a synthetic demonstration dataset, source-release metadata, and a citation-integrity ledger. It intentionally excludes controlled, bulky, and redistributability-limited source matrices.

## Main conclusion

The frozen axis showed modest advanced-stage enrichment across prespecified OSCC cohorts. Directional single-cell drift, overall survival, resistance-selective target dependency, and axis-linked drug response were not validated. Candidate compounds are hypotheses for laboratory cisplatin-combination testing—not effective treatments or validated sensitizers.

## Contents

- `code/analysis_v1.py`: manuscript analysis/figure workflow.
- `code/validation_v1.py`: DepMap/PRISM orthogonal validation workflow.
- `data/signature/`: frozen UP and DOWN gene lists.
- `data/demo/demo_v1.tsv`: synthetic input for format testing only.
- `data/source_v1.json`: source releases, URLs, and checksums.
- `results/`: compact manuscript result tables.
- `docs/citations_v1.tsv`: PubMed/OpenAlex integrity audit.

## Public data

The workflow uses GEO GSE117872, GSE168424, GSE103322, GSE172577, GSE215403, GSE41613, and GSE70138; oral-site TCGA-HNSC; PRISM 23Q2; and DepMap 24Q2. Obtain source files from their official repositories and verify versions/checksums before use.

## Reproduction boundary

The scripts preserve the project’s original directory contract and are provided for transparent inspection and rerunning with the documented source releases. The small demonstration file is synthetic and cannot reproduce biological findings. No patient-level raw expression matrix is redistributed here.

## Citation and archive

Please cite the paper after publication and the archived software release:

> Alizargar J. OSCC cisplatin-resistance-associated transcriptional axis reproducibility package. Zenodo. [DOI to be added after release].

## License

Code is released under the MIT License. Source datasets remain governed by their original terms.
