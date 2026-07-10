# Data card

## Dataset

The repository contains a supplied coursework CSV with 800 rows, ten binary screening scores,
demographic/background fields, and a binary `Class/ASD` label. The repository does not currently
contain a primary-source URL, collection protocol, consent statement, or license specific to the
dataset. That provenance gap must be resolved before reuse beyond coursework.

The raw file is preserved unchanged at `Data/Raw Data/Raw Data.csv`. Generated processed CSVs are
legacy notebook outputs and are not inputs to the production pipeline.

## Reviewed scope

- Source rows: 800
- Positive labels: 161 (20.1%)
- Adult production rows: 577
- Adult positive labels: 124 (21.5%)
- Excluded ages below 18: 223
- Exact duplicate rows: 0

The source sets `age_desc` to “18 and more” for every record while 223 numeric ages are below 18.
The reviewed model therefore declares an adult-only 18–100 scope and reports the excluded rows.

## Missingness and representation

Question-mark placeholders occur in 203 ethnicity values and 40 relation values. Several ethnicity,
country, and relation categories have very small samples. The available gender field has only `f`
and `m`, which does not represent gender diversity. These constraints make subgroup performance and
fairness claims inappropriate.

## Feature policy

The production contract excludes:

- `ID`, an identifier with no intended predictive meaning;
- `age_desc`, a constant that conflicts with numeric ages;
- `result`, whose continuous source semantics are undocumented and cannot be reproduced by the app;
- target/frequency-encoded derived columns, which were generated before splitting in legacy work.

The release uses 18 deployable fields: ten binary screening scores, age, and seven categorical
background fields. Missing-value imputation and one-hot encoding are learned inside training folds.

## Appropriate use

This data supports a small educational classification study. It does not support diagnosis, clinical
triage, prevalence estimates, or claims about performance in a real population. Any future work must
document the original source, questionnaire version, data-generation process, and legal basis for use.
