# Repository audit

This review covers every tracked notebook, the raw and derived data files, the serialized model,
the Streamlit application, dependency declarations, and project documentation. The legacy files
are retained for learning history; the reviewed production path is the `asd_screening` package.

## High-impact findings

| Finding | Why it matters | Resolution |
|---|---|---|
| Target and frequency encodings were calculated before the train/test split. | Target statistics from future validation rows can leak into training. | Production uses one-hot encoding fitted inside each inner CV fold. |
| Hyperparameter search selected the overall winner using repeated access to the test F1. | The reported test result became part of model selection and was no longer an untouched estimate. | Model-family selection now uses nested CV on the training partition; the holdout is evaluated once. |
| Several train/test splits omitted `stratify=y`. | With only 20% positives, class proportions can shift and make comparisons noisy. | Every split and fold is stratified and reproducible. |
| `Best_Model.ipynb` loaded one encoding, loaded a potentially different artifact, and then refit it on the evaluation split. | Artifact type, feature contract, and published result could disagree. | The release artifact contains one complete pipeline plus threshold and versioned metadata. |
| The README claimed SVC/F1 0.743 while the dirty artifact was an XGBoost model expecting 37 one-hot columns. | Documentation and deployed behavior were not traceable to the same run. | Reports and model metadata are generated together by one training command. |
| The app described Logistic Regression but loaded whichever estimator happened to be serialized. | Users could not know which model produced a result. | The interface reads the model label and version directly from artifact metadata. |
| The app converted AQ answers to a 0–10 sum and passed it as `result`; the training field is continuous from about −6.1 to 15.9. | The deployed feature had different meaning and distribution than training. | `result` is removed from the deployable model contract. |
| The app hard-coded full-dataset target/frequency encodings for demographic groups. | This duplicated leakage and failed for new categories. | The pipeline uses train-fitted imputation and `OneHotEncoder(handle_unknown="ignore")`. |
| `age_desc` says “18 and more,” but 223 of 800 ages are below 18. | Scope and data contradict each other. | Production evaluation is explicitly adult-only (18–100); the exclusion is reported. |
| The source contains 203 unknown ethnicities and 40 unknown relations. | Missingness and subgroup imbalance limit generalization and fairness claims. | Missing values are retained, imputed in-fold, and documented as a limitation. |
| The hybrid notebook's CV loop calculated scores but then trained each base model on all training rows to populate `meta_train`. | The MLP learned from in-sample base predictions, producing an optimistic stacking result. | The production hybrid uses `StackingClassifier` to generate genuine out-of-fold probabilities inside every outer training partition. |
| The README was truncated, its tree was mojibake, and no exact run command was complete. | A reviewer could not reproduce the project. | Documentation, package metadata, CI, tests, and commands are rebuilt. |

## Notebook-by-notebook disposition

- `Dataprocessing.ipynb`: useful exploratory history; its pre-split target/frequency encoding must
  not be used for production evaluation.
- `BaseModels.ipynb`: useful baseline exploration; CV scores are not directly comparable to the new
  nested protocol.
- `Ensemble.ipynb`: useful tree/boosting exploration; retained as a historical experiment.
- `Meta_Learning.ipynb`: demonstrates stacking concepts, but adds variance to a small dataset and is
  not selected merely for complexity.
- `Hybrid_Ensemble.ipynb`: contains the novel Random Forest + SVC + AdaBoost → MLP idea, but its
  `meta_train` values are in-sample despite the nearby CV loop. The corrected implementation reaches
  nested F1 0.629 ± 0.072 and is retained as a negative complexity result.
- `Hyper_Parameter.ipynb`: historical tuning log; test-set winner selection has been replaced.
- `Best_Model.ipynb`: historical visualization; release metrics now come from `reports/metrics.json`.

## Production definition

The reviewed production path is:

1. `load_training_data` validates schema and limits scope to adults.
2. A deterministic stratified holdout is removed before model selection.
3. Logistic Regression, RBF SVC, Random Forest, Extra Trees, and the corrected hybrid OOF MLP stack
   are compared with nested stratified CV.
4. Imputation, scaling, and one-hot encoding are fitted inside every inner fold.
5. The decision threshold is selected from out-of-fold training probabilities.
6. The selected pipeline is fitted on all training rows and evaluated once on the holdout.
7. The model, threshold, schema, category options, versions, and data hash are serialized together.
8. The selected Extra Trees pipeline is exported for browser inference and parity-tested against Python.
