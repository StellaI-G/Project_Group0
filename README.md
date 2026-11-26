# Predicting Poor Households in Indonesia with Machine Learning

**Group 0** · Workshop Fundamentals of Data Science, Fall 2026, University of St. Gallen
Alexa Cerda Roch · Yael Erdin · Anna Frei · Stella Inès Gindre

## Project in short

Indonesia's government selects households for cash transfers with a *proxy-means test* (PMT): a formula that predicts consumption from things an enumerator can observe at the door (house, assets, household members). Each village has a fixed number of transfers (its quota), and the households with the lowest predicted consumption get them.

We ask: **using the same information, can machine learning select more poor households under each village's quota than the government formula?** This is a pure prediction question, not a causal one.

- **Unit:** household. **Cutoff:** the door-to-door asset survey, before any allocation. Consumption is never an input.
- **Two races:** predict log consumption and rank (regression race, notebooks 04a), or predict the probability of being poor directly (classification race, notebooks 04b).
- **Decision rule:** in each village, the `round(village size × share)` households ranked poorest get a transfer.
- **Main metric:** **recall at the quota**, the share of all poor households that receive a transfer.
- **References:** random selection (floor), the government formula `pcexppred` (benchmark), ranking by true consumption (ceiling), and an OLS refit of the government's kind of formula.

## Data

Alatas, V., Banerjee, A., Hanna, R., Olken, B. A., & Tobias, J. (2012). *Targeting the Poor: Evidence from a Field Experiment in Indonesia.* American Economic Review, 102(4), 1206–1240.
Replication package: Harvard Dataverse, doi:[10.7910/DVN/M7SKQZ](https://doi.org/10.7910/DVN/M7SKQZ).

We use two files from the package, placed in `data/raw/`:

| File | Content |
|---|---|
| `coding_suseti_pmt.dta` | Household survey, the government formula's variables and its prediction `pcexppred` |
| `mistargeting_CORRECTED.dta` | Poverty label `poor` and the subvillage-head flag |

After cleaning: **5,113 households in 640 villages** (3 provinces, 12 districts), 36% poor. The 640 subvillage heads are removed, because they were added to each village on purpose and not sampled.

## Repository structure

```
Project_Group0/
├── data/
│   ├── raw/                 original .dta files (input of 01)
│   └── processed/           written by 01: clean data, test villages, CV folds
├── notebooks/
│   ├── 01_data_prep.ipynb               cleaning, merge, test split, CV folds
│   ├── 02_eda.ipynb                     exploration (training villages only)
│   ├── 03_baselines.ipynb               random, government formula, perfect ranking, OLS refit
│   ├── 04a_regression_setA.ipynb        regression race, feature set A
│   ├── 04a_regression_setB.ipynb        regression race, feature set B (extension)
│   ├── 04b_classification_setA.ipynb    classification race, feature set A
│   ├── 04b_classification_setB.ipynb    classification race, feature set B (extension)
│   ├── 05_evaluation.ipynb              final comparison and the one-time test
│   └── main_notebooks.ipynb             all notebooks above combined in one file
├── outputs/figures/         figures for the report and slides (see below)
├── src/
│   ├── utils.py             SEED, load_data(), quota rule, recall, within-village AUC
│   └── features.py          feature sets A and B, preprocessing pipeline
├── requirements.txt
└── README.md
```

## Setup

**With uv:**
```
uv venv .venv --python 3.15.2
source .venv/bin/activate
uv pip install -r requirements.txt
```

**With pip:**
```
python3.15 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebooks in VS Code or Jupyter and select the `.venv` kernel.

**pandas 3.0 or newer is required.** The clean data is saved as a pickle with pandas 3, and older versions cannot read it.

## How to reproduce

**Option 1: one notebook.** `notebooks/main_notebooks.ipynb` contains the whole project
(cleaning, exploration, baselines, both races, final test). Run it top to bottom (Restart → Run All).

**Option 2: the individual notebooks.** The same steps, split by topic:

- `01_data_prep` cleans the data and writes `data/processed/` (clean households, the
  128 test villages and the 5 CV folds). These files are included in the repository,
  so this step is only needed to recreate them.
- `02_eda`, `03_baselines`, `04a_regression_setA/B`, `04b_classification_setA/B` and
  `05_evaluation` only need `data/processed/` and can be run in any order. The 04
  notebooks take a few minutes each because of the tree-boosting and neural-network searches.

`05_evaluation` is the only notebook that uses the test villages, and only once, at the end.

### Random seed

`SEED = 40` is set once in `src/utils.py` and imported everywhere. It fixes:
- the test split (20% of villages within each district) and the 5 CV folds,
- the random-selection baseline (average of 100 random draws),
- `RandomizedSearchCV`, the tree-boosting models and the neural network (`random_state=SEED`).

With the same package versions, all results are reproduced exactly. The neural network is the only model that is sensitive to its seed; notebook 04b shows how its result changes with 5 different seeds.

### Rules that prevent leakage

- **Village split.** Train/test split and CV folds are drawn by village (all households of a village stay together), stratified by district. The test villages are used only once, in `05_evaluation`.
- **Same folds for every model.** The 5 folds are saved in `data/processed/cv_folds.csv` and passed with `PredefinedSplit`.
- **Preprocessing inside pipelines.** Median / most-frequent imputation, Yeo-Johnson and scaling are learned on the training folds only (`make_preprocessor` in `src/features.py`).
- **Allowed inputs only.** Consumption, the government score and its pieces, and anything recorded after the allocation are dropped in `01_data_prep`. `pcexppred` is used only as the benchmark, never as a feature.

## Feature sets

- **Set A (main):** the 62 variables with a coefficient in the government formula, 3 missing-value indicators and the district (66 columns). The linear models drop the 13 village-level columns, because a column that is the same for every household in a village cannot change the ranking inside the village (53 columns). Tree boosting and the neural network keep them.
- **Set B (extension):** set A plus 39 items an enumerator can see or verify (13 assets and 26 asset counts), 105 columns.

## Main results

Recall at the quota (share of poor households reached), feature set A:

| | Cross-validation (512 villages, mean of 5 folds) | Test (128 villages, used once) |
|---|---|---|
| Random selection | 36.4% | 32.4% |
| Government formula | 54.0% | 49.1% |
| Logistic regression (classification winner) | 56.2% | 50.7% |
| **OLS (regression winner, final model)** | **56.1%** | **51.5%** |
| Perfect ranking | 68.6% | 64.8% |

- In both races, all models (penalized linear models, tree boosting, neural network) were within one standard deviation of the best, so the simplest model won: OLS and logistic regression.
- OLS was chosen as the final model before the test: it ties with the logistic regression on recall, ranks neighbours slightly better (within-village AUC 0.824 vs. 0.814) and still works if the poverty line or the quota changes.
- On the test villages, OLS reaches 9 more poor households than the government formula (193 vs. 184 of 375). The gain is small, and the test has only 128 villages.
- Set B lowers the prediction error, but improves recall by less than one point.

## Figures

All figures are in `outputs/figures/`:

| File | Content | Notebook |
|---|---|---|
| `eda_consumption.png` | Consumption of poor and non-poor households | 02 |
| `04a_regression_*.png` | Regression race: penalty paths and CV recall | 04a |
| `04b_setA_logloss_by_C.png`, `04b_setB_logloss_by_C.png` | Logistic regression: CV log loss by penalty strength | 04b |
| `04b_setA_cv_recall.png`, `04b_setB_cv_recall.png` | Classification race: CV recall at the quota | 04b |
| `final_cv_recall.png` | Final comparison of the winners (cross-validation) | 05 |
| `final_test_recall.png` | Result on the test villages | 05 |
