"""Which columns each model sees: feature sets A and B.

Rows (training / test / folds) are chosen by the split in 01_data_prep and loaded with
src.utils.load_data(). This file only chooses COLUMNS, so the same lists work for the
training data, every CV fold and the test data:

    from src.utils import load_data
    from src.features import SET_A, SET_B

    train, test = load_data()
    X_train, X_test = train[SET_A], test[SET_A]
    y_train, y_test = train["LOGCONSUMPTION"], test["LOGCONSUMPTION"]   # or "poor"

Nothing here is learned from the data (no means, medians, ...), so using the lists
cannot leak test information. Filling, scaling and one-hot encoding happen in the
model pipeline, fitted on the training folds only.
"""

# ---------------------------------------------------------------------------
# Set A: the 62 variables of the government PMT formula (those with a c_* coefficient)
# why: same information as the government, so any gain comes from the method
# ---------------------------------------------------------------------------
A_DWELLING  = ["pcfloor", "tfloor", "twall", "troof", "toilet", "water", "lighting", "fcook", "house"]
A_HOUSEHOLD = ["hhsize", "hhsize2", "age04", "eschild", "jschild", "sschild", "depratio"]
A_HEAD      = ["hhage", "hhage2", "hhmale", "hhmarried", "hhmalemarr",
               "hheduc2", "hheduc3", "hheduc4",              # hheduc1 (no school) = reference, dropped
               "hhsector1", "hhsector2", "hhsector3", "formal", "informal"]
A_EDUC      = ["higheduc2", "higheduc3", "higheduc4"]
A_CREDIT    = ["credit"]                                     # self-reported, but in the formula
A_LOCATION  = ["urban"]
A_VILLAGE   = ["vill2dist", "villsd", "villsmp", "villpusk", "villpolin", "villposy",
               "villdokt", "villbidan", "villrtype", "villpmplace", "villcred"]
A_ASSETS    = ["se2_17", "se2_11_12", "se2_125", "se2_126_127", "se2_119", "se2_117",
               "se2_115", "se2_13_15", "se2_14", "se2_114", "se2_18", "se2_124",
               "se2_122_123", "se5_110", "se5_11_12", "se5_15_16", "se2_138"]

FORMULA = (A_DWELLING + A_HOUSEHOLD + A_HEAD + A_EDUC + A_CREDIT
           + A_LOCATION + A_VILLAGE + A_ASSETS)

# Indicators built in 01_data_prep (step 7)
# why: they keep the information "value was missing / cannot be computed" before filling
INDICATORS = ["hheduc_missing", "no_workingage", "pcfloor_missing"]

# District: a code, not a quantity -> one-hot in the pipeline, never used as a number
DISTRICT = ["kabu"]

SET_A = FORMULA + INDICATORS + DISTRICT

# ---------------------------------------------------------------------------
# Set B (extension): set A + extra items an enumerator can see or verify
# ---------------------------------------------------------------------------
B_ASSETS = ["se2_16", "se2_111", "se2_110", "se2_118", "se2_112", "se2_121",
            "se2_132_133", "se2_128", "se2_129", "se5_17", "se5_13", "se2_139",
            "FLOORTOPQUALITY"]
B_COUNTS = ["hr0609", "hr0610", "hr0611", "hr0612", "hr0613", "hr0614", "hr0615",
            "hr0616", "hr0617", "hr0618", "hr0619", "hr0620", "hr0621", "hr0622",
            "hr0623", "hr0625", "hr0626", "hr0627", "hr0628", "hr0629", "hr0630",
            "hr0631", "hr0632", "hr0633", "hr0634", "hr0635"]   # hr0607, hr0608 excluded: no label

SET_B = SET_A + B_ASSETS + B_COUNTS

# ---------------------------------------------------------------------------
# Never features: IDs, grouping, quota, baseline, targets, fold
# ---------------------------------------------------------------------------
TARGETS      = ["LOGCONSUMPTION", "poor"]
NOT_FEATURES = ["hhid", "hhea", "prop", "share", "pcexppred", "fold"] + TARGETS

# ---------------------------------------------------------------------------
# Column types, for the preprocessing pipeline (make_preprocessor, to be added)
# ---------------------------------------------------------------------------
# numbers: impute, Yeo-Johnson on the skewed ones, then scale (linear models, NN)
#   missing values left by the cleaning: pcfloor (51), hhage and hhage2 (6 each) -> training median;
#   depratio (220, no member aged 15-64) -> training maximum (outline section 2), flagged by no_workingage;
#   a few asset counts where the item is owned but the number is unknown -> training median
NUMERIC = ["pcfloor", "hhsize", "hhsize2", "age04", "eschild", "jschild", "sschild",
           "depratio", "hhage", "hhage2", "vill2dist"] + B_COUNTS
SKEWED  = ["pcfloor", "vill2dist", "depratio"] + B_COUNTS
# codes: one-hot
CATEGORICAL = DISTRICT
# everything else in a set is a 0/1 dummy: impute most frequent, pass through

# Same value for every household in a village: cannot change the ranking WITHIN a village.
# Feedback: linear models leave them out, boosting and the neural net keep them.
VILLAGE_LEVEL = A_LOCATION + A_VILLAGE + DISTRICT


def numeric(cols):
    """Numeric columns of a feature set (keeps the order of the set)."""
    return [c for c in cols if c in NUMERIC]


def binary(cols):
    """0/1 columns of a feature set."""
    return [c for c in cols if c not in NUMERIC + CATEGORICAL]


def categorical(cols):
    """Code columns of a feature set (one-hot in the pipeline)."""
    return [c for c in cols if c in CATEGORICAL]


def without_village_level(cols):
    """A feature set without the village-level columns (for the linear models)."""
    return [c for c in cols if c not in VILLAGE_LEVEL]


def check_feature_sets(data):
    """Stop with a clear message if the lists do not fit the data."""
    assert len(FORMULA) == 62, f"set A formula should have 62 variables, has {len(FORMULA)}"
    for name, cols in [("SET_A", SET_A), ("SET_B", SET_B)]:
        assert len(cols) == len(set(cols)), f"{name}: a column is listed twice"
        missing = [c for c in cols if c not in data.columns]
        assert not missing, f"{name}: columns not in the data: {missing}"
        leak = [c for c in cols if c in NOT_FEATURES]
        assert not leak, f"{name}: contains non-features (targets, IDs, baseline): {leak}"
    assert set(SET_A) <= set(SET_B), "set B must contain all of set A"
    for name, cols in [("SET_A", SET_A), ("SET_B", SET_B)]:
        assert set(numeric(cols)) | set(binary(cols)) | set(categorical(cols)) == set(cols)
        wrong = [c for c in binary(cols) if not set(data[c].dropna().unique()) <= {0, 1}]
        assert not wrong, f"{name}: listed as 0/1 but has other values: {wrong}"
    unused = [c for c in data.columns if c not in SET_B + NOT_FEATURES]
    assert not unused, f"columns in the clean data that are in no list: {unused}"
