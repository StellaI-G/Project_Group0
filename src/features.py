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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

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
# Column types, for the preprocessing pipeline (make_preprocessor, below)
# ---------------------------------------------------------------------------
# numbers: impute, Yeo-Johnson on the skewed ones, then scale (linear models, NN)
#   missing values left by the cleaning: pcfloor (51), hhage and hhage2 (6 each) -> training median;
#   depratio (220, no member aged 15-64) -> training maximum (outline section 2), flagged by no_workingage;
#   a few asset counts where the item is owned but the number is unknown -> training median
NUMERIC = ["pcfloor", "hhsize", "hhsize2", "age04", "eschild", "jschild", "sschild",
           "depratio", "hhage", "hhage2", "vill2dist"] + B_COUNTS

# SKEWED: numeric columns that get Yeo-Johnson (PowerTransformer) in the linear models and the NN.
# why: a skewed column has many small values and a few very large ones (e.g. pcfloor: median 17 m²,
#      maximum 490 m²). In a linear model those few large values dominate the fit. Yeo-Johnson
#      stretches the crowded small values and compresses the long tail (notebook 211, section 3).
# rule (measured on the TRAINING villages only, so the test villages played no part in the choice):
#      skewness above 1  AND  Yeo-Johnson brings it below 0.5 (i.e. the transform actually works)
#                skew before -> after
#   pcfloor        5.4 -> 0.0    floor area per person
#   vill2dist      2.0 -> 0.0    distance to district capital
#   depratio       1.7 -> 0.1    dependency ratio
#   eschild        1.3 -> 0.4    children in elementary school
#   hr0628        17.1 -> 0.4    number of chickens
#   hr0627        15.5 -> 0.3    jewelry
#   hr0612         2.0 -> 0.3    fans
#   hr0619         1.9 -> 0.1    mobile phones
#   hr0621         1.4 -> 0.3    bicycles
#   hr0615         1.4 -> 0.1    TVs
#   hr0614         1.3 -> 0.3    radios
#   hr0623         1.3 -> 0.2    motorcycles
# NOT in the list, on purpose:
#   - counts where 85-100% of households have 0 (boats, AC, car, pigs, horses, ...): skew stays
#     high after the transform (e.g. 27 -> 20), because a pile of zeros cannot be made symmetric
#   - hhsize2, hhage2: they are the squares of hhsize and hhage from the government formula;
#     transforming them would break "size + size²"
#   - hhsize, hhage, age04, jschild, sschild, hr0616, hr0610, ...: skew below 1, or not fixed
# The columns not in the list are ordinary numbers: they are only standardized (StandardScaler).
SKEWED = ["pcfloor", "vill2dist", "depratio", "eschild",
          "hr0628", "hr0627", "hr0612", "hr0619", "hr0621", "hr0615", "hr0614", "hr0623"]

# codes: one-hot
CATEGORICAL = DISTRICT
# everything else in a set is a 0/1 dummy: impute most frequent, pass through

# Same value for every household in a village: cannot change the ranking WITHIN a village.
# Feedback: linear models leave them out, boosting and the neural net keep them.
VILLAGE_LEVEL = A_LOCATION + A_VILLAGE + DISTRICT


def numeric(cols):
    """Numeric columns of a feature set (keeps the order of the set)."""
    return [c for c in cols if c in NUMERIC]


def skewed(cols):
    """Skewed numeric columns of a feature set: impute, then Yeo-Johnson.

    PowerTransformer() already standardizes its output (standardize=True is the default),
    so these columns need NO StandardScaler afterwards.
    """
    return [c for c in cols if c in SKEWED]


def other_numeric(cols):
    """Numeric columns of a feature set that are not skewed: impute, then StandardScaler."""
    return [c for c in cols if c in NUMERIC and c not in SKEWED]


def binary(cols):
    """0/1 columns of a feature set."""
    return [c for c in cols if c not in NUMERIC + CATEGORICAL]


def categorical(cols):
    """Code columns of a feature set (one-hot in the pipeline)."""
    return [c for c in cols if c in CATEGORICAL]


def without_village_level(cols):
    """A feature set without the village-level columns (for the linear models)."""
    return [c for c in cols if c not in VILLAGE_LEVEL]


def make_preprocessor(cols, scale=True, drop_first=False):
    """Preprocessing for one feature set; put it in a Pipeline in front of the model.

    scale=True: linear models and neural net.  scale=False: boosting.
    drop_first=True: models without penalty (OLS, plain logistic regression).

    Everything here is learned from data (medians, Yeo-Johnson lambda, means, standard
    deviations, list of districts). Inside a Pipeline it is fitted on the training fold
    only and then applied unchanged to the validation / test data ("estimate, then freeze").
    """
    skewed_cols = skewed(cols)
    plain_cols  = other_numeric(cols)

    median = SimpleImputer(strategy="median")
    if scale:
        skewed_steps = Pipeline([
            # missing values are filled with the median: large values distort the mean,
            # so the typical value is the safer fill
            ("fill", median),
            # a few very large values could otherwise dominate the fit; this compresses the
            # long tail while keeping the order of the values
            ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True)),
        ])
    else:
        skewed_steps = median   # boosting: only fill with the median, trees need no transform

    columns = ColumnTransformer([   # (name, what is done, which columns)
        ("skewed",   skewed_steps, skewed_cols),
        ("numbers",  median, plain_cols),          # not skewed: only the median fill
        # most frequent of 0 and 1: at most 13 missing values per column, too few to move the model
        ("dummies",  SimpleImputer(strategy="most_frequent"), binary(cols)),
        # district code -> one 0/1 column per district; with drop_first one district is the reference
        ("district", OneHotEncoder(drop="first" if drop_first else None,
                                   handle_unknown="ignore", sparse_output=False),
                     categorical(cols)),
    ], remainder="drop", verbose_feature_names_out=False)

    if not scale:
        return columns
    # StandardScaler on all inputs. After Yeo-Johnson it is harmless: PowerTransformer already
    # returns mean 0 / sd 1, so the scaler subtracts 0 and divides by 1 for those columns.
    return Pipeline([("columns", columns), ("scale", StandardScaler())])


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
