"""
Single source of truth for the model set.

Both the baseline trainer and the degradation tester build their models here.
They previously each defined their own, and the definitions silently diverged:
the degradation script omitted XGBoost's learning_rate, so it defaulted to 0.3
against the baseline's 0.1, and 360 rows measured a hyperparameter change on top
of the data corruption. Defining the models once makes that failure impossible
rather than merely unlikely.

Model set (8), deliberately balanced 4 tree-based vs 4 non-tree so RQ3 can ask
whether cleaning priority depends on the model FAMILY rather than on individual
algorithms:

    tree-based : Decision Tree, Random Forest, XGBoost, LightGBM
    non-tree   : Logistic Regression, Gaussian Naive Bayes, KNN, MLP

Every model exposes predict_proba, which the AUC calculation requires. Models
without it (LinearSVC, RidgeClassifier) are deliberately excluded: including
them would mean changing the metric code, and the measurement path is the last
place to take on avoidable risk.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Family labels, used by the RQ3 analysis to group models.
MODEL_FAMILY = {
    'Logistic Regression': 'non-tree',
    'Gaussian Naive Bayes': 'non-tree',
    'KNN': 'non-tree',
    'MLP': 'non-tree',
    'Decision Tree': 'tree',
    'Random Forest': 'tree',
    'XGBoost': 'tree',
    'LightGBM': 'tree',
}


def build_models(random_state=42):
    """
    Return a fresh, unfitted model per name.

    Fresh instances matter: a model reused across experiments would carry state
    from the previous fit.
    """
    return {
        # -- non-tree --------------------------------------------------------
        'Logistic Regression': LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        'Gaussian Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # One hidden layer and a 200-iteration cap keep the run tractable: MLP
        # is the slowest model here at ~9s per fit on the largest dataset. It
        # may stop before full convergence, which is a deliberate cost/benefit
        # choice and applies identically to clean and corrupted data, so it
        # cannot bias the comparison between them.
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=200, random_state=random_state
        ),

        # -- tree-based ------------------------------------------------------
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state, max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=random_state, max_depth=10, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=random_state, max_depth=6,
            learning_rate=0.1, verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, random_state=random_state, max_depth=6,
            learning_rate=0.1, verbose=-1
        ),
    }
