import logging
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.isotonic import IsotonicRegression

# Configure logger
logger = logging.getLogger(__name__)


class StackedEnsemble:
    """
    Ensemble classifier that generates propensity scores using cross-validation predictions.

    Parameters:
    -----------
    base_learners : dict
        Dictionary of {name: estimator} pairs for base models
    scaler : sklearn transformer, default=RobustScaler()
        Preprocessing scaler to apply before each base learner
    stacking_learner : sklearn classifier, default=LogisticRegression()
        Meta-learner that combines base model predictions
    """

    def __init__(
        self, base_learners, scaler=None, stacking_learner=None, n_folds_stacking=5
    ):
        self.base_learners = base_learners
        self.scaler = scaler if scaler is not None else RobustScaler()
        self.stacking_learner = (
            stacking_learner
            if stacking_learner is not None
            else LogisticRegression(random_state=42)
        )
        self.n_folds_stacking = n_folds_stacking

        logger.info(
            f"StackedEnsemble initialized with {len(base_learners)} base learners: {list(base_learners.keys())}"
        )
        logger.info(f"Scaler: {type(self.scaler).__name__}")
        logger.info(f"Stacking learner: {type(self.stacking_learner).__name__}")

    def predict_ps(self, X, y, n_folds=5, calibrate=True):
        """
        Get propensity score predictions using cross-validation from base learners and stacked learner.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict
        y : array-like of shape (n_samples,)
            Target labels for cross-validation
        n_folds : int, default=3
            Number of cross-validation folds

        Returns:
        --------
        predictions : dict
            Dictionary with propensity scores from each base learner and stacked learner
        """
        logger.info(
            f"Generating propensity scores for {len(X)} samples using {n_folds}-fold CV"
        )
        predictions = {}

        # Get cross-validation predictions from each base learner
        for name, learner in self.base_learners.items():
            try:
                # Create pipeline with scaling
                pipeline = Pipeline(
                    [("scaler", clone(self.scaler)), ("learner", clone(learner))]
                )
                # Use cross_val_predict to get out-of-fold predictions
                cv_proba = cross_val_predict(
                    pipeline, X, y, method="predict_proba", cv=n_folds
                )
                ps_scores = cv_proba[:, 1]  # Get probability of positive class
                predictions[name] = ps_scores
                logger.debug(
                    f"Generated CV PS for {name}: mean={ps_scores.mean():.4f}, std={ps_scores.std():.4f}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate CV predictions for {name}: {e}")

        # Get cross-validation predictions from stacked learner
        try:
            stacked_classifer = StackingClassifier(
                estimators=[
                    (name, clone(learner))
                    for name, learner in self.base_learners.items()
                ],
                final_estimator=clone(self.stacking_learner),
                cv=self.n_folds_stacking,
            )
            stacked_pipeline = Pipeline(
                [("scaler", clone(self.scaler)), ("learner", stacked_classifer)]
            )
            cv_proba = cross_val_predict(
                stacked_pipeline, X, y, method="predict_proba", cv=n_folds
            )
            stacked_ps = cv_proba[:, 1]
            predictions["stacked"] = stacked_ps
            logger.debug(
                f"Generated CV PS for stacked: mean={stacked_ps.mean():.4f}, std={stacked_ps.std():.4f}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to generate CV predictions for stacked learner: {e}"
            )

        if calibrate:
            # Calibrate propensity scores if needed
            calibration_dict = {}
            for name, ps in predictions.items():
                if len(ps) > 0:
                    # Fit Isotonic Regression or Logistic Regression for calibration
                    try:
                        calibration_model = IsotonicRegression(
                            out_of_bounds="clip", y_min=0, y_max=1
                        )
                        calibration_model.fit(ps, y)
                        calibrated_ps = calibration_model.predict(ps)
                        calibration_dict[name + "_calibrated"] = calibrated_ps
                        logger.debug(
                            f"Calibrated PS for {name}: mean={calibrated_ps.mean():.4f}, std={calibrated_ps.std():.4f}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to calibrate PS for {name}: {e}")
            predictions.update(calibration_dict)
        logger.info(
            f"Propensity score predictions completed for {len(predictions)} models"
        )
        return predictions
