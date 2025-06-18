import shap
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

class ShapAnalyzer:
    def __init__(self, X, y, model_type, target_name, scaler=None):
        """
        X: DataFrame or np.ndarray, features (not scaled)
        y: Series or np.ndarray, target values
        model_type: str, one of 'lgbm', 'xgb', 'cat', 'rf'
        target_name: str, e.g. 'Q1', 'S1', etc.
        scaler: fitted scaler (e.g., StandardScaler), or None
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.target_name = target_name
        self.scaler = scaler
        self.model = None

    def fit_model(self):
        X = self.X
        if self.scaler:
            X = self.scaler.transform(X)
        if self.model_type == 'lgbm':
            if self.target_name == 'S1':
                self.model = LGBMClassifier(objective='multiclass', n_estimators=300, class_weight='balanced', random_state=42)
            else:
                self.model = LGBMClassifier(objective='binary', n_estimators=300, class_weight='balanced', random_state=42)
        elif self.model_type == 'xgb':
            if self.target_name == 'S1':
                self.model = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            else:
                self.model = XGBClassifier(objective='binary:logistic', n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif self.model_type == 'cat':
            if self.target_name == 'S1':
                self.model = CatBoostClassifier(loss_function='MultiClass', iterations=300, verbose=0, random_seed=42)
            else:
                self.model = CatBoostClassifier(loss_function='Lo                cd /home/juhyeong/20251R0136COSE47101
                git add shap.py
                git commit -m "Add ShapAnalyzer class for SHAP value analysis"
                git pushgloss', iterations=300, verbose=0, random_seed=42)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        else:
            raise ValueError("Unsupported model type for SHAP analysis.")
        self.model.fit(X, self.y)

    def compute_shap(self, plot=True, max_display=20):
        X = self.X
        if self.scaler:
            X = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        if plot:
            if self.target_name == 'S1' and isinstance(shap_values, list):
                # For multiclass, plot for each class
                for i, class_shap in enumerate(shap_values):
                    print(f"SHAP summary for class {i} of {self.target_name}")
                    shap.summary_plot(class_shap, X, feature_names=self.X.columns, max_display=max_display)
            else:
                shap.summary_plot(shap_values, X, feature_names=self.X.columns, max_display=max_display)
        return shap_values

    @staticmethod
    def find_best_target(f1_score_list, targets):
        best_idx = np.argmax(f1_score_list)
        return targets[best_idx], best_idx