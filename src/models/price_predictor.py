import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, \
    StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectFromModel, RFE
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.best_model = None
        self.model_performance = {}
        self.feature_importance = {}
        self.is_trained = False

    def advanced_feature_selection(self, X, y, method='ensemble'):
        """Advanced feature selection using multiple methods"""
        try:
            feature_importance_df = None  # Initialize variable

            if method == 'ensemble':
                # Use multiple feature selection methods
                selected_features_sets = []

                # Method 1: Random Forest importance
                rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_selector.fit(X, y)
                rf_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf_selector.feature_importances_
                }).sort_values('importance', ascending=False)
                selected_features_sets.append(set(rf_importance.head(30)['feature']))

                # Method 2: XGBoost importance
                xgb_selector = XGBClassifier(n_estimators=100, random_state=42)
                xgb_selector.fit(X, y)
                xgb_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': xgb_selector.feature_importances_
                }).sort_values('importance', ascending=False)
                selected_features_sets.append(set(xgb_importance.head(30)['feature']))

                # Method 3: LGBM importance
                lgbm_selector = LGBMClassifier(n_estimators=100, random_state=42)
                lgbm_selector.fit(X, y)
                lgbm_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': lgbm_selector.feature_importances_
                }).sort_values('importance', ascending=False)
                selected_features_sets.append(set(lgbm_importance.head(30)['feature']))

                # Get features selected by at least 2 methods
                feature_votes = {}
                for feature_set in selected_features_sets:
                    for feature in feature_set:
                        feature_votes[feature] = feature_votes.get(feature, 0) + 1

                selected_features = [feature for feature, votes in feature_votes.items() if votes >= 2]

                if len(selected_features) < 15:
                    # Fallback to top features from RF
                    selected_features = rf_importance.head(20)['feature'].tolist()

                # Use RF importance for feature importance tracking
                feature_importance_df = rf_importance

            else:
                # Simple method using RandomForest
                selector = RandomForestClassifier(n_estimators=100, random_state=42)
                selector.fit(X, y)

                importance = selector.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                # Select top features
                n_features = max(20, min(40, len(X.columns) // 2))
                selected_features = feature_importance_df.head(n_features)['feature'].tolist()

            # Store feature importance
            if feature_importance_df is not None:
                self.feature_importance = {row['feature']: row['importance']
                                           for _, row in feature_importance_df.iterrows()}

            logger.info(f"Selected {len(selected_features)} features using {method} method")
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            # Fallback: return all features
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': [1.0 / len(X.columns)] * len(X.columns)
            })
            self.feature_importance = {row['feature']: row['importance']
                                       for _, row in feature_importance_df.iterrows()}
            return X, X.columns.tolist()

    def train_models(self, X, y, test_size=0.2):
        """Train advanced ensemble models with enhanced features"""
        try:
            # Validate input data
            if X.empty or y.empty:
                logger.error("Empty training data provided")
                return {}

            # Advanced data preprocessing
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)

            # Remove constant features
            constant_features = X.columns[X.nunique() <= 1]
            if len(constant_features) > 0:
                logger.info(f"Removing {len(constant_features)} constant features")
                X = X.drop(columns=constant_features)

            # Advanced feature selection
            X_selected, selected_features = self.advanced_feature_selection(X, y)
            self.feature_columns = selected_features

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42, stratify=y, shuffle=True
            )

            logger.info(f"Training data - X_train: {X_train.shape}, X_test: {X_test.shape}")

            # Advanced scaling strategies
            scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(output_distribution='normal', random_state=42)
            }

            scaled_data = {}
            for name, scaler in scalers.items():
                try:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    scaled_data[name] = (X_train_scaled, X_test_scaled)
                    self.scalers[name] = scaler
                except Exception as e:
                    logger.error(f"Error with {name} scaler: {e}")

            # Define advanced models with hyperparameter tuning
            models = {
                'xgboost_tuned': XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'lightgbm_tuned': LGBMClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'catboost_tuned': CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=False
                ),
                'random_forest_enhanced': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced'
                ),
                'gradient_boosting_enhanced': GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42
                ),
                'logistic_regression_enhanced': LogisticRegression(
                    C=0.1,
                    penalty='l2',
                    solver='liblinear',
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                )
            }

            # Train and evaluate each model
            best_score = 0
            self.best_model = None

            for name, model in models.items():
                logger.info(f"Training {name}...")

                try:
                    # Choose appropriate scaler
                    if name in ['logistic_regression_enhanced']:
                        scaler_type = 'quantile'
                    elif name in ['xgboost_tuned', 'lightgbm_tuned', 'catboost_tuned']:
                        scaler_type = 'none'
                    else:
                        scaler_type = 'standard'

                    if scaler_type != 'none' and scaler_type in scaled_data:
                        X_train_scaled, X_test_scaled = scaled_data[scaler_type]
                    else:
                        X_train_scaled, X_test_scaled = X_train.values, X_test.values

                    # Fit model
                    model.fit(X_train_scaled, y_train)

                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)

                    # Calculate comprehensive metrics with error handling
                    accuracy = accuracy_score(y_test, y_pred)

                    # Handle precision calculation safely
                    try:
                        precision = precision_score(y_test, y_pred, zero_division=0)
                    except:
                        precision = 0

                    # Handle recall calculation safely
                    try:
                        recall = recall_score(y_test, y_pred, zero_division=0)
                    except:
                        recall = 0

                    # Handle F1 calculation safely
                    try:
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                    except:
                        f1 = 0

                    # ROC AUC for binary classification
                    roc_auc = 0
                    try:
                        if len(np.unique(y_test)) == 2 and len(y_pred_proba[0]) == 2:
                            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    except:
                        roc_auc = 0

                    # Enhanced cross-validation with stratification
                    cv_mean, cv_std = 0, 0
                    try:
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    except:
                        cv_mean, cv_std = 0, 0

                    # Store model and performance with all required keys
                    self.models[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'cv_score': cv_mean,
                        'cv_std': cv_std,
                        'scaler_type': scaler_type,
                        'feature_importance': getattr(model, 'feature_importances_', None)
                    }

                    logger.info(
                        f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}, CV: {cv_mean:.3f} ± {cv_std:.3f}")

                    # Update best model based on F1 score and ROC AUC
                    combined_score = f1 * 0.6 + roc_auc * 0.4
                    if combined_score > best_score:
                        best_score = combined_score
                        self.best_model = name

                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue

            # Create advanced ensemble
            self.create_advanced_ensemble(X_train, y_train, X_test, y_test, scaled_data)

            # Store best model performance with safe key access
            if self.best_model and self.best_model in self.models:
                best_model_info = self.models[self.best_model]

                # Create performance dictionary with safe defaults
                performance_dict = {
                    'best_model': self.best_model,
                    'accuracy': best_model_info.get('accuracy', 0),
                    'precision': best_model_info.get('precision', 0),
                    'recall': best_model_info.get('recall', 0),
                    'f1_score': best_model_info.get('f1_score', 0),
                    'roc_auc': best_model_info.get('roc_auc', 0),
                    'cv_score': best_model_info.get('cv_score', 0),
                    'cv_std': best_model_info.get('cv_std', 0),
                    'feature_columns': self.feature_columns
                }

                self.model_performance = performance_dict
                self.is_trained = True

                logger.info(f"🎯 Best model: {self.best_model}")
                logger.info(f"📊 Accuracy: {performance_dict['accuracy']:.3f}, "
                            f"F1: {performance_dict['f1_score']:.3f}, "
                            f"ROC AUC: {performance_dict['roc_auc']:.3f}")
                return self.model_performance
            else:
                logger.error("No model was successfully trained")
                return {}

        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def create_advanced_ensemble(self, X_train, y_train, X_test, y_test, scaled_data):
        """Create advanced ensemble model"""
        try:
            # Select top models for ensemble
            ensemble_candidates = []
            for name, info in self.models.items():
                if info.get('f1_score', 0) > 0.55 and info.get('roc_auc', 0) > 0.6:  # Only include good models
                    ensemble_candidates.append((name, info['model']))

            if len(ensemble_candidates) >= 2:
                # Create voting classifier
                voting_clf = VotingClassifier(
                    estimators=ensemble_candidates,
                    voting='soft',
                    n_jobs=-1
                )

                # Choose scaler for ensemble (use standard as default)
                if 'standard' in scaled_data:
                    X_train_ensemble, X_test_ensemble = scaled_data['standard']
                else:
                    X_train_ensemble, X_test_ensemble = X_train.values, X_test.values

                voting_clf.fit(X_train_ensemble, y_train)
                y_pred = voting_clf.predict(X_test_ensemble)
                y_pred_proba = voting_clf.predict_proba(X_test_ensemble)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) == 2 else 0

                self.models['advanced_ensemble'] = {
                    'model': voting_clf,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'scaler_type': 'standard',
                    'feature_importance': None
                }

                logger.info(f"Advanced Ensemble - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")

                # Update best model if ensemble is better
                ensemble_score = f1 * 0.6 + roc_auc * 0.4
                current_best_score = self.models.get(self.best_model, {}).get('f1_score', 0) * 0.6 + \
                                     self.models.get(self.best_model, {}).get('roc_auc', 0) * 0.4

                if ensemble_score > current_best_score:
                    self.best_model = 'advanced_ensemble'
                    logger.info("🏆 Advanced ensemble model selected as best!")

        except Exception as e:
            logger.error(f"Error creating advanced ensemble: {e}")

    def predict_next_hour(self, current_features):
        """Predict next hour price movement with enhanced confidence"""
        try:
            if not self.best_model or self.best_model not in self.models:
                raise ValueError("No trained model available")

            model_info = self.models[self.best_model]
            model = model_info['model']

            # Ensure we have the right features in right order
            current_df = pd.DataFrame([current_features])
            missing_cols = set(self.feature_columns) - set(current_df.columns)
            extra_cols = set(current_df.columns) - set(self.feature_columns)

            # Add missing columns with 0
            for col in missing_cols:
                current_df[col] = 0

            # Remove extra columns
            current_df = current_df[self.feature_columns]

            # Scale features if needed
            if model_info['scaler_type'] in self.scalers:
                current_data = self.scalers[model_info['scaler_type']].transform(current_df)
            else:
                current_data = current_df.values

            # Make prediction
            prediction = model.predict(current_data)[0]
            probability = model.predict_proba(current_data)[0]

            confidence = max(probability)

            # Enhanced confidence calculation
            if len(probability) == 2:  # Binary classification
                up_probability = probability[1]
                down_probability = probability[0]
                prediction_label = 'UP' if prediction == 1 else 'DOWN'
            else:
                up_probability = probability[1] if len(probability) > 1 else 0.5
                down_probability = probability[0] if len(probability) > 0 else 0.5
                prediction_label = 'UP' if prediction == 1 else 'DOWN'

            result = {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'up_probability': float(up_probability),
                'down_probability': float(down_probability),
                'model_used': self.best_model,
                'model_accuracy': self.model_performance.get('accuracy', 0),
                'model_f1_score': self.model_performance.get('f1_score', 0)
            }

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'up_probability': 0.5,
                'down_probability': 0.5,
                'model_used': 'none',
                'error': str(e)
            }

    def save_model(self, filepath='models/trained_models.pkl'):
        """Save trained models and scalers"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            save_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'best_model': self.best_model,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Models saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_model(self, filepath='models/trained_models.pkl'):
        """Load trained models and scalers - FIXED to prevent repeated loading"""
        try:
            # Check if models are already loaded
            if self.is_trained and self.models:
                logger.info("Models already loaded, skipping reload")
                return True

            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.models = save_data['models']
            self.scalers = save_data['scalers']
            self.feature_columns = save_data['feature_columns']
            self.best_model = save_data['best_model']
            self.model_performance = save_data['model_performance']
            self.feature_importance = save_data.get('feature_importance', {})
            self.is_trained = save_data.get('is_trained', True)

            logger.info(f"Models loaded from {filepath}")
            logger.info(f"Best model: {self.best_model}")
            logger.info(f"Model accuracy: {self.model_performance.get('accuracy', 0):.3f}")

            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False