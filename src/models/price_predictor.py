import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 6, 8]
}
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

    def feature_selection(self, X, y):
        """Select most important features"""
        try:
            # Use RandomForest for feature importance
            selector = RandomForestClassifier(n_estimators=100, random_state=42)
            selector.fit(X, y)

            # Get feature importance
            importance = selector.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Select top features (keep at least 15)
            n_features = max(15, min(30, len(X.columns) // 2))
            selected_features = feature_importance_df.head(n_features)['feature'].tolist()

            self.feature_importance = feature_importance_df.set_index('feature')['importance'].to_dict()

            logger.info(f"Selected {len(selected_features)} most important features")
            return X[selected_features], selected_features

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return X, X.columns.tolist()

    def train_models(self, X, y, test_size=0.2):
        """Train advanced models with enhanced features"""
        try:
            # Validate input data
            if X.empty or y.empty:
                logger.error("Empty training data provided")
                return {}

            # Final data cleaning - ensure no infinite values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)

            # Feature selection
            X_selected, selected_features = self.feature_selection(X, y)
            self.feature_columns = selected_features

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42, shuffle=False
            )

            logger.info(f"Training data - X_train: {X_train.shape}, X_test: {X_test.shape}")

            # Robust scaling that handles outliers better
            scaler_standard = StandardScaler()
            scaler_robust = RobustScaler()  # Better for data with outliers

            # Apply scaling with error handling
            try:
                X_train_standard = scaler_standard.fit_transform(X_train)
                X_test_standard = scaler_standard.transform(X_test)

                X_train_robust = scaler_robust.fit_transform(X_train)
                X_test_robust = scaler_robust.transform(X_test)

                self.scalers['standard'] = scaler_standard
                self.scalers['robust'] = scaler_robust

            except Exception as e:
                logger.error(f"Error in scaling: {e}")
                # Fallback: use original data without scaling
                X_train_standard = X_train.values
                X_test_standard = X_test.values
                X_train_robust = X_train.values
                X_test_robust = X_test.values

            # Define advanced models with better parameters
            models = {
                'random_forest_optimized': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced'
                ),
                'gradient_boosting_optimized': GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42
                ),
                'logistic_regression_optimized': LogisticRegression(
                    C=0.1,
                    penalty='l2',
                    solver='liblinear',
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'svm_rbf': SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ),
                'mlp_nn': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                )
            }

            # Train and evaluate each model
            best_score = 0
            self.best_model = None

            for name, model in models.items():
                logger.info(f"Training {name}...")

                try:
                    # Choose appropriate scaler and data
                    if name in ['logistic_regression_optimized', 'svm_rbf', 'mlp_nn']:
                        X_train_scaled = X_train_standard
                        X_test_scaled = X_test_standard
                        scaler_type = 'standard'
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                        scaler_type = 'none'

                    # Fit model
                    model.fit(X_train_scaled, y_train)

                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    # Cross-validation score (reduced folds for speed)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                    cv_mean = cv_scores.mean()

                    # Store model and performance
                    self.models[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_score': cv_mean,
                        'scaler_type': scaler_type,
                        'feature_importance': getattr(model, 'feature_importances_', None)
                    }

                    logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_mean:.3f}")

                    # Update best model based on F1 score (balanced metric)
                    if f1 > best_score:
                        best_score = f1
                        self.best_model = name

                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue

            # Create ensemble model if we have multiple good models
            self.create_ensemble_model(X_train_standard, y_train, X_test_standard, y_test)

            # Store best model performance
            if self.best_model and self.best_model in self.models:
                best_model_info = self.models[self.best_model]
                self.model_performance = {
                    'best_model': self.best_model,
                    'accuracy': best_model_info['accuracy'],
                    'precision': best_model_info['precision'],
                    'recall': best_model_info['recall'],
                    'f1_score': best_model_info['f1_score'],
                    'cv_score': best_model_info['cv_score'],
                    'feature_columns': self.feature_columns
                }

                logger.info(f"🎯 Best model: {self.best_model}")
                logger.info(f"📊 Accuracy: {best_model_info['accuracy']:.3f}, F1: {best_model_info['f1_score']:.3f}")
                return self.model_performance
            else:
                logger.error("No model was successfully trained")
                return {}

        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Create ensemble model from best individual models"""
        try:
            # Select models for ensemble (exclude weak ones)
            ensemble_models = []
            for name, info in self.models.items():
                if info['f1_score'] > 0.55:  # Only include decent models
                    ensemble_models.append((name, info['model']))

            if len(ensemble_models) >= 2:
                # Create voting classifier
                voting_clf = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft',
                    n_jobs=-1
                )

                voting_clf.fit(X_train, y_train)
                y_pred = voting_clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                self.models['ensemble_voting'] = {
                    'model': voting_clf,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'scaler_type': 'standard',
                    'feature_importance': None
                }

                logger.info(f"Ensemble Voting - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

                # Update best model if ensemble is better
                if f1 > self.models.get(self.best_model, {}).get('f1_score', 0):
                    self.best_model = 'ensemble_voting'
                    logger.info("🏆 Ensemble model selected as best!")

        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")

    def predict_next_hour(self, current_features):
        """Predict next hour price movement"""
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
            if model_info['scaler_type'] == 'standard':
                current_data = self.scalers['standard'].transform(current_df)
            elif model_info['scaler_type'] == 'robust':
                current_data = self.scalers['robust'].transform(current_df)
            else:
                current_data = current_df

            # Make prediction
            prediction = model.predict(current_data)[0]
            probability = model.predict_proba(current_data)[0]

            confidence = max(probability)

            result = {
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': float(confidence),
                'up_probability': float(probability[1]),
                'down_probability': float(probability[0]),
                'model_used': self.best_model
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
                'feature_importance': self.feature_importance
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Models saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_model(self, filepath='models/trained_models.pkl'):
        """Load trained models and scalers"""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.models = save_data['models']
            self.scalers = save_data['scalers']
            self.feature_columns = save_data['feature_columns']
            self.best_model = save_data['best_model']
            self.model_performance = save_data['model_performance']
            self.feature_importance = save_data.get('feature_importance', {})

            logger.info(f"Models loaded from {filepath}")
            logger.info(f"Best model: {self.best_model}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")