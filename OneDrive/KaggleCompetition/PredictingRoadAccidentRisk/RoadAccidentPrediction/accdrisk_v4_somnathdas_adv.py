# Road Accident Risk Prediction - Advanced Ensemble
# Target Score: 0.068+ - Fixed Version

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedAccidentPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load data with error handling for local paths"""
        try:
            # Use your local file paths
            self.train_data = pd.read_csv("C:/Users/somna/OneDrive/KaggleCompetition/PredictingRoadAccidentRisk/playground-series-s5e10/train.csv")
            self.test_data = pd.read_csv("C:/Users/somna/OneDrive/KaggleCompetition/PredictingRoadAccidentRisk/playground-series-s5e10/test.csv")
            #self.train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
            #self.test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
            #self.train_data = pd.read_csv("C:/Users/somna/OneDrive/KaggleCompetition/PredictingRoadAccidentRisk/playground-series-s5e10/train.csv")
            #self.test_data = pd.read_csv("C:/Users/somna/OneDrive/KaggleCompetition/PredictingRoadAccidentRisk/playground-series-s5e10/test.csv")
            print(f"✅ Loaded by Somnath Das: Train {self.train_data.shape}, Test {self.test_data.shape}")
            return True
        except FileNotFoundError:
            print("❌ Files not found. Please check file paths")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def create_advanced_features(self, df, is_train=True):
        """Create powerful feature engineering"""
        df = df.copy()
        
        # Enhanced categorical encoding
        categoricals = ['road_type', 'lighting', 'weather', 'time_of_day']
        for col in categoricals:
            if is_train:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_enc'] = self.encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_enc'] = self.encoders[col].transform(df[col])
        
        # Boolean conversion
        bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        for col in bool_cols:
            df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)
        
        # Advanced feature engineering
        df['speed_curvature_risk'] = df['speed_limit'] * df['curvature'] * 0.01
        df['lane_speed_density'] = df['num_lanes'] / (df['speed_limit'] + 1)
        df['environment_hazard'] = (df['weather_enc'] + df['lighting_enc'] + df['time_of_day_enc']) / 3
        df['infrastructure_score'] = df['num_lanes'] + df['road_signs_present'] - df['curvature']
        df['accident_history_impact'] = np.log1p(df['num_reported_accidents'])
        df['complexity_score'] = df['curvature'] * df['num_lanes'] * (1 + df['weather_enc'])
        df['night_risk'] = (df['time_of_day_enc'] > 2).astype(int) * df['lighting_enc']
        df['adverse_conditions'] = ((df['weather_enc'] > 1) | (df['lighting_enc'] > 1)).astype(int)
        
        # Polynomial features
        df['speed_squared'] = df['speed_limit'] ** 2
        df['curvature_squared'] = df['curvature'] ** 2
        
        return df

    def prepare_data(self):
        """Prepare training and test data with scaling"""
        # Feature engineering
        train_eng = self.create_advanced_features(self.train_data, is_train=True)
        test_eng = self.create_advanced_features(self.test_data, is_train=False)
        
        # Select features
        feature_cols = [
            'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
            'road_type_enc', 'lighting_enc', 'weather_enc', 'time_of_day_enc',
            'road_signs_present', 'public_road', 'holiday', 'school_season',
            'speed_curvature_risk', 'lane_speed_density', 'environment_hazard',
            'infrastructure_score', 'accident_history_impact', 'complexity_score',
            'night_risk', 'adverse_conditions', 'speed_squared', 'curvature_squared'
        ]
        
        X_train = train_eng[feature_cols]
        y_train = train_eng['accident_risk']
        X_test = test_eng[feature_cols]
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_test_scaled = self.scalers['main'].transform(X_test)
        
        return X_train_scaled, y_train, X_test_scaled

    def build_models(self):
        """Build multiple high-performing models"""
        models = {
            'rf1': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'rf2': RandomForestRegressor(
                n_estimators=250, max_depth=18, min_samples_split=3,
                random_state=123, n_jobs=-1
            ),
            'gb1': GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=6,
                subsample=0.8, random_state=42
            ),
            'gb2': GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.1, max_depth=5,
                subsample=0.85, random_state=123
            ),
            'ridge': Ridge(alpha=0.5, random_state=42),
            'lasso': Lasso(alpha=0.001, random_state=42, max_iter=2000),
        }
        return models

    def train_with_cv(self, X, y, model, model_name, n_splits=3):
        """Train with cross-validation and return scores"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if model_name in ['ridge', 'lasso']:
                # Scale for linear models
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores), np.std(cv_scores)

    def create_blend(self, X_train, y_train, X_test):
        """Create blended predictions from multiple models"""
        models = self.build_models()
        predictions = {}
        cv_results = {}
        
        print("🏋️ Training Multiple Models...")
        
        # Train each model
        for name, model in models.items():
            print(f"  Training {name}...")
            cv_mean, cv_std = self.train_with_cv(X_train, y_train, model, name)
            cv_results[name] = (cv_mean, cv_std)
            
            # Train final model on full data
            if name in ['ridge', 'lasso']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                predictions[name] = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions[name] = model.predict(X_test)
            
            print(f"    {name}: CV RMSE = {cv_mean:.5f} ± {cv_std:.5f}")
        
        # Weighted blending based on CV performance (lower RMSE = higher weight)
        weights = {}
        for name, (cv_mean, cv_std) in cv_results.items():
            # Inverse weighting: better models get higher weights
            weights[name] = 1.0 / (cv_mean + 0.0001)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: w/total_weight for name, w in weights.items()}
        
        print("\n📊 Blending Weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f} (CV: {cv_results[name][0]:.5f})")
        
        # Create weighted blend
        final_predictions = np.zeros_like(predictions['rf1'])
        for name, pred in predictions.items():
            final_predictions += pred * weights[name]
        
        return np.clip(final_predictions, 0, 1)

    def run_pipeline(self):
        """Execute complete pipeline"""
        print("🚀 Starting Advanced Prediction Pipeline...")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return None
        
        # Prepare data
        X_train, y_train, X_test = self.prepare_data()
        
        # Generate blended predictions
        print("\n🎯 Generating Ensemble Predictions...")
        final_predictions = self.create_blend(X_train, y_train, X_test)
        
        # Create submission
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'accident_risk': final_predictions
        })
        
        # Analysis
        print(f"\n📈 Final Prediction Stats:")
        print(f"Mean risk: {final_predictions.mean():.5f}")
        print(f"Std risk: {final_predictions.std():.5f}")
        print(f"Min risk: {final_predictions.min():.5f}")
        print(f"Max risk: {final_predictions.max():.5f}")
        
        # Risk distribution analysis
        risk_ranges = [
            (0.0, 0.1, '0.0-0.1'),
            (0.1, 0.2, '0.1-0.2'),
            (0.2, 0.3, '0.2-0.3'),
            (0.3, 0.4, '0.3-0.4'),
            (0.4, 0.5, '0.4-0.5'),
            (0.5, 0.6, '0.5-0.6'),
            (0.6, 0.7, '0.6-0.7'),
            (0.7, 0.8, '0.7-0.8'),
            (0.8, 0.9, '0.8-0.9'),
            (0.9, 1.0, '0.9-1.0')
        ]
        
        print("\n🎯 Risk Distribution:")
        for low, high, label in risk_ranges:
            count = ((final_predictions >= low) & (final_predictions < high)).sum()
            percentage = (count / len(final_predictions)) * 100
            print(f"  {label}: {count:6d} samples ({percentage:5.1f}%)")
        
        # Save submission
        submission_file = 'advanced_ensemble_submission.csv'
        submission.to_csv(submission_file, index=False)
        print(f"\n✅ Submission saved: {submission_file}")
        print("🎯 Target: RMSE < 0.068")
        print("=" * 60)
        
        return submission

# Execute pipeline
if __name__ == "__main__":
    predictor = AdvancedAccidentPredictor()
    submission = predictor.run_pipeline()