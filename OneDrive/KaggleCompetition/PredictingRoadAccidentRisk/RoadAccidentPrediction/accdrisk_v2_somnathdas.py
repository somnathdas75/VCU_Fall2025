# Road Accident Risk Prediction - Pure Python Application
# Kaggle Playground Series S5E10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RoadAccidentPredictor:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None
        self.feature_importance = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load train and test data from Kaggle path"""
        try:
            print("Loading data from Kaggle path...")
            self.train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
            self.test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
            print("✅ Data loaded successfully!")
            return True
        except FileNotFoundError:
            print("❌ Data files not found at the specified Kaggle path.")
            print("Please ensure the files exist at: /kaggle/input/playground-series-s5e10/")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def validate_data(self):
        """Validate the loaded data for required columns and structure"""
        if self.train_data is None or self.test_data is None:
            return False
        
        required_train_cols = ['id', 'accident_risk']
        required_common_cols = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 
                               'lighting', 'weather', 'road_signs_present', 'public_road', 
                               'time_of_day', 'holiday', 'school_season', 'num_reported_accidents']
        
        train_missing = [col for col in required_train_cols + required_common_cols if col not in self.train_data.columns]
        test_missing = [col for col in required_common_cols + ['id'] if col not in self.test_data.columns]
        
        if train_missing:
            print(f"❌ Missing columns in training data: {train_missing}")
            return False
        
        if test_missing:
            print(f"❌ Missing columns in test data: {test_missing}")
            return False
        
        print("✅ Data validation passed!")
        return True
    
    def preprocess_data(self, df, is_train=True):
        """Preprocess the data by encoding categorical variables"""
        df_processed = df.copy()
        
        # Categorical columns to encode
        categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
        
        # Encode categorical variables
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    df_processed[col + '_encoded'] = le.transform(df_processed[col])
        
        # Convert boolean columns to binary
        bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        for col in bool_cols:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].map({'TRUE': 1, 'True': 1, True: 1, 
                                                            'FALSE': 0, 'False': 0, False: 0})
            else:
                df_processed[col] = df_processed[col].astype(int)
        
        return df_processed
    
    def prepare_features(self, df, is_train=True):
        """Prepare feature matrix for modeling"""
        feature_cols = [
            'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
            'road_type_encoded', 'lighting_encoded', 'weather_encoded', 'time_of_day_encoded',
            'road_signs_present', 'public_road', 'holiday', 'school_season'
        ]
        
        X = df[feature_cols]
        
        if is_train:
            y = df['accident_risk']
            return X, y
        else:
            return X
    
    def train_model(self, model_type='Random Forest', **kwargs):
        """Train the selected machine learning model"""
        print(f"Training {model_type} model...")
        
        # Preprocess data
        train_processed = self.preprocess_data(self.train_data, is_train=True)
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_processed, is_train=True)
        
        # Split data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Initialize model based on selection
        if model_type == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 15),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
        elif model_type == "Ridge Regression":
            model = Ridge(alpha=1.0, random_state=42)
        else:  # Ensemble
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            
            # Train both models and average predictions
            rf_model.fit(X_train_split, y_train_split)
            gb_model.fit(X_train_split, y_train_split)
            
            # Validation predictions
            rf_pred = rf_model.predict(X_val)
            gb_pred = gb_model.predict(X_val)
            val_pred = (rf_pred + gb_pred) / 2
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            r2 = r2_score(y_val, val_pred)
            
            # Store both models
            model = {'rf': rf_model, 'gb': gb_model}
            
            print("✅ Model trained successfully!")
            print(f"Validation RMSE: {rmse:.4f}")
            print(f"Validation R² Score: {r2:.4f}")
            
            self.model = model
            return model, rmse, r2
        
        # Train model
        model.fit(X_train_split, y_train_split)
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        print("✅ Model trained successfully!")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R² Score: {r2:.4f}")
        
        # Feature importance
        if model_type in ["Random Forest", "Gradient Boosting"]:
            if model_type == "Ensemble":
                rf_importance = model['rf'].feature_importances_
                gb_importance = model['gb'].feature_importances_
                importance = (rf_importance + gb_importance) / 2
            else:
                importance = model.feature_importances_
            
            feature_names = X_train.columns
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\n📊 Feature Importance:")
            print(self.feature_importance.to_string(index=False))
        
        self.model = model
        return model, rmse, r2
    
    def make_predictions(self):
        """Make predictions using the trained model"""
        if self.model is None:
            print("❌ No trained model found. Please train the model first.")
            return None
        
        print("Generating predictions...")
        
        # Preprocess test data
        test_processed = self.preprocess_data(self.test_data, is_train=False)
        
        # Prepare features
        X_test = self.prepare_features(test_processed, is_train=False)
        
        # Make predictions
        if isinstance(self.model, dict):  # Ensemble model
            rf_pred = self.model['rf'].predict(X_test)
            gb_pred = self.model['gb'].predict(X_test)
            predictions = (rf_pred + gb_pred) / 2
        else:
            predictions = self.model.predict(X_test)
        
        # Clip predictions to valid range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        self.predictions = predictions
        
        print("✅ Predictions generated successfully!")
        print(f"Mean Risk: {predictions.mean():.4f}")
        print(f"Median Risk: {np.median(predictions):.4f}")
        print(f"Min Risk: {predictions.min():.4f}")
        print(f"Max Risk: {predictions.max():.4f}")
        
        return predictions
    
    def create_submission(self, output_path="submission.csv"):
        """Create submission file"""
        if self.predictions is None:
            print("❌ No predictions found. Please generate predictions first.")
            return None
        
        submission_df = pd.DataFrame({
            'id': self.test_data['id'],
            'accident_risk': self.predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        print(f"✅ Submission file saved as: {output_path}")
        
        # Print summary statistics
        print(f"\n📈 Submission Summary:")
        print(f"Total predictions: {len(submission_df):,}")
        print(f"Average risk: {submission_df['accident_risk'].mean():.4f}")
        print(f"Risk std dev: {submission_df['accident_risk'].std():.4f}")
        
        # Risk categories
        low_risk = (submission_df['accident_risk'] < 0.3).sum()
        medium_risk = ((submission_df['accident_risk'] >= 0.3) & 
                      (submission_df['accident_risk'] < 0.6)).sum()
        high_risk = (submission_df['accident_risk'] >= 0.6).sum()
        
        print(f"\n🎯 Risk Categories:")
        print(f"Low Risk (<0.3): {low_risk} samples ({low_risk/len(submission_df)*100:.1f}%)")
        print(f"Medium Risk (0.3-0.6): {medium_risk} samples ({medium_risk/len(submission_df)*100:.1f}%)")
        print(f"High Risk (≥0.6): {high_risk} samples ({high_risk/len(submission_df)*100:.1f}%)")
        
        return submission_df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        if self.train_data is None:
            print("❌ No training data loaded.")
            return
        
        print("\n📊 Performing Exploratory Data Analysis...")
        
        # Basic statistics
        print(f"\n📈 Training Data Shape: {self.train_data.shape}")
        print(f"📈 Test Data Shape: {self.test_data.shape}")
        
        # Target distribution
        print(f"\n🎯 Target Variable (accident_risk) Statistics:")
        print(self.train_data['accident_risk'].describe())
        
        # Feature analysis
        categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
        
        for col in categorical_cols:
            print(f"\n📊 {col.upper()} Analysis:")
            stats = self.train_data.groupby(col)['accident_risk'].agg(['mean', 'count']).round(4)
            print(stats)
        
        # Numerical features correlation with target
        numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
        correlations = self.train_data[numerical_cols + ['accident_risk']].corr()['accident_risk'].drop('accident_risk')
        
        print(f"\n📈 Correlation with Target:")
        for feature, corr in correlations.items():
            print(f"{feature}: {corr:.4f}")
    
    def plot_analysis(self, save_plots=True):
        """Create visualization plots"""
        if self.train_data is None:
            print("❌ No training data loaded.")
            return
        
        print("\n📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Road Accident Risk Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        axes[0, 0].hist(self.train_data['accident_risk'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Accident Risk')
        axes[0, 0].set_xlabel('Accident Risk')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Road type analysis
        road_stats = self.train_data.groupby('road_type')['accident_risk'].mean().sort_values(ascending=False)
        axes[0, 1].bar(road_stats.index, road_stats.values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Accident Risk by Road Type')
        axes[0, 1].set_xlabel('Road Type')
        axes[0, 1].set_ylabel('Average Risk')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Weather analysis
        weather_stats = self.train_data.groupby('weather')['accident_risk'].mean().sort_values(ascending=False)
        axes[1, 0].bar(weather_stats.index, weather_stats.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average Accident Risk by Weather')
        axes[1, 0].set_xlabel('Weather Condition')
        axes[1, 0].set_ylabel('Average Risk')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Speed limit vs risk
        speed_stats = self.train_data.groupby('speed_limit')['accident_risk'].mean()
        axes[1, 1].plot(speed_stats.index, speed_stats.values, marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Speed Limit vs Average Accident Risk')
        axes[1, 1].set_xlabel('Speed Limit')
        axes[1, 1].set_ylabel('Average Risk')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('accident_risk_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Analysis plot saved as 'accident_risk_analysis.png'")
        
        plt.show()
        
        # Feature importance plot if available
        if self.feature_importance is not None:
            plt.figure(figsize=(10, 6))
            top_features = self.feature_importance.head(10)
            plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                print("✅ Feature importance plot saved as 'feature_importance.png'")
            
            plt.show()

def main():
    """Main function to run the Road Accident Risk Prediction pipeline"""
    print("=" * 70)
    print("🚗 ROAD ACCIDENT RISK PREDICTION SYSTEM")
    print("Kaggle Playground Series - Season 5, Episode 10")
    print("=" * 70)
    
    # Initialize predictor
    predictor = RoadAccidentPredictor()
    
    # Load data
    if not predictor.load_data():
        return
    
    # Validate data
    if not predictor.validate_data():
        return
    
    # Perform exploratory analysis
    predictor.exploratory_analysis()
    
    # Create visualizations
    predictor.plot_analysis(save_plots=True)
    
    # Model configuration
    model_type = "Random Forest"  # Change to "Gradient Boosting", "Ridge Regression", or "Ensemble"
    
    # Train model
    print(f"\n🤖 Training {model_type} model...")
    predictor.train_model(
        model_type=model_type,
        n_estimators=200,
        max_depth=15
    )
    
    # Generate predictions
    print(f"\n🎯 Generating predictions...")
    predictions = predictor.make_predictions()
    
    if predictions is not None:
        # Create submission file
        print(f"\n📥 Creating submission file...")
        submission = predictor.create_submission("submission.csv")
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📋 Next steps:")
        print("1. Check 'submission.csv' for your predictions")
        print("2. Submit to Kaggle competition")
        print("3. Check 'accident_risk_analysis.png' and 'feature_importance.png' for insights")
        print("=" * 70)

if __name__ == "__main__":
    main()