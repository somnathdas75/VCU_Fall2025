"""
Road Accident Risk Prediction - Streamlit Application
Kaggle Playground Series S5E10

This application allows users to:
1. Upload train.csv and test.csv files
2. Explore the data with visualizations
3. Train a machine learning model
4. Generate predictions
5. Download submission.csv file

Author: Your Name
Date: October 19, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Title and description
st.markdown('<div class="main-header">🚗 Road Accident Risk Prediction</div>', unsafe_allow_html=True)
st.markdown("### Kaggle Playground Series - Season 5, Episode 10")
st.markdown("---")

# Sidebar for file upload and configuration
st.sidebar.header("📁 Upload Data Files")
st.sidebar.markdown("Upload your training and test CSV files to begin analysis.")

# File uploaders
train_file = st.sidebar.file_uploader("Upload train.csv", type=['csv'], key='train')
test_file = st.sidebar.file_uploader("Upload test.csv", type=['csv'], key='test')

# Model selection
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "Ridge Regression", "Ensemble"]
)

# Hyperparameters based on model type
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, 50)
    max_depth = st.sidebar.slider("Max Depth", 5, 30, 15, 5)
elif model_type == "Gradient Boosting":
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 200, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.sidebar.slider("Max Depth", 3, 10, 5, 1)

# Function to load and validate data
@st.cache_data
def load_data(file, is_train=True):
    """
    Load and validate CSV data
    
    Parameters:
    - file: uploaded file object
    - is_train: boolean indicating if this is training data
    
    Returns:
    - DataFrame: loaded data
    """
    try:
        df = pd.read_csv(file)
        
        # Validate required columns
        required_cols = ['id', 'road_type', 'num_lanes', 'curvature', 'speed_limit', 
                        'lighting', 'weather', 'road_signs_present', 'public_road', 
                        'time_of_day', 'holiday', 'school_season', 'num_reported_accidents']
        
        if is_train:
            required_cols.append('accident_risk')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to preprocess data
def preprocess_data(df, is_train=True):
    """
    Preprocess the data by encoding categorical variables
    
    Parameters:
    - df: input DataFrame
    - is_train: boolean indicating if this is training data
    
    Returns:
    - DataFrame: preprocessed data
    - dict: label encoders for categorical columns
    """
    df_processed = df.copy()
    label_encoders = {}
    
    # Categorical columns to encode
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Convert boolean columns to binary
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'TRUE': 1, 'True': 1, True: 1, 
                                                        'FALSE': 0, 'False': 0, False: 0})
        else:
            df_processed[col] = df_processed[col].astype(int)
    
    return df_processed, label_encoders

# Function to prepare features
def prepare_features(df, is_train=True):
    """
    Prepare feature matrix for modeling
    
    Parameters:
    - df: preprocessed DataFrame
    - is_train: boolean indicating if this is training data
    
    Returns:
    - X: feature matrix
    - y: target vector (only for training data)
    """
    # Select feature columns
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

# Function to train model
def train_model(X, y, model_type, **kwargs):
    """
    Train the selected machine learning model
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - model_type: type of model to train
    - kwargs: hyperparameters for the model
    
    Returns:
    - trained model
    - training metrics (RMSE, R2)
    """
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Validation predictions
        rf_pred = rf_model.predict(X_val)
        gb_pred = gb_model.predict(X_val)
        val_pred = (rf_pred + gb_pred) / 2
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        r2 = r2_score(y_val, val_pred)
        
        # Store both models
        model = {'rf': rf_model, 'gb': gb_model}
        
        return model, rmse, r2
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    return model, rmse, r2

# Function to make predictions
def make_predictions(model, X, model_type):
    """
    Make predictions using the trained model
    
    Parameters:
    - model: trained model
    - X: feature matrix
    - model_type: type of model
    
    Returns:
    - predictions array
    """
    if model_type == "Ensemble":
        rf_pred = model['rf'].predict(X)
        gb_pred = model['gb'].predict(X)
        predictions = (rf_pred + gb_pred) / 2
    else:
        predictions = model.predict(X)
    
    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)
    
    return predictions

# Main application logic
if train_file is not None and test_file is not None:
    
    # Load data
    with st.spinner("Loading data..."):
        train_data = load_data(train_file, is_train=True)
        test_data = load_data(test_file, is_train=False)
        
        if train_data is not None and test_data is not None:
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.success("✅ Data loaded successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📈 Exploratory Analysis", 
        "🤖 Model Training", 
        "🎯 Predictions", 
        "📥 Download Results"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Data")
            st.write(f"**Shape:** {train_data.shape[0]} rows × {train_data.shape[1]} columns")
            st.dataframe(train_data.head(10), use_container_width=True)
            
            st.markdown("#### Training Data Statistics")
            st.dataframe(train_data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Test Data")
            st.write(f"**Shape:** {test_data.shape[0]} rows × {test_data.shape[1]} columns")
            st.dataframe(test_data.head(10), use_container_width=True)
            
            st.markdown("#### Test Data Statistics")
            st.dataframe(test_data.describe(), use_container_width=True)
        
        # Data quality check
        st.markdown("#### Data Quality Check")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Data Missing Values**")
            missing_train = train_data.isnull().sum()
            if missing_train.sum() == 0:
                st.success("No missing values found! ✅")
            else:
                st.warning(f"Found {missing_train.sum()} missing values")
                st.dataframe(missing_train[missing_train > 0])
        
        with col2:
            st.markdown("**Test Data Missing Values**")
            missing_test = test_data.isnull().sum()
            if missing_test.sum() == 0:
                st.success("No missing values found! ✅")
            else:
                st.warning(f"Found {missing_test.sum()} missing values")
                st.dataframe(missing_test[missing_test > 0])
    
    # Tab 2: Exploratory Analysis
    with tab2:
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        # Target distribution
        st.markdown("#### Target Variable Distribution")
        fig = px.histogram(
            train_data, 
            x='accident_risk', 
            nbins=50,
            title='Distribution of Accident Risk',
            labels={'accident_risk': 'Accident Risk'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Road type analysis
            st.markdown("#### Risk by Road Type")
            road_type_stats = train_data.groupby('road_type')['accident_risk'].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                road_type_stats,
                x='road_type',
                y='mean',
                text='count',
                title='Average Accident Risk by Road Type',
                labels={'mean': 'Average Risk', 'road_type': 'Road Type'},
                color='mean',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather analysis
            st.markdown("#### Risk by Weather Condition")
            weather_stats = train_data.groupby('weather')['accident_risk'].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                weather_stats,
                x='weather',
                y='mean',
                text='count',
                title='Average Accident Risk by Weather',
                labels={'mean': 'Average Risk', 'weather': 'Weather'},
                color='mean',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Lighting analysis
            st.markdown("#### Risk by Lighting Condition")
            lighting_stats = train_data.groupby('lighting')['accident_risk'].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                lighting_stats,
                x='lighting',
                y='mean',
                text='count',
                title='Average Accident Risk by Lighting',
                labels={'mean': 'Average Risk', 'lighting': 'Lighting'},
                color='mean',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time of day analysis
            st.markdown("#### Risk by Time of Day")
            time_stats = train_data.groupby('time_of_day')['accident_risk'].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                time_stats,
                x='time_of_day',
                y='mean',
                text='count',
                title='Average Accident Risk by Time of Day',
                labels={'mean': 'Average Risk', 'time_of_day': 'Time of Day'},
                color='mean',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("#### Numerical Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Curvature vs Risk
            fig = px.scatter(
                train_data,
                x='curvature',
                y='accident_risk',
                color='road_type',
                title='Curvature vs Accident Risk',
                labels={'curvature': 'Curvature', 'accident_risk': 'Accident Risk'},
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Speed limit vs Risk
            speed_stats = train_data.groupby('speed_limit')['accident_risk'].mean().reset_index()
            fig = px.line(
                speed_stats,
                x='speed_limit',
                y='accident_risk',
                title='Speed Limit vs Average Accident Risk',
                labels={'speed_limit': 'Speed Limit', 'accident_risk': 'Average Risk'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Number of lanes analysis
        st.markdown("#### Risk by Number of Lanes")
        lanes_stats = train_data.groupby('num_lanes')['accident_risk'].agg(['mean', 'count']).reset_index()
        fig = px.bar(
            lanes_stats,
            x='num_lanes',
            y='mean',
            text='count',
            title='Average Accident Risk by Number of Lanes',
            labels={'mean': 'Average Risk', 'num_lanes': 'Number of Lanes'},
            color='mean',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Training
    with tab3:
        st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        **Selected Model:** {model_type}
        
        This section trains the model on your training data and evaluates its performance.
        """)
        
        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                
                # Preprocess data
                train_processed, label_encoders = preprocess_data(train_data, is_train=True)
                
                # Prepare features
                X_train, y_train = prepare_features(train_processed, is_train=True)
                
                # Train model with hyperparameters
                if model_type == "Random Forest":
                    model, rmse, r2 = train_model(
                        X_train, y_train, model_type,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                elif model_type == "Gradient Boosting":
                    model, rmse, r2 = train_model(
                        X_train, y_train, model_type,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth
                    )
                else:
                    model, rmse, r2 = train_model(X_train, y_train, model_type)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.label_encoders = label_encoders
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Validation RMSE",
                        value=f"{rmse:.4f}",
                        help="Root Mean Squared Error on validation set"
                    )
                
                with col2:
                    st.metric(
                        label="Validation R² Score",
                        value=f"{r2:.4f}",
                        help="R-squared score on validation set"
                    )
                
                # Feature importance (if available)
                if model_type in ["Random Forest", "Gradient Boosting"]:
                    st.markdown("#### Feature Importance")
                    
                    if model_type == "Ensemble":
                        # Average importance from both models
                        rf_importance = model['rf'].feature_importances_
                        gb_importance = model['gb'].feature_importances_
                        importance = (rf_importance + gb_importance) / 2
                    else:
                        importance = model.feature_importances_
                    
                    feature_names = X_train.columns
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    st.session_state.feature_importance = importance_df
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show existing model info if available
        if st.session_state.model is not None:
            st.info(f"✅ Model is trained and ready for predictions!")
            
            if st.session_state.feature_importance is not None:
                with st.expander("View Feature Importance"):
                    st.dataframe(st.session_state.feature_importance, use_container_width=True)
    
    # Tab 4: Predictions
    with tab4:
        st.markdown('<div class="sub-header">Generate Predictions</div>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            st.markdown("""
            Generate predictions for the test dataset using the trained model.
            """)
            
            if st.button("🎯 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    
                    # Preprocess test data
                    test_processed, _ = preprocess_data(test_data, is_train=False)
                    
                    # Prepare features
                    X_test = prepare_features(test_processed, is_train=False)
                    
                    # Make predictions
                    predictions = make_predictions(
                        st.session_state.model,
                        X_test,
                        model_type
                    )
                    
                    # Store predictions
                    st.session_state.predictions = predictions
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display prediction statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Risk", f"{predictions.mean():.4f}")
                    
                    with col2:
                        st.metric("Median Risk", f"{np.median(predictions):.4f}")
                    
                    with col3:
                        st.metric("Min Risk", f"{predictions.min():.4f}")
                    
                    with col4:
                        st.metric("Max Risk", f"{predictions.max():.4f}")
                    
                    # Prediction distribution
                    st.markdown("#### Prediction Distribution")
                    fig = px.histogram(
                        x=predictions,
                        nbins=50,
                        title='Distribution of Predicted Accident Risk',
                        labels={'x': 'Predicted Risk'},
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sample predictions
                    st.markdown("#### Sample Predictions")
                    sample_results = pd.DataFrame({
                        'id': test_data['id'].values[:20],
                        'accident_risk': predictions[:20]
                    })
                    st.dataframe(sample_results, use_container_width=True)
    
    # Tab 5: Download Results
    with tab5:
        st.markdown('<div class="sub-header">Download Submission File</div>', unsafe_allow_html=True)
        
        if st.session_state.predictions is None:
            st.warning("⚠️ Please generate predictions first in the 'Predictions' tab.")
        else:
            st.markdown("""
            Your submission file is ready for download! 
            
            The file contains predictions for all test samples in the required format:
            - **id**: Test sample ID
            - **accident_risk**: Predicted accident risk (0-1)
            """)
            
            # Create submission dataframe
            submission_df = pd.DataFrame({
                'id': test_data['id'],
                'accident_risk': st.session_state.predictions
            })
            
            # Display submission preview
            st.markdown("#### Submission File Preview")
            st.dataframe(submission_df.head(20), use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", f"{len(submission_df):,}")
            
            with col2:
                st.metric("Average Risk", f"{submission_df['accident_risk'].mean():.4f}")
            
            with col3:
                st.metric("Risk Std Dev", f"{submission_df['accident_risk'].std():.4f}")
            
            # Convert to CSV
            csv = submission_df.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download submission.csv",
                data=csv,
                file_name="submission.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.success("✅ Submission file is ready! Click the button above to download.")
            
            # Additional visualizations
            st.markdown("#### Prediction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution by percentile
                percentiles = [10, 25, 50, 75, 90]
                percentile_values = [np.percentile(submission_df['accident_risk'], p) for p in percentiles]
                
                percentile_df = pd.DataFrame({
                    'Percentile': [f"{p}th" for p in percentiles],
                    'Risk Value': percentile_values
                })
                
                st.markdown("**Risk Percentiles**")
                st.dataframe(percentile_df, use_container_width=True)
            
            with col2:
                # Risk categories
                low_risk = (submission_df['accident_risk'] < 0.3).sum()
                medium_risk = ((submission_df['accident_risk'] >= 0.3) & 
                              (submission_df['accident_risk'] < 0.6)).sum()
                high_risk = (submission_df['accident_risk'] >= 0.6).sum()
                
                category_df = pd.DataFrame({
                    'Category': ['Low Risk (<0.3)', 'Medium Risk (0.3-0.6)', 'High Risk (≥0.6)'],
                    'Count': [low_risk, medium_risk, high_risk],
                    'Percentage': [
                        f"{low_risk/len(submission_df)*100:.1f}%",
                        f"{medium_risk/len(submission_df)*100:.1f}%",
                        f"{high_risk/len(submission_df)*100:.1f}%"
                    ]
                })
                
                st.markdown("**Risk Categories**")
                st.dataframe(category_df, use_container_width=True)
            
            # Risk category pie chart
            fig = px.pie(
                values=[low_risk, medium_risk, high_risk],
                names=['Low Risk', 'Medium Risk', 'High Risk'],
                title='Distribution of Risk Categories',
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728']
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Instructions when no files are uploaded
    st.info("👆 Please upload both train.csv and test.csv files using the sidebar to begin.")
    
    st.markdown("""
    ## 📋 Instructions
    
    ### Step 1: Upload Data
    - Upload **train.csv** (contains training data with accident_risk target)
    - Upload **test.csv** (contains test data for predictions)
    
    ### Step 2: Explore Data
    - View dataset overview and statistics
    - Analyze feature distributions and correlations
    - Identify patterns in the data
    
    ### Step 3: Configure Model
    - Select your preferred model type
    - Adjust hyperparameters using the sidebar
    
    ### Step 4: Train Model
    - Train the model on your training data
    - View validation metrics (RMSE, R²)
    - Analyze feature importance
    
    ### Step 5: Generate Predictions
    - Create predictions for the test dataset
    - Review prediction statistics
    
    ### Step 6: Download Results
    - Download submission.csv file
    - Submit to Kaggle competition
    
    ---
    
    ## 📊 Dataset Features
    
    - **id**: Unique identifier
    - **road_type**: Type of road (highway, urban, rural)
    - **num_lanes**: Number of lanes (1-4)
    - **curvature**: Road curvature (0-1)
    - **speed_limit**: Speed limit (25-70 mph)
    - **lighting**: Lighting condition (daylight, dim, night)
    - **weather**: Weather condition (clear, rainy, foggy)
    - **road_signs_present**: Presence of road signs (True/False)
    - **public_road**: Whether it's a public road (True/False)
    - **time_of_day**: Time period (morning, afternoon, evening)
    - **holiday**: Holiday status (True/False)
    - **school_season**: School season status (True/False)
    - **num_reported_accidents**: Number of reported accidents (0-3)
    - **accident_risk**: Target variable - accident risk score (0-1) [Training only]
    
    ---
    
    ## 🎯 Competition Goal
    
    Predict the **accident_risk** for each road segment in the test set. The metric used for evaluation is **Root Mean Squared Error (RMSE)**.
    
    ---
    
    ## 🤖 Available Models
    
    1. **Random Forest**: Ensemble of decision trees with averaging
    2. **Gradient Boosting**: Sequential ensemble with gradient descent optimization
    3. **Ridge Regression**: Linear model with L2 regularization
    4. **Ensemble**: Combination of Random Forest and Gradient Boosting
    
    ---
    
    ## 💡 Tips for Better Predictions
    
    - **Feature Engineering**: Consider interactions between features
    - **Hyperparameter Tuning**: Experiment with different model parameters
    - **Cross-Validation**: Use multiple folds for robust evaluation
    - **Ensemble Methods**: Combine multiple models for better performance
    - **Data Analysis**: Understand feature relationships before modeling
    
    ---
    
    ## 📈 Model Performance Metrics
    
    - **RMSE (Root Mean Squared Error)**: Lower is better, measures prediction accuracy
    - **R² Score**: Closer to 1.0 is better, measures explained variance
    - **Feature Importance**: Shows which features contribute most to predictions
    
    ---
    
    ## 🚀 Getting Started
    
    Ready to begin? Upload your CSV files using the sidebar! 👈
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Road Accident Risk Prediction System</strong></p>
        <p>Kaggle Playground Series S5E10 | Developed with Streamlit & Scikit-learn</p>
        <p>📊 Analyze • 🤖 Train • 🎯 Predict • 📥 Submit</p>
    </div>
    """, unsafe_allow_html=True)