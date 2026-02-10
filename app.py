"""
Interactive Streamlit Web Application for Multiple Classification Models
Demonstrates various classification algorithms with interactive features
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Classification Models Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🤖 Classification Models")
st.markdown("""
    This application demonstrates various classification algorithms with interactive
    features. You can select different datasets, models, and hyperparameters to explore
    how different classifiers perform.
""")

# =====================
# Sidebar Configuration
# =====================
st.sidebar.header("⚙️ Configuration")

# Dataset Selection
st.sidebar.subheader("Dataset Selection")
dataset_option = st.sidebar.radio(
    "Choose dataset source:",
    ["Default (Breast Cancer)", "Upload CSV"]
)

# File upload for CSV
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Force target column selection if file is uploaded
    target_col = None
    if uploaded_file is not None:
        st.sidebar.subheader("CSV Configuration")
        
        # Load CSV temporarily to get column names
        try:
            from io import StringIO, BytesIO
            
            # Read bytes and reset position
            file_bytes = uploaded_file.getbuffer().tobytes()
            
            # Decode and clean content
            content = file_bytes.decode('utf-8')
            lines = content.split('\n')
            cleaned_lines = []
            for i, line in enumerate(lines):
                line = line.rstrip()
                if line.endswith(','):
                    line = line[:-1]
                if line.strip():  # Only keep non-empty lines
                    cleaned_lines.append(line)
            
            if not cleaned_lines:
                st.sidebar.error("❌ CSV file is empty or contains no valid data")
            else:
                cleaned_content = '\n'.join(cleaned_lines)
                
                # Try to read CSV
                try:
                    df_temp = pd.read_csv(StringIO(cleaned_content))
                    
                    if len(df_temp.columns) == 0:
                        st.sidebar.error("❌ CSV has no columns")
                    else:
                        # Force user to select target column
                        target_col = st.sidebar.selectbox(
                            "Select target column (REQUIRED):",
                            df_temp.columns,
                            index=len(df_temp.columns) - 1,
                            key="target_column_selector"
                        )
                        st.sidebar.success(f"✓ CSV loaded: {len(df_temp)} rows, {len(df_temp.columns)} columns")
                except pd.errors.ParserError as pe:
                    st.sidebar.error(f"❌ CSV parsing error: {str(pe)}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error reading CSV: {str(e)}")
                    
        except UnicodeDecodeError:
            st.sidebar.error("❌ File encoding error. Please ensure the CSV is UTF-8 encoded.")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing CSV: {str(e)}")
else:
    uploaded_file = None
    target_col = None

# Model Selection
st.sidebar.subheader("Model Selection")
selected_models = st.sidebar.multiselect(
    "Select classification models to compare:",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "K-Nearest Neighbors",
        "Naive Bayes"
    ],
    default=["Logistic Regression", "Random Forest"]
)

# Test-Train Split
st.sidebar.subheader("Data Split")
test_size = st.sidebar.slider("Test data percentage:", 10, 50, 20) / 100
random_state = st.sidebar.slider("Random State:", 1, 100, 42)

# =====================
# Load and Prepare Data
# =====================
@st.cache_data
def load_default_data():
    # Download Breast Cancer dataset directly from OpenML
    data = fetch_openml(
        name='breast-cancer',
        version=1,
        as_frame=True,
        parser='auto'
    )
    df = data.frame
    
    # Get target column (always 'Class' for this dataset)
    target_col = 'Class'
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    
    # Encode target labels to numeric (recurrence-events->1, no-recurrence-events->0)
    y = (y == 'recurrence-events').astype(int)
    
    # Encode all categorical features to numeric
    X_df = X_df.copy()
    for col in X_df.columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    
    X = X_df.values
    feature_names = X_df.columns.tolist()
    target_names = ['No Recurrence', 'Recurrence']
    
    return X, y, feature_names, target_names

def load_custom_csv(uploaded_file, target_col):
    """Load data from uploaded CSV file"""
    try:
        from io import StringIO
        
        # Read bytes to avoid file pointer issues
        file_bytes = uploaded_file.getbuffer().tobytes()
        content = file_bytes.decode('utf-8')
        
        # Fix CSV format issues - Remove trailing commas from ALL lines
        lines = content.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Remove trailing commas and whitespace
            line = line.rstrip()  # Remove trailing whitespace
            if line.endswith(','):
                line = line[:-1]  # Remove trailing comma
            if line.strip():  # Only keep non-empty lines
                cleaned_lines.append(line)
        
        if not cleaned_lines:
            st.error("❌ CSV file is empty or contains no valid data")
            return None
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Parse the cleaned CSV
        try:
            df = pd.read_csv(StringIO(cleaned_content))
        except pd.errors.ParserError as pe:
            st.error(f"❌ CSV parsing error: {str(pe)}")
            return None
        
        if len(df.columns) == 0:
            st.error("❌ CSV has no columns to parse")
            return None
        
        st.success(f"✓ CSV loaded with {len(df)} rows and {len(df.columns)} columns")
        
        if target_col not in df.columns:
            st.error(f"❌ Target column '{target_col}' not found in CSV")
            return None
        
        # Separate target and features
        y = df[target_col].dropna().values
        
        if len(y) == 0:
            st.error("❌ Target column is empty")
            return None
        
        # Get all other columns
        other_cols = [col for col in df.columns if col != target_col]
        
        # Try to identify numeric features
        # Skip ID-like columns (all unique values, usually integers with large numbers)
        numeric_features = []
        for col in other_cols:
            try:
                # Try to convert to numeric
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                
                # Check if most values are numeric (>90% successful conversion)
                conversion_rate = numeric_vals.notna().sum() / len(df)
                
                # Skip if it looks like an ID column (all unique or mostly unique)
                uniqueness = df[col].nunique() / len(df)
                is_id_like = uniqueness > 0.9 and col.lower() in ['id', 'patient_id', 'sample_id', 'index']
                
                if conversion_rate > 0.9 and not is_id_like:
                    numeric_features.append(col)
            except:
                pass
        
        if not numeric_features:
            st.error("❌ No numeric feature columns found")
            st.write(f"Available columns: {other_cols}")
            return None
        
        st.success(f"✓ Found {len(numeric_features)} numeric features")
        
        # Select rows with valid target
        valid_target_idx = df[target_col].notna()
        
        X_df = df.loc[valid_target_idx, numeric_features].copy()
        y = df.loc[valid_target_idx, target_col].values
        
        # Convert to numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        # Remove rows with any missing values
        X_df = X_df.dropna()
        y = y[:len(X_df)]
        
        if len(X_df) == 0:
            st.error("❌ No valid data rows found after cleaning")
            return None
        
        st.success(f"✓ Loaded {len(X_df)} valid samples with {len(numeric_features)} numeric features")
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        target_names = le_target.classes_.tolist()
        
        st.info(f"✓ Target classes: {target_names}")
        
        X = X_df.values.astype(float)
        feature_names = numeric_features
        
        return X, y_encoded, feature_names, target_names
    
    except UnicodeDecodeError:
        st.error("❌ File encoding error. Please ensure the CSV is UTF-8 encoded.")
        return None
    except pd.errors.ParserError as pe:
        st.error(f"❌ CSV parsing error: {str(pe)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Load dataset based on selection
if dataset_option == "Upload CSV":
    if uploaded_file is not None:
        if target_col is None:
            st.error("⚠️ Please select a target column from the sidebar to continue.")
            st.stop()
        data_result = load_custom_csv(uploaded_file, target_col)
        if data_result is None:
            st.stop()
        X, y, feature_names, target_names = data_result
    else:
        st.error("⚠️ Please upload a CSV file to continue.")
        st.stop()
else:
    X, y, feature_names, target_names = load_default_data()
 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# Display Dataset Info
# =====================
st.sidebar.subheader("Dataset Statistics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Samples", len(X))
with col2:
    st.metric("Features", X.shape[1])

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Training Samples", len(X_train))
with col2:
    st.metric("Testing Samples", len(X_test))

# =====================
# Main Content Area
# =====================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "🎯 Model Comparison", "📈 Detailed Analysis", "🔮 Predictions"]
)

# =====================
# TAB 1: Data Overview
# =====================
with tab1:
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = "Breast Cancer (OpenML)" if dataset_option == "Default (Breast Cancer)" else uploaded_file.name
        st.write(f"**Dataset Name:** {dataset_name}")
        st.write(f"**Total Samples:** {len(X)}")
        st.write(f"**Number of Features:** {X.shape[1]}")
        st.write(f"**Number of Classes:** {len(np.unique(y))}")
        
    with col2:
        class_dist = pd.Series(y).value_counts().sort_index()
        st.write("**Class Distribution:**")
        fig, ax = plt.subplots(figsize=(8, 5))
        class_dist.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    df_stats = pd.DataFrame(X, columns=feature_names).describe()
    st.dataframe(df_stats, use_container_width=True)

# =====================
# TAB 2: Model Comparison
# =====================
with tab2:
    st.subheader("Model Performance Comparison")
    
    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        # Dictionary to store models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }
        
        # Train and evaluate models
        results = {}
        trained_models = {}
        
        for model_name in selected_models:
            model = models[model_name]
            
            # Train
            model.fit(X_train_scaled, y_train)
            trained_models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            results[model_name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'MCC Score': mcc,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
        
        # Display comparison table
        comparison_df = pd.DataFrame({
            model: {
                'Accuracy': f"{results[model]['Accuracy']:.4f}",
                'Precision': f"{results[model]['Precision']:.4f}",
                'Recall': f"{results[model]['Recall']:.4f}",
                'F1-Score': f"{results[model]['F1-Score']:.4f}",
                'MCC Score': f"{results[model]['MCC Score']:.4f}"
            }
            for model in selected_models
        }).T
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_data = pd.DataFrame({
                model: results[model]
                for model in selected_models
            }).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_data.plot(kind='bar', ax=ax)
            ax.set_ylabel("Score")
            ax.set_title("Model Metrics Comparison")
            ax.legend(loc='best')
            ax.set_ylim([0, 1.1])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            accuracy_data = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['Accuracy'] for m in results.keys()]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy_data)))
            ax.barh(accuracy_data['Model'], accuracy_data['Accuracy'], color=colors)
            ax.set_xlabel("Accuracy")
            ax.set_title("Model Accuracy Comparison")
            ax.set_xlim([0, 1])
            for i, v in enumerate(accuracy_data['Accuracy']):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
            plt.tight_layout()
            st.pyplot(fig)

# =====================
# TAB 3: Detailed Analysis
# =====================
with tab3:
    st.subheader("Detailed Model Analysis")
    
    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        # Re-train models for detailed analysis
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }
        
        selected_model = st.selectbox("Select a model for detailed analysis:", selected_models)
        
        model = models[selected_model]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Confusion Matrix
        st.subheader(f"Confusion Matrix - {selected_model}")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve (for binary classification)
        if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
            st.subheader("ROC Curve")
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {selected_model}')
            ax.legend(loc="lower right")
            st.pyplot(fig)

# =====================
# TAB 4: Predictions
# =====================
with tab4:
    st.subheader("Make Predictions on New Data")
    
    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        st.write("Enter feature values to make predictions:")
        
        # Create input fields for features
        input_data = {}
        cols = st.columns(min(5, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            col_idx = i % len(cols)
            with cols[col_idx]:
                min_val = X[:, i].min()
                max_val = X[:, i].max()
                mean_val = X[:, i].mean()
                input_data[feature] = st.slider(
                    feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=(max_val - min_val) / 100
                )
        
        # Prepare input for prediction
        input_array = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
        input_array_scaled = scaler.transform(input_array)
        
        # Re-train models for predictions
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }
        
        # Make predictions
        st.subheader("Predictions")
        prediction_results = []
        
        for model_name in selected_models:
            model = models[model_name]
            model.fit(X_train_scaled, y_train)
            
            prediction = model.predict(input_array_scaled)[0]
            prediction_label = target_names[int(prediction)]
            
            prob = None
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_array_scaled)[0]
            
            prediction_results.append({
                'Model': model_name,
                'Prediction': prediction_label,
                'Confidence': f"{max(prob)*100:.2f}%" if prob is not None else "N/A"
            })
        
        results_df = pd.DataFrame(prediction_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Visualize predictions
        if selected_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            predictions = [target_names[int(models[m].predict(input_array_scaled)[0])] for m in selected_models]
            colors = ['green' if p == target_names[0] else 'red' for p in predictions]
            ax.barh(selected_models, [1]*len(selected_models), color=colors, alpha=0.7)
            ax.set_xlim([0, 1.2])
            ax.set_xlabel("Prediction")
            ax.set_title("Model Predictions Overview")
            for i, (model, pred) in enumerate(zip(selected_models, predictions)):
                ax.text(0.5, i, pred, ha='center', va='center', fontweight='bold', color='white')
            st.pyplot(fig)

