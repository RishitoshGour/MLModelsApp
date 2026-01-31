# Interactive Classification Models Demonstration

## Overview
This is an interactive Streamlit web application that demonstrates multiple machine learning classification models. It allows you to explore different classifiers, compare their performance, and make predictions on new data.

## Features

### 1. **Multiple Classification Models**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes

### 2. **Interactive Features**
   - **Dataset Selection**: Choose between Iris, Breast Cancer, or custom synthetic data
   - **Model Comparison**: Select multiple models to compare side-by-side
   - **Configurable Parameters**: Adjust test-train split and random state
   - **Real-time Visualization**: Charts and graphs update based on selections

### 3. **Four Main Tabs**

#### Tab 1: Data Overview 📊
- Dataset statistics and information
- Class distribution visualization
- Feature statistics and summary

#### Tab 2: Model Comparison 🎯
- Side-by-side performance metrics (Accuracy, Precision, Recall, F1-Score)
- Comparative visualizations
- Quick model ranking by accuracy

#### Tab 3: Detailed Analysis 📈
- Confusion matrix for selected model
- Classification report with detailed metrics
- ROC curve (for binary classification)
- Performance breakdown by class

#### Tab 4: Predictions 🔮
- Interactive sliders to input feature values
- Real-time predictions from all selected models
- Prediction confidence levels
- Visual prediction summary

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Navigate to the project directory:**
   ```bash
   cd "d:\Study\MTech\ML\Assignment2"
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run classification_models.py
   ```

The application will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Configure Settings (Sidebar)**
   - Select a dataset from the dropdown
   - Choose one or more classification models to compare
   - Adjust the test-train split percentage (10-50%)
   - Set random state for reproducibility

2. **Explore Tabs**
   - **Data Overview**: Understand your dataset characteristics
   - **Model Comparison**: See which models perform best
   - **Detailed Analysis**: Dive deep into a specific model's performance
   - **Predictions**: Test the models with custom inputs

3. **Interpret Results**
   - Use confusion matrix to understand misclassifications
   - Compare metrics across models to select the best one
   - Check ROC curve to evaluate model discrimination ability
   - Test predictions with different feature combinations

## Datasets Included

1. **Iris Dataset**
   - 150 samples, 4 features
   - 3 classes (flower species)
   - Classic multi-class classification

2. **Breast Cancer Dataset**
   - 569 samples, 30 features
   - 2 classes (malignant, benign)
   - Binary classification with real medical data

3. **Custom Synthetic Data**
   - 500 samples, 10 features
   - 2 classes
   - Randomly generated for testing

## Performance Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **Precision**: Among positive predictions, how many were correct
- **Recall**: Among actual positives, how many were identified
- **F1-Score**: Harmonic mean of precision and recall (balance metric)
- **ROC-AUC**: Area under the ROC curve (0 to 1, higher is better)

## Model Descriptions

### Logistic Regression
- Linear classification model
- Fast and interpretable
- Best for linearly separable data

### Decision Tree
- Tree-based model
- Easy to interpret
- Prone to overfitting on complex data

### Random Forest
- Ensemble of decision trees
- Generally robust and accurate
- Handles non-linear relationships well

### Gradient Boosting
- Sequential ensemble method
- Often achieves high accuracy
- Can be computationally expensive

### Support Vector Machine (SVM)
- Finds optimal hyperplane
- Effective in high-dimensional space
- Needs feature scaling (applied automatically)

### K-Nearest Neighbors
- Instance-based learning
- Simple but effective
- Lazy learner (no training phase)

### Naive Bayes
- Probabilistic classifier
- Based on Bayes' theorem
- Assumes feature independence

## Tips for Best Results

1. **For Binary Classification**: Check the ROC curve in the Detailed Analysis tab
2. **For Imbalanced Data**: Pay attention to precision, recall, and F1-score
3. **For Feature Selection**: Run multiple models on the same dataset
4. **For Hyperparameter Tuning**: Use the Custom Synthetic Data to test concepts
5. **For Production Models**: Combine metrics analysis with domain knowledge

## Troubleshooting

**Issue**: Application won't start
- **Solution**: Make sure all packages in requirements.txt are installed

**Issue**: Models not appearing in comparison
- **Solution**: Select at least one model in the sidebar

**Issue**: Slow performance with large datasets
- **Solution**: The app is optimized for the included datasets; for larger data, consider sampling

## Requirements

See `requirements.txt` for the complete list of dependencies:
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Author Notes

This application is designed for educational purposes to demonstrate:
- How different classification algorithms work
- Importance of model comparison
- How to evaluate classifier performance
- Interactive machine learning visualization

Feel free to extend this application by:
- Adding more datasets
- Implementing custom models
- Adding hyperparameter tuning sliders
- Including more evaluation metrics
- Creating a cross-validation analysis tab

## License

This project is open source and free to use for educational purposes.

## Application

This application is defined in `app.py`.
