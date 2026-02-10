## Problem statement
This is an interactive Streamlit web application that demonstrates multiple machine learning classification models. It allows you to explore different classifiers, compare their performance, and make predictions on new data.

## Dataset Description
The Breast Cancer Wisconsin dataset is a binary classification dataset used to predict whether a tumor is:
Malignant (cancerous)
Benign (non-cancerous)

Dataset Name: Breast Cancer (OpenML)
Total Samples: 286
Number of Features: 9
Number of Classes: 2

## Models used
ModelName           Accuracy	Precision	Recall	F1-Score	MCC Score
Logistic Regression	0.7931	0.7943	   0.7931	0.773	   0.4831
Decision Tree	      0.6379	0.6799	   0.6379	0.65	   0.2439
kNN	               0.7414	0.7257	   0.7414	0.7206	0.3409
Naive Bayes	         0.7586	0.7481	   0.7586	0.7493	0.4054
Random Forest	      0.7241	0.7058	   0.7241	0.7062	0.3014
XGBoost	            0.7069	0.7217	   0.7069	0.7125	0.3476
 
 
## Logistic Regression (Observation about model performance):
	Best overall balance. Highest accuracy & recall, strong precision, best MCC.This suggests the dataset has strong linear separability, and logistic regression captures it efficiently. Not optimal.
## Random Forest(Observation about model performance):
	Moderate performance. Lower accuracy than expected for an ensemble model
   Precision slightly better than recall → mild false negatives
   MCC noticeably lower than logistic regression
## Decision Tree(Observation about model performance):
   Weakest performer. Lowest accuracy and F1-score.
   Poor MCC → weak balanced prediction quality.
   Likely overfitting or unstable splits.
   High variance.
## XGBoost(Observation about model performance):
   Moderate performance.
   Good precision/recall symmetry.
   A good alternative when tuned.
## K-Nearest Neighbors(Observation about model performance):	
   Recall is good. Second best alternate.
   Balanced precision.
## Naive Bayes	(Observation about model performance):
   Second runner up on recall side.
   Very strong
   competitive F1-score
   Good accuracy
   High recall → medically favorable
   Features behave sufficiently independently for Naive Bayes to work well.
   
### 1. **Four Main Tabs**

#### Tab 1: Data Overview 📊
- Dataset statistics and information
- Class distribution visualization
- Feature statistics and summary

#### Tab 2: Model Comparison 🎯
- Side-by-side performance metrics (Accuracy, Precision, Recall, F1-Score, MCC Score)
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
  

The application will open in your default web browser at `http://localhost:8501`
 
## Requirements

See `requirements.txt` for the complete list of dependencies:
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
 