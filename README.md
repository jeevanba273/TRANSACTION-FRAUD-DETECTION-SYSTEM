 # 🔍 Transaction Fraud Detection System

  A machine learning-powered web application
   for fraud detection in financial
  transactions, built with Streamlit and deployed on
  Railway.

  ## 🚀 Live Application

  **🌐 Access the app:** [Transaction Fraud Detection System](https://transaction-fraud-detection-system.up.railway.app/)

  ## 📋 Overview

  This system uses advanced machine learning algorithms to analyze financial transactions and predict fraud
  probability. The model achieved exceptional performance with **99.97% F1-score** using Random Forest algorithm.

  ### ✨ Key Features

  - **Real-time Fraud Detection**: Instant analysis of transaction legitimacy
  - **Probability Scoring**: Detailed fraud probability percentages
  - **Interactive Web Interface**: User-friendly Streamlit dashboard
  - **Feature Engineering**: Advanced data preprocessing and feature creation
  - **High Accuracy**: 99.99% accuracy with minimal false positives
  - **Responsive Design**: Works seamlessly on desktop and mobile

  ## 🎯 Model Performance

  | Metric | Score |
  |--------|--------|
  | **Accuracy** | 99.99% |
  | **Precision** | 100.00% |
  | **Recall** | 99.45% |
  | **F1-Score** | 99.72% |
  | **ROC-AUC** | 99.99% |

  ## 🛠️ Technical Architecture

  ### Machine Learning Pipeline

  1. **Data Preprocessing**
     - Feature engineering and scaling
     - Transaction type encoding
     - Balance ratio calculations

  2. **Model Training**
     - Random Forest Classifier (Best Performer)
     - Comparison with XGBoost, Logistic Regression, Naive Bayes, LightGBM
     - Cross-validation and hyperparameter tuning

  3. **Feature Engineering**
     - `balance_diff_orig`: Origin account balance difference
     - `balance_diff_dest`: Destination account balance difference
     - `amount_to_balance_ratios`: Transaction amount to balance ratios
     - `amount_equals_diff`: Transaction amount validation checks

  ### Technology Stack

  - **Backend**: Python 3.8+
  - **ML Framework**: scikit-learn, joblib
  - **Web Framework**: Streamlit
  - **Data Processing**: pandas, numpy
  - **Deployment**: Railway.app
  - **Model**: Random Forest Classifier

  ## 📊 Input Features

  The system analyzes the following transaction
  parameters:

  | Feature | Description |
  |---------|-------------|
  | **Step** | Time step of transaction (1-500) |
  | **Transaction Type** | PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT |
  | **Amount** | Transaction amount in currency units |        
  | **Origin Old Balance** | Account balance before transaction |
  | **Origin New Balance** | Account balance after transaction |
  | **Destination Old Balance** | Recipient account balance before |
  | **Destination New Balance** | Recipient account balance after |

  ## 🎮 How to Use

  1. **Visit the Application**: [Transaction Fraud Detection System](https://transaction-fraud-detection-system.up.railway.app/)

  2. **Input Transaction Details**:
     - Enter transaction step (time)
     - Select transaction type
     - Input transaction amount
     - Fill in account balance information

  3. **Get Results**:
     - Click "Analyze Transaction"
     - View fraud probability percentages
     - See detailed feature analysis

  ## Sample Analysis

  ### Legitimate Transaction Example
  Transaction Type: PAYMENT
  Amount: $1,000
  Fraud Probability: 0.12%
  Status: ✅ LEGITIMATE

  ### Fraudulent Transaction Example
  Transaction Type: TRANSFER
  Amount: $500,000
  Fraud Probability: 98.7%
  Status: 🚨 FRAUD DETECTED

  ## 🔬 Dataset Information

  - **Total Transactions**: 5.8M+ records
  - **Fraud Rate**: 0.077% (highly imbalanced dataset)
  - **Features**: 12 engineered features
  - **Time Period**: Multi-step transaction simulation
  - **Transaction Types**: 5 categories

  ## 🚀 Local Development

  ### Prerequisites
  ```bash
  Python 3.8+
  pip package manager
  ```

  Installation

  # Clone the repository
  ```bash
  git clone https://github.com/jeevanba273/TRANSACTION-FRAUD-DETECTION-SYSTEM.git
  cd TRANSACTION-FRAUD-DETECTION-SYSTEM
  ```
  # Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
  # Run the application
  ```bash
  streamlit run fraud_detection_app.py
  ```
  Dependencies
  ```bash
  streamlit>=1.45.0
  pandas>=2.0.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  joblib>=1.3.0
  ```

  🏗️ Project Structure
  
  ```bash
  TRANSACTION-FRAUD-DETECTION-SYSTEM/
  ├── fraud_detection_app.py      # Main Streamlit
  application
  ├── best_fraud_model.pkl        # Trained Random Forest      
  model
  ├── Fraud Detection Dataset.csv # Training dataset
  ├── analysis.ipynb             # Model development
  notebook
  ├── model_results.csv          # Performance metrics
  ├── requirements.txt           # Python dependencies
  └── README.md                  # Documentation
  ```

  🎯 Model Selection Process

  1. Data Exploration: Comprehensive EDA on 5.8M transactions
  2. Feature Engineering: Created 12 predictive features       
  3. Model Comparison: Tested 5 different algorithms
  4. Performance Evaluation: Selected Random Forest based on F1-score
  5. Validation: Rigorous testing on holdout dataset

  📊 Performance Metrics

  ## Confusion Matrix (Test Set)
  
  | **Predicted →** <br> **Actual ↓** | **Normal** | **Fraud** |
  |------------------------------------|------------|-----------|
  | **Normal**                         | 1,172,476  | 0         |
  | **Fraud**                          | 5          | 897       |
  
  Key Insights

  - Zero False Positives: No legitimate transactions flagged as fraud
  - High Recall: Catches 99.45% of actual fraud cases
  - Balanced Performance: Excellent across all metrics



  🙏 Acknowledgments

  - Dataset: Financial transaction simulation data (https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data)
  - Frameworks: Streamlit team for the amazing framework       
  - ML Libraries: scikit-learn contributors
  - Deployment: Railway.app for seamless hosting
