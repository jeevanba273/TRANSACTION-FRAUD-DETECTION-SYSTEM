import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_fraud_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Feature engineering function
def engineer_features(data):
    """Apply the same feature engineering as in training"""
    # Create engineered features
    data['balance_diff_orig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
    data['balance_diff_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['amount_to_balance_orig_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
    data['amount_to_balance_dest_ratio'] = data['amount'] / (data['oldbalanceDest'] + 1)
    data['amount_equals_diff'] = (data['amount'] == data['balance_diff_orig']).astype(int)
    
    return data

# Encode transaction type
def encode_transaction_type(transaction_type):
    """Encode transaction type to numeric value"""
    type_mapping = {
        'CASH_IN': 0,
        'CASH_OUT': 1, 
        'DEBIT': 2,
        'PAYMENT': 3,
        'TRANSFER': 4
    }
    return type_mapping.get(transaction_type, 0)

# Main app
def main():
    st.title("🔍 Fraud Detection System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for input method selection
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ("Manual Input", "Upload CSV File")
    )
    
    if input_method == "Manual Input":
        # Manual input section
        st.header("📝 Manual Transaction Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            step = st.number_input("Step (Time)", min_value=1, max_value=500, value=1)
            transaction_type = st.selectbox(
                "Transaction Type", 
                options=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
            )
            amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, format="%.2f")
            
        with col2:
            st.subheader("Account Balances")
            old_balance_org = st.number_input("Origin Old Balance", min_value=0.0, value=10000.0, format="%.2f")
            new_balance_orig = st.number_input("Origin New Balance", min_value=0.0, value=9000.0, format="%.2f")
            old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0, format="%.2f")
            new_balance_dest = st.number_input("Destination New Balance", min_value=0.0, value=1000.0, format="%.2f")
        
        # Predict button
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            # Create input data
            input_data = {
                'step': step,
                'type_encoded': encode_transaction_type(transaction_type),
                'amount': amount,
                'oldbalanceOrg': old_balance_org,
                'newbalanceOrig': new_balance_orig,
                'oldbalanceDest': old_balance_dest,
                'newbalanceDest': new_balance_dest
            }
            
            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Apply feature engineering
            df_input = engineer_features(df_input)
            
            # Select features used in training
            feature_cols = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
                           'amount_to_balance_orig_ratio', 'amount_to_balance_dest_ratio', 'amount_equals_diff']
            
            X_input = df_input[feature_cols]
            
            # Make prediction
            try:
                prediction = model.predict(X_input)[0]
                prediction_proba = model.predict_proba(X_input)[0]
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **FRAUD DETECTED**")
                        st.metric("Prediction", "FRAUD")
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION**")
                        st.metric("Prediction", "LEGITIMATE")
                
                with col2:
                    fraud_probability = prediction_proba[1] * 100
                    st.metric("Fraud Probability", f"{fraud_probability:.2f}%")
                    
                with col3:
                    legitimate_probability = prediction_proba[0] * 100
                    st.metric("Legitimate Probability", f"{legitimate_probability:.2f}%")
                
                # Probability bar chart
                st.subheader("📈 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Category': ['Legitimate', 'Fraud'],
                    'Probability': [legitimate_probability, fraud_probability]
                })
                st.bar_chart(prob_df.set_index('Category'))
                
                # Feature values display
                st.subheader("🔧 Engineered Features")
                feature_info = pd.DataFrame({
                    'Feature': feature_cols,
                    'Value': X_input.iloc[0].values
                })
                st.dataframe(feature_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    else:
        # CSV upload section
        st.header("📁 CSV File Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display first few rows
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check if required columns exist
                required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                               'oldbalanceDest', 'newbalanceDest']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("🔍 Analyze All Transactions", type="primary"):
                        # Encode transaction types
                        df['type_encoded'] = df['type'].apply(encode_transaction_type)
                        
                        # Apply feature engineering
                        df_processed = engineer_features(df.copy())
                        
                        # Select features
                        feature_cols = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                       'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
                                       'amount_to_balance_orig_ratio', 'amount_to_balance_dest_ratio', 'amount_equals_diff']
                        
                        X_batch = df_processed[feature_cols]
                        
                        # Make batch predictions
                        predictions = model.predict(X_batch)
                        prediction_probas = model.predict_proba(X_batch)
                        
                        # Add results to original dataframe
                        df['Prediction'] = ['FRAUD' if p == 1 else 'LEGITIMATE' for p in predictions]
                        df['Fraud_Probability'] = prediction_probas[:, 1] * 100
                        df['Legitimate_Probability'] = prediction_probas[:, 0] * 100
                        
                        # Display results
                        st.markdown("---")
                        st.header("📊 Batch Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_transactions = len(df)
                        fraud_count = sum(predictions)
                        legitimate_count = total_transactions - fraud_count
                        avg_fraud_prob = np.mean(prediction_probas[:, 1]) * 100
                        
                        with col1:
                            st.metric("Total Transactions", total_transactions)
                        with col2:
                            st.metric("Fraud Detected", fraud_count)
                        with col3:
                            st.metric("Legitimate", legitimate_count)
                        with col4:
                            st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.2f}%")
                        
                        # Show transactions with fraud predictions
                        st.subheader("🚨 Detected Fraud Transactions")
                        fraud_transactions = df[df['Prediction'] == 'FRAUD']
                        
                        if len(fraud_transactions) > 0:
                            st.dataframe(fraud_transactions[['step', 'type', 'amount', 'Prediction', 'Fraud_Probability']], 
                                       use_container_width=True)
                        else:
                            st.success("✅ No fraudulent transactions detected!")
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        csv_results = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Results (CSV)",
                            data=csv_results,
                            file_name="fraud_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🛡️ Fraud Detection System | Built with Streamlit and Machine Learning
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()