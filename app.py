import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Set up the app title and description
st.title("AI-Powered Wallet Trust Score System")
st.write("""
Upload your blockchain transaction CSV file to calculate trust scores for wallets.  
Ensure your CSV contains the following columns:
- **wallet_id**: Unique identifier for the wallet.  
- **timestamp**: Date and time of the transaction.  
- **transaction_amount**: Value of the transaction.  
- **counterparty_wallet**: Wallet involved in the transaction.  
- **flagged**: (Optional) Whether the transaction is known to be fraudulent (1 for yes, 0 for no).
""")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load the uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
        
        # Validate required columns
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet', 'flagged']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.write("Processing your data...")

            # Feature engineering
            def create_features(df):
                features = df.groupby('wallet_id').agg({
                    'transaction_amount': ['mean', 'count'],
                    'counterparty_wallet': pd.Series.nunique,
                    'flagged': 'sum'
                })
                features.columns = ['avg_tx_amount', 'tx_count', 'unique_peers', 'flagged_connections']
                return features.reset_index()

            features = create_features(data)

            # Train the model
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(features.drop('wallet_id', axis=1))

            # Calculate trust scores
            features['trust_score'] = model.decision_function(features.drop('wallet_id', axis=1))

            # Display results
            st.subheader("Trust Score Results")
            st.write(features[['wallet_id', 'trust_score']])

            # Visualization
            st.subheader("Trust Score Visualization")
            fig, ax = plt.subplots()
            ax.bar(features['wallet_id'], features['trust_score'], color='skyblue')
            ax.set_xlabel('Wallet ID')
            ax.set_ylabel('Trust Score')
            ax.set_title('Wallet Trust Scores')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

# Footer
st.write("Developed with ❤️ using Streamlit.")
