import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# App title and description
st.set_page_config(page_title="Wallet Trust Score System", layout="wide")
st.title("🌟 AI-Powered Wallet Trust Score System")
st.markdown("""
This tool calculates **Trust Scores** for blockchain wallets to help you identify potential risks.  
Simply upload your transaction data, and we'll handle the rest!  
""")

# Sidebar for upload and instructions
st.sidebar.title("📄 Upload Your File")
st.sidebar.info("""
**Required CSV Columns:**  
- `wallet_id`: Unique identifier for the wallet.  
- `timestamp`: Date and time of the transaction.  
- `transaction_amount`: Value of the transaction.  
- `counterparty_wallet`: Wallet involved in the transaction.  
- `flagged`: (Optional) 1 for flagged (fraudulent), 0 for not flagged.  
""")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file here", type="csv")

# Process uploaded file
if uploaded_file:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet', 'flagged']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.success("File uploaded successfully!")
            
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
            st.header("📊 Trust Score Results")
            st.dataframe(features[['wallet_id', 'trust_score']])

            # Allow download of results
            csv = features.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='wallet_trust_scores.csv',
                mime='text/csv',
            )

            # Visualization
            st.header("📈 Trust Score Visualization")
            fig = px.bar(
                features, 
                x='wallet_id', 
                y='trust_score', 
                color='trust_score',
                color_continuous_scale='RdYlGn',
                title="Wallet Trust Scores",
                labels={'wallet_id': "Wallet ID", 'trust_score': "Trust Score"}
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a CSV file to get started!")

# Footer
st.markdown("""
---
Developed with ❤️ using **Streamlit**.  
""")
