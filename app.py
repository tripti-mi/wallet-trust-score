import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Set page layout
st.set_page_config(page_title="Wallet Risk Profile System", layout="wide")

# App title and description
st.title("🌟 AI-Powered Wallet Risk Profile System")
st.markdown("""
Welcome to the Wallet Risk Profile System!  
This tool helps you identify potential risks in blockchain wallets by calculating a **Risk Profile Score** based on customizable metrics.

### How it works:
1. **Upload Data**: Provide transaction data for analysis.  
2. **Customize Metrics**: Adjust the weights of selected metrics or turn them on/off.  
3. **Thresholds**: Define thresholds for "Safe," "Monitor," and "Investigate."  
4. **Get Results**: View risk categorizations and download the results.

Upload your transaction data to get started!
""")

# Sidebar for file upload
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

# Feature engineering
def create_features(df):
    features = df.groupby('wallet_id').agg({
        'transaction_amount': ['mean', 'count'],
        'counterparty_wallet': pd.Series.nunique,
        'flagged': 'sum'
    })
    features.columns = ['avg_tx_amount', 'tx_count', 'unique_peers', 'flagged_connections']
    return features.reset_index()

# Calculate Risk Profile Score
def calculate_risk_score(features, weights, variables):
    score = 0
    for var in variables:
        if variables[var]:  # Only include variables turned ON
            score += features[var] * weights[var]
    features['risk_profile_score'] = score
    return features

# Categorize wallets based on thresholds
def categorize_wallets(score, safe_threshold, monitor_threshold):
    if score <= safe_threshold:
        return 'Safe'
    elif safe_threshold < score <= monitor_threshold:
        return 'Monitor'
    else:
        return 'Investigate'

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet', 'flagged']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.success("File uploaded successfully!")
            features = create_features(data)

            if len(features) < 10:
                st.warning("Dataset is too small for reliable categorization. Please upload more data (>= 10 rows).")
            else:
                # Sidebar for customization
                st.sidebar.title("⚙️ Customize Metrics")
                st.sidebar.markdown("Adjust weights or toggle variables on/off.")

                # Variables and default weights
                variables = {
                    'avg_tx_amount': st.sidebar.checkbox("Average Transaction Amount", value=True),
                    'tx_count': st.sidebar.checkbox("Transaction Count", value=True),
                    'unique_peers': st.sidebar.checkbox("Unique Counterparties", value=True),
                    'flagged_connections': st.sidebar.checkbox("Flagged Connections", value=True),
                }
                weights = {
                    'avg_tx_amount': st.sidebar.slider("Weight: Average Transaction Amount", 0.0, 1.0, 0.25),
                    'tx_count': st.sidebar.slider("Weight: Transaction Count", 0.0, 1.0, 0.25),
                    'unique_peers': st.sidebar.slider("Weight: Unique Counterparties", 0.0, 1.0, 0.25),
                    'flagged_connections': st.sidebar.slider("Weight: Flagged Connections", 0.0, 1.0, 0.25),
                }

                # Sidebar for thresholds
                st.sidebar.title("⚙️ Set Thresholds")
                safe_threshold = st.sidebar.slider("Safe Threshold", 0.0, 1.0, 0.3)
                monitor_threshold = st.sidebar.slider("Monitor Threshold", 0.0, 1.0, 0.7)

                # Calculate Risk Profile Score
                features = calculate_risk_score(features, weights, variables)

                # Categorize wallets
                features['risk_category'] = features['risk_profile_score'].apply(
                    lambda score: categorize_wallets(score, safe_threshold, monitor_threshold)
                )

                risk_counts = features['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Count']

                # Container for charts
                with st.container(border=True):
                    st.markdown("### 📊 Risk Analysis")

                    # Row for Risk Summary and Wallet Risk Profile
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.subheader("Risk Level Summary")
                        fig_pie = px.pie(
                            risk_counts, values='Count', names='Risk Category',
                            color='Risk Category',
                            color_discrete_map={
                                'Safe': 'green', 'Monitor': 'orange', 'Investigate': 'red'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        st.subheader("Wallet Risk Profiles")
                        fig_bar = px.bar(
                            features, x='wallet_id', y='risk_profile_score',
                            color='risk_category',
                            color_discrete_map={
                                'Safe': 'green', 'Monitor': 'orange', 'Investigate': 'red'
                            },
                            title="Wallet Risk Profile Scores"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                # Download Results
                csv = features.to_csv(index=False)
                st.download_button("Download Results as CSV", csv, "wallet_risk_profiles.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a CSV file to get started!")
