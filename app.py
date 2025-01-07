import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Set page layout
st.set_page_config(page_title="RiskProfiler AI", layout="wide")

# App title and description
st.title("🌟 RiskProfiler AI: Wallet Risk Analysis System")
st.markdown("""
Welcome to **RiskProfiler AI**!  
This tool analyzes blockchain wallet data by calculating a **Risk Profile Score** based on customizable metrics.

### How it works:
1. **Upload Data**: Upload your wallet transaction data for analysis.  
2. **Customize Metrics**: Adjust the weights of metrics or toggle them on/off.  
3. **Set Thresholds**: Define thresholds for `Safe`, `Monitor`, and `Investigate`.  
4. **Get Results**: View risk categorizations and download the results.

---

**Formula for Risk Profile Score**:  
\[
\text{Risk Profile Score} = (w_1 \cdot \text{Avg Transaction Amount}) + (w_2 \cdot \text{Transaction Count}) + (w_3 \cdot \text{Unique Counterparties})
\]  

**Thresholds** (Default):  
- `Safe`: Normalized score below **0.3**.  
- `Monitor`: Normalized score between **0.3** and **0.7**.  
- `Investigate`: Normalized score above **0.7**.  
""")

# Sidebar for file upload
st.sidebar.title("📄 Upload Your File")
st.sidebar.info("""
**Required CSV Columns:**  
- `wallet_id`: Unique identifier for the wallet.  
- `timestamp`: Date and time of the transaction.  
- `transaction_amount`: Value of the transaction.  
- `counterparty_wallet`: Wallet involved in the transaction.  

---

**Note:** The app will automatically calculate these metrics from your dataset:  
- **Average Transaction Amount**: Average of `transaction_amount` for each wallet.  
- **Transaction Count**: Count of transactions for each wallet.  
- **Unique Counterparties**: Count of unique `counterparty_wallet` values for each wallet.  
""")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file here", type="csv")

# Sidebar for metric customization
st.sidebar.title("⚙️ Customize Metrics")
metrics = {
    "avg_tx_amount": st.sidebar.slider("Average Transaction Amount", 0.0, 1.0, 0.33),
    "tx_count": st.sidebar.slider("Transaction Count", 0.0, 1.0, 0.33),
    "unique_peers": st.sidebar.slider("Unique Counterparties", 0.0, 1.0, 0.34),
}

# Sidebar for thresholds
st.sidebar.title("🚦 Set Thresholds")
safe_threshold = st.sidebar.slider("Safe Threshold", 0.0, 1.0, 0.3)
monitor_threshold = st.sidebar.slider("Monitor Threshold", safe_threshold, 1.0, 0.7)

# Feature engineering
def create_features(df):
    """Derive the metrics from raw transaction data."""
    features = df.groupby('wallet_id').agg({
        'transaction_amount': ['mean', 'count'],
        'counterparty_wallet': pd.Series.nunique,
    })
    features.columns = ['avg_tx_amount', 'tx_count', 'unique_peers']
    return features.reset_index()

# Normalize scores
def normalize_scores(features, weights):
    scaler = MinMaxScaler()
    features['raw_score'] = (
        weights["avg_tx_amount"] * features["avg_tx_amount"] +
        weights["tx_count"] * features["tx_count"] +
        weights["unique_peers"] * features["unique_peers"]
    )
    features['normalized_score'] = scaler.fit_transform(features[['raw_score']])
    return features

# Categorize risk
def categorize_risk(score):
    if score < safe_threshold:
        return 'Safe'
    elif safe_threshold <= score < monitor_threshold:
        return 'Monitor'
    else:
        return 'Investigate'

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.success("File uploaded successfully!")
            
            # Derive features
            features = create_features(data)
            
            # Normalize and calculate scores
            features = normalize_scores(features, metrics)
            features['risk_category'] = features['normalized_score'].apply(categorize_risk)

            # Risk Summary
            risk_counts = features['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['Risk Category', 'Count']

            # Container for charts
            with st.container():
                st.markdown("#### 📊 Risk Analysis")  # Header for the dashboard

                # Row for Risk Summary and Wallet Risk Profiles
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown("#### Risk Level Summary")
                    fig_pie = px.pie(
                        risk_counts, values='Count', names='Risk Category',
                        color='Risk Category',
                        color_discrete_map={
                            'Safe': 'green', 'Monitor': 'orange', 'Investigate': 'red'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    st.markdown("#### Wallet Risk Profiles")
                    fig_bar = px.bar(
                        features, x='wallet_id', y='normalized_score',
                        color='risk_category',
                        color_discrete_map={
                            'Safe': 'green', 'Monitor': 'orange', 'Investigate': 'red'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Download Results
            csv = features.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "wallet_risk_profiles.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a CSV file to get started!")
