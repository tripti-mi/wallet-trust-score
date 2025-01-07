import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Set page layout
st.set_page_config(page_title="RiskProfiler AI", layout="wide")

# App title and description
st.title("🛡️ RiskProfiler AI: Wallet Risk Analysis System")
st.markdown("""
Welcome to **RiskProfiler AI**!  
This tool analyzes blockchain wallet data by calculating a **Risk Profile Score** based on customizable metrics.

### How it works:
1. **Upload Data**: Upload your wallet transaction data for analysis.  
2. **Customize Metrics**: Adjust the weights of metrics or toggle them on/off.  
3. **Set Thresholds**: Define thresholds for `Safe`, `Monitor`, and `Investigate`.  
4. **Get Results**: View risk categorizations and download the results.
""")

# Expandable section for formula and behind-the-scenes explanation
with st.expander("📖 Behind the Scenes: Explanation and Formula"):
    st.markdown("""
    ### Formula for Risk Profile Score
    The Risk Profile Score is calculated as:  

    ```
    Risk Profile Score = 
      (Weight 1 × Avg Transaction Amount) +
      (Weight 2 × Transaction Count) +
      (Weight 3 × Unique Counterparties)
    ```

    ### Explanation of Metrics:
    - **Average Transaction Amount**: Measures the typical transaction size of a wallet.  
    - **Transaction Count**: Reflects the activity level of a wallet.  
    - **Unique Counterparties**: Indicates the number of unique wallets interacting with this wallet.  

    ### How It Works:
    - **Unsupervised Learning**: In addition to the rule-based scoring system, we apply clustering (e.g., KMeans) to detect patterns in wallet behavior.
    - **Clustering**: KMeans groups wallets into clusters based on their feature similarity. These clusters can indicate different behavioral patterns.
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

**Note**: The app will automatically calculate these metrics from your dataset:  
- **Average Transaction Amount**: Average of `transaction_amount` for each wallet.  
- **Transaction Count**: Count of transactions for each wallet.  
- **Unique Counterparties**: Count of unique `counterparty_wallet` values for each wallet.  

If no file is uploaded, a sample dataset will be used for demonstration purposes.
""")

# Load sample dataset
@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_transactions.csv")

# Sidebar file uploader
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

# Main workflow
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    data = load_sample_data()
    st.info("Using the sample dataset. Upload your file to replace the sample data.")

# Verify columns in dataset
required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet']
if not all(col in data.columns for col in required_columns):
    st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
else:
    # Derive features
    features = create_features(data)

    # Normalize and calculate scores
    features = normalize_scores(features, metrics)
    features['risk_category'] = features['normalized_score'].apply(categorize_risk)

    # Risk Summary
    risk_counts = features['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']

    # Container for charts
    with st.container(border=True):
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
    
    st.divider()
    
    with st.container(border=True):
        # Unsupervised Learning Section
        st.markdown("### 🔍 Unsupervised Learning: Wallet Clustering")

        # Explanation and Interpretation Section in Expander
        with st.expander("📖 How to Interpret the Graph & Business Insights"):
            st.markdown("""
            ### How to Interpret the Graph:
            - **Axes**:
            - `Avg Transaction Amount` (X-axis): Represents the average size of transactions per wallet.
            - `Transaction Count` (Y-axis): Represents how many transactions each wallet has conducted.
            - `Unique Counterparties` (Z-axis): Represents the number of unique wallets interacting with a specific wallet.

            - **Clusters**:
            - Each cluster (denoted by a different color) represents a group of wallets with similar behavioral patterns.
            - Wallets in the same cluster exhibit similar transaction volumes, activity levels, or diversity of counterparties.

            ### Business Insights:
            - Use this graph to identify unusual clusters that may represent suspicious activity.
            - Clusters with low transaction amounts but high unique counterparties could indicate micro-transactions for fraud.
            - Clusters with high transaction amounts and low counterparties could indicate high-value wallets or corporate accounts.
            """)

        # Apply KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        features['cluster'] = kmeans.fit_predict(features[['avg_tx_amount', 'tx_count', 'unique_peers']])

        # Cluster Insights
        cluster_summary = features.groupby('cluster').agg({
            'avg_tx_amount': ['mean', 'std'],
            'tx_count': ['mean', 'std'],
            'unique_peers': ['mean', 'std'],
        }).reset_index()
        cluster_summary.columns = ['Cluster', 'Avg Tx Amount (Mean)', 'Avg Tx Amount (Std)', 
                                'Tx Count (Mean)', 'Tx Count (Std)', 
                                'Unique Peers (Mean)', 'Unique Peers (Std)']

        # Visualization and Insights in Two Columns
        col1, col2 = st.columns([2, 1])

        with col1:
            fig_cluster = px.scatter_3d(
                features, x='avg_tx_amount', y='tx_count', z='unique_peers', color='cluster',
                title="Wallet Clusters (Unsupervised Learning)",
                labels={
                    'avg_tx_amount': 'Avg Transaction Amount',
                    'tx_count': 'Transaction Count',
                    'unique_peers': 'Unique Counterparties',
                    'cluster': 'Cluster'
                }
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Cluster Insights")
            num_clusters = len(features['cluster'].unique())
            st.markdown(f"**Number of Clusters Found:** {num_clusters}")

            for cluster in range(num_clusters):
                cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]
                avg_tx = cluster_info['Avg Tx Amount (Mean)'].values[0]
                tx_count = cluster_info['Tx Count (Mean)'].values[0]
                unique_peers = cluster_info['Unique Peers (Mean)'].values[0]
                st.markdown(f"""
                **Cluster {cluster}:**
                - **Avg Transaction Amount:** {avg_tx:.2f}
                - **Transaction Count:** {tx_count:.2f}
                - **Unique Counterparties:** {unique_peers:.2f}
                """)

            # Personalized Key Business Insights
            st.markdown("### 🛠️ Personalized Key Business Insights")
            with st.expander("💡 Business Insights for Clustering Results"):
                for cluster in range(num_clusters):
                    cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]
                    avg_tx = cluster_info['Avg Tx Amount (Mean)'].values[0]
                    tx_count = cluster_info['Tx Count (Mean)'].values[0]
                    unique_peers = cluster_info['Unique Peers (Mean)'].values[0]

                    st.markdown(f"""
                    #### Cluster {cluster}:
                    - **Average Transaction Amount**: Indicates that wallets in this cluster typically handle transactions around **{avg_tx:.2f}** in value.
                    - **Transaction Count**: Suggests that wallets in this cluster are conducting approximately **{tx_count:.2f}** transactions on average.
                    - **Unique Counterparties**: Highlights that wallets in this cluster interact with around **{unique_peers:.2f}** unique wallets.

                    **Actionable Insights**:
                    - For clusters with **high average transaction amounts** and **low counterparties**, focus on identifying high-value wallets (e.g., corporate accounts).
                    - For clusters with **low transaction amounts** but **high unique counterparties**, investigate for potential micro-transaction-based fraud.
                    - Clusters with **moderate transaction activity** and **diverse counterparties** may warrant monitoring for emerging risks or suspicious activity.
                    """)


    # Download Results
    csv = features.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "wallet_risk_profiles.csv", "text/csv")
