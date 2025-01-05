import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# App title and description
st.set_page_config(page_title="Wallet Trust Score System", layout="wide")

# Apply proper CSS styling
st.markdown("""
<style>
/* Reset all default padding/margin */
body, .block-container {
    margin: 0 auto;
    padding: 20px;
}

/* Add a card-like container for charts */
.card-container {
    background-color: #1e1e1e; /* Match dark mode */
    padding: 20px;
    border-radius: 10px;
    border: 2px solid #444444; /* Add border */
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3); /* Subtle shadow */
    margin-bottom: 20px;
}

/* Chart headers */
.chart-header {
    text-align: left;
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("🌟 AI-Powered Wallet Trust Score System")
st.markdown("""
Welcome to the Wallet Trust Score System!  
This tool helps you identify potential risks in blockchain transactions by categorizing wallets into **High Risk**, **Medium Risk**, and **Low Risk**.

### How it works:
- **Trust Score**: Measures the likelihood of suspicious activity for each wallet.  
- **Risk Levels**:  
  - **High Risk**: Wallets with low trust scores that may require immediate investigation.  
  - **Medium Risk**: Wallets with moderate scores that should be monitored.  
  - **Low Risk**: Wallets with high trust scores that appear safe.  

Upload your transaction data, and we’ll handle the rest!
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

# Train model
def train_model(features):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features.drop('wallet_id', axis=1))
    features['trust_score'] = model.decision_function(features.drop('wallet_id', axis=1))
    return features, model

# Categorize risk levels dynamically
def categorize_risk_dynamic(score, lower_threshold, upper_threshold):
    if score < lower_threshold:
        return 'High Risk'
    elif lower_threshold <= score <= upper_threshold:
        return 'Medium Risk'
    else:
        return 'Low Risk'

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet', 'flagged']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.success("File uploaded successfully!")
            features = create_features(data)
            features, model = train_model(features)

            if len(features) < 10:
                st.warning("Dataset is too small for reliable categorization. Please upload more data.")
            else:
                lower_threshold = features['trust_score'].quantile(0.25)
                upper_threshold = features['trust_score'].quantile(0.75)
                features['risk_category'] = features['trust_score'].apply(
                    lambda score: categorize_risk_dynamic(score, lower_threshold, upper_threshold)
                )

                risk_counts = features['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Count']

                # Encapsulate all charts in a card container
                with st.container():
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)

                    # Risk Summary and Trust Score Distribution
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown('<div class="chart-header">Risk Level Summary</div>', unsafe_allow_html=True)
                        fig_pie = px.pie(
                            risk_counts, values='Count', names='Risk Category',
                            color='Risk Category',
                            color_discrete_map={
                                'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        st.markdown('<div class="chart-header">Trust Score Distribution</div>', unsafe_allow_html=True)
                        fig_hist = px.histogram(features, x='trust_score', nbins=20, color_discrete_sequence=["#636EFA"])
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Wallet Trust Scores
                    st.markdown('<div class="chart-header">Wallet Trust Scores by Category</div>', unsafe_allow_html=True)
                    fig_bar = px.bar(
                        features, x='wallet_id', y='trust_score',
                        color='risk_category',
                        color_discrete_map={
                            'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Download Results
                csv = features.to_csv(index=False)
                st.download_button("Download Results as CSV", csv, "wallet_trust_scores.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a CSV file to get started!")
