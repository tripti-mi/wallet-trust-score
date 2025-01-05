import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# App title and description
st.set_page_config(page_title="Wallet Trust Score System", layout="wide")
st.markdown("""
<style>
/* Add border and background for the entire chart container */
.dashboard-container {
    background-color: #f5f5f5; /* Subtle gray */
    padding: 20px;
    margin: 0 auto; /* Center align the container */
    border-radius: 10px;
    border: 2px solid #dddddd; /* Add a border */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
}

/* Align chart headers to the left */
.chart-header {
    text-align: left;
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: var(--text-color); /* Dynamically adapts to light and dark modes */
}

/* Remove unnecessary white lines or padding */
body {
    margin: 0;
    padding: 0;
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

# Feature engineering
def create_features(df):
    """Generate wallet-level features from transaction data."""
    features = df.groupby('wallet_id').agg({
        'transaction_amount': ['mean', 'count'],
        'counterparty_wallet': pd.Series.nunique,
        'flagged': 'sum'
    })
    features.columns = ['avg_tx_amount', 'tx_count', 'unique_peers', 'flagged_connections']
    return features.reset_index()

# Train the model
def train_model(features):
    """Train Isolation Forest and calculate trust scores."""
    model = IsolationForest(contamination=0.1, random_state=42)  # Adjusted contamination
    model.fit(features.drop('wallet_id', axis=1))
    features['trust_score'] = model.decision_function(features.drop('wallet_id', axis=1))
    return features, model

# Categorize risk levels dynamically based on percentiles
def categorize_risk_dynamic(score, lower_threshold, upper_threshold):
    """Categorize wallets into High, Medium, and Low Risk based on dynamic thresholds."""
    if score < lower_threshold:
        return 'High Risk'
    elif lower_threshold <= score <= upper_threshold:
        return 'Medium Risk'
    else:
        return 'Low Risk'

# Updated pipeline
if uploaded_file:
    try:
        # Load and validate data
        data = pd.read_csv(uploaded_file)
        required_columns = ['wallet_id', 'timestamp', 'transaction_amount', 'counterparty_wallet', 'flagged']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Your CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            st.success("File uploaded successfully!")
            
            # Generate features and calculate trust scores
            features = create_features(data)
            features, model = train_model(features)

            # Check dataset size
            if len(features) < 10:
                st.warning("Dataset is too small for reliable percentile-based categorization. Please upload more data.")
            else:
                # Calculate percentiles for dynamic thresholds
                lower_threshold = features['trust_score'].quantile(0.25)  # 25th percentile
                upper_threshold = features['trust_score'].quantile(0.75)  # 75th percentile
                
                # Categorize wallets by dynamic risk levels
                features['risk_category'] = features['trust_score'].apply(
                    lambda score: categorize_risk_dynamic(score, lower_threshold, upper_threshold)
                )

                # Calculate the risk category summary
                risk_counts = features['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Count']

                # Encapsulate all charts in a single container
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

                # Risk Level Summary and Trust Score Distribution in one row (1:3)
                col1, col2 = st.columns([1, 3])  # Create two columns with a 1:3 ratio

                # Risk Level Summary
                with col1:
                    st.markdown('<div class="chart-header">Risk Level Summary</div>', unsafe_allow_html=True)
                    fig_pie = px.pie(
                        risk_counts,
                        values='Count',
                        names='Risk Category',
                        title="",
                        color='Risk Category',
                        color_discrete_map={
                            'High Risk': 'red',
                            'Medium Risk': 'orange',
                            'Low Risk': 'green'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Trust Score Distribution
                with col2:
                    st.markdown('<div class="chart-header">Trust Score Distribution</div>', unsafe_allow_html=True)
                    fig_hist = px.histogram(
                        features,
                        x='trust_score',
                        nbins=20,
                        title="",
                        color_discrete_sequence=["#636EFA"]  # Default color
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Bar Chart of Trust Scores
                st.markdown('<div class="chart-header">Wallet Trust Scores by Category</div>', unsafe_allow_html=True)
                fig_bar = px.bar(
                    features,
                    x='wallet_id',
                    y='trust_score',
                    color='risk_category',
                    color_discrete_map={
                        'High Risk': 'red',
                        'Medium Risk': 'orange',
                        'Low Risk': 'green'
                    },
                    title="",
                    labels={'wallet_id': "Wallet ID", 'trust_score': "Trust Score"}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # End the encapsulating container
                st.markdown('</div>', unsafe_allow_html=True)

                # Download Results
                csv = features.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='wallet_trust_scores.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a CSV file to get started!")
