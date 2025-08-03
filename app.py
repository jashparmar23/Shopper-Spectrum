import streamlit as st
import pandas as pd
import joblib

# ---------- Load Models & Data ----------
# Load KMeans Model
kmeans = joblib.load('Datasets/kmeans_rfm_model.pkl')

# Load RFM clustering data
rfm_df = pd.read_csv('Datasets/rfm_clustered_customers.csv')

# Load Item Similarity Matrix
item_sim = joblib.load('Datasets/item_similarity_df.pkl')

# ---------- App Title ----------
st.set_page_config(page_title="E-Commerce AI Suite", layout="centered")
st.title("🛒 Shopper Spectrum")
st.subheader("🎯 E-Commerce Customer Segmentation & Product Recommendation App")

st.markdown("---")

# ---------- Sidebar Navigation ----------
st.sidebar.title("🔍 Choose Module")
module = st.sidebar.radio("", ["Product Recommendation", "Customer Segmentation"])

# ===========================
# 🔹 MODULE 1: Product Recommender
# ===========================
if module == "Product Recommendation":
    st.header("📦 Product Recommendation System")

    # Step 1: Get all product names
    all_products = sorted(item_sim.index.tolist())

    # Step 2: Search bar
    search_query = st.text_input("🔎 Search for a product (type any keyword):")

    # Step 3: Filter product names based on input
    matched_products = [p for p in all_products if search_query.lower() in p.lower()]

    if matched_products:
        # Step 4: Show filtered dropdown
        product_input = st.selectbox("Select a matching product:", matched_products)

        # Step 5: Recommendation logic
        if st.button("Get Recommendations"):
            top_recs = item_sim[product_input].sort_values(ascending=False)[1:6].index.tolist()
            st.success("✅ Top 5 Similar Products:")
            for i, rec in enumerate(top_recs, 1):
                st.markdown(f"**{i}.** {rec}")
    else:
        if search_query:
            st.error("❌ No matching products found. Try a different keyword.")
        else:
            st.info("ℹ️ Start typing to search for a product...")


# ===========================
# 🔹 MODULE 2: Customer Segmentation
# ===========================
elif module == "Customer Segmentation":
    st.header("👤 Customer Segmentation Predictor")

    recency = st.number_input("Enter Recency (in days):", min_value=0, step=1)
    frequency = st.number_input("Enter Frequency (total purchases):", min_value=0, step=1)
    monetary = st.number_input("Enter Monetary (total spend):", min_value=0.0, step=0.1)

    if st.button("Predict Cluster"):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = rfm_df[['Recency', 'Frequency', 'Monetary']]
        scaler.fit(X)

        input_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_scaled)[0]

        # Cluster label mapping from your insights
        cluster_labels = {
            0: "Regular",
            1: "At-Risk",
            2: "High-Value",
            3: "Loyal"
        }

        cluster_descriptions = {
            "Regular": "Steady buyers with moderate frequency and spending.",
            "At-Risk": "Haven’t purchased in a while and spend less.",
            "High-Value": "Very frequent and recent buyers with high spending.",
            "Loyal": "Consistent and engaged customers with strong purchase history."
        }

        label = cluster_labels.get(cluster, "Unknown")
        st.success(f"🧠 Predicted Customer Segment: **{label}**")
        st.info(cluster_descriptions.get(label, ""))

    # Expandable cluster averages reference
    with st.expander("📊 View RFM Cluster Averages"):
        cluster_averages = {
            "High-Value": {"Recency": 7.38, "Frequency": 82.54, "Monetary": 127338.31},
            "At-Risk": {"Recency": 248.08, "Frequency": 1.55, "Monetary": 480.62},
            "Regular": {"Recency": 43.70, "Frequency": 3.68, "Monetary": 1359.05},
            "Loyal": {"Recency": 15.50, "Frequency": 22.33, "Monetary": 12709.09}
        }

        df_avg = pd.DataFrame(cluster_averages).T.reset_index()
        df_avg.columns = ["Customer Segment", "Recency (↓ better)", "Frequency (↑ better)", "Monetary (↑ better)"]
        st.dataframe(df_avg.style.highlight_max(axis=0, color="lightgreen"))
