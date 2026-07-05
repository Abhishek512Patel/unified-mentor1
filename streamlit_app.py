"""
Buyer Segmentation & Investment Profiling — Streamlit Dashboard
Parcl Co. Limited / Unified Mentor Data Science Internship

Run with:
    streamlit run streamlit_app.py

Expects clients.csv and properties.csv in the same folder as this script
(or upload them via the sidebar).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Parcl Buyer Segmentation", layout="wide", page_icon="🏢")

RANDOM_STATE = 42
REFERENCE_DATE = pd.Timestamp("2025-12-31")

SEGMENT_COLORS = {
    "Global Investors": "#4C72B0",
    "First-Time Buyers": "#55A868",
    "Corporate Buyers": "#C44E52",
    "Luxury Investors": "#8172B2",
}

# --------------------------------------------------------------------------------
# Data loading & pipeline (cached so filters don't force recomputation)
# --------------------------------------------------------------------------------

def parse_dob(value):
    s = str(value).strip()
    try:
        if "/" in s:
            return pd.to_datetime(s, format="%m/%d/%Y")
        return pd.to_datetime(s, format="%d-%m-%Y")
    except Exception:
        return pd.NaT


@st.cache_data(show_spinner="Loading & cleaning data...")
def load_and_clean(clients_bytes, properties_bytes):
    clients = pd.read_csv(clients_bytes)
    properties = pd.read_csv(properties_bytes)

    clients.columns = clients.columns.str.strip().str.lower()
    properties.columns = properties.columns.str.strip().str.lower()

    clients = clients.drop_duplicates(subset="client_id").reset_index(drop=True)
    properties = properties.drop_duplicates().reset_index(drop=True)

    cat_cols = ["client_type", "gender", "country", "region",
                "acquisition_purpose", "loan_applied", "referral_channel"]
    for col in cat_cols:
        clients[col] = clients[col].astype(str).str.strip().str.title()

    for col in ["unit_category", "listing_status"]:
        properties[col] = properties[col].astype(str).str.strip().str.title()

    clients["date_of_birth"] = clients["date_of_birth"].apply(parse_dob)
    clients["age"] = ((REFERENCE_DATE - clients["date_of_birth"]).dt.days // 365).astype(float)

    properties["transaction_date"] = pd.to_datetime(properties["transaction_date"], format="%m-%d-%Y")
    properties["sale_price"] = properties["sale_price"].replace(r"[\$,]", "", regex=True).astype(float)

    sold = properties[properties["listing_status"] == "Sold"].copy()
    agg = sold.groupby("client_ref").agg(
        num_purchases=("listing_id", "count"),
        total_spend=("sale_price", "sum"),
        avg_price=("sale_price", "mean"),
        avg_floor_area=("floor_area_sqft", "mean"),
        n_apartment=("unit_category", lambda x: (x == "Apartment").sum()),
        n_office=("unit_category", lambda x: (x == "Office").sum()),
        n_towers=("tower_number", "nunique"),
        first_purchase=("transaction_date", "min"),
        last_purchase=("transaction_date", "max"),
    ).reset_index()
    agg["pct_office_units"] = agg["n_office"] / agg["num_purchases"]
    agg["tenure_months"] = ((agg["last_purchase"] - agg["first_purchase"]).dt.days / 30).round(1)

    df = clients.merge(agg, left_on="client_id", right_on="client_ref", how="left")
    df.drop(columns=["client_ref"], inplace=True)
    fill_cols = ["num_purchases", "total_spend", "avg_price", "avg_floor_area",
                 "pct_office_units", "tenure_months"]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df


@st.cache_data(show_spinner="Encoding, scaling & clustering...")
def run_clustering(df, k):
    model_df = df.copy()
    onehot_cols = ["client_type", "acquisition_purpose", "referral_channel", "loan_applied", "gender"]
    model_df = pd.get_dummies(model_df, columns=onehot_cols, drop_first=False)

    for col in ["country", "region"]:
        freq_map = df[col].value_counts(normalize=True)
        model_df[f"{col}_freq"] = df[col].map(freq_map)

    feature_cols = [
        "age", "satisfaction_score", "num_purchases", "total_spend", "avg_price",
        "avg_floor_area", "pct_office_units", "tenure_months", "country_freq", "region_freq",
    ] + [c for c in model_df.columns if c.startswith((
        "client_type_", "acquisition_purpose_", "referral_channel_", "loan_applied_", "gender_"
    ))]

    X = model_df[feature_cols].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    out = df.copy()
    out["cluster"] = clusters
    out["pca_1"], out["pca_2"] = coords[:, 0], coords[:, 1]

    # Auto-label clusters based on profile
    profile = out.groupby("cluster").agg(
        pct_company=("client_type", lambda x: (x == "Company").mean() * 100),
        pct_loan=("loan_applied", lambda x: (x == "Yes").mean() * 100),
        avg_age=("age", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
    )

    labels_map = {}
    for cl, row in profile.iterrows():
        if row["pct_company"] > 50:
            labels_map[cl] = "Corporate Buyers"
        elif row["avg_total_spend"] == profile["avg_total_spend"].max() and row["avg_satisfaction"] >= profile["avg_satisfaction"].median():
            labels_map[cl] = "Luxury Investors"
        elif row["pct_loan"] > 50 and row["avg_age"] < profile["avg_age"].median():
            labels_map[cl] = "First-Time Buyers"
        else:
            labels_map[cl] = "Global Investors"

    if k == 4 and len(set(labels_map.values())) < 4:
        fallback = ["Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"]
        labels_map = {cl: fallback[i] for i, cl in enumerate(profile.index)}

    out["segment_name"] = out["cluster"].map(labels_map)
    return out


# --------------------------------------------------------------------------------
# Sidebar — data source & controls
# --------------------------------------------------------------------------------

st.sidebar.title("🏢 Parcl Buyer Intelligence")
st.sidebar.caption("Machine Learning Buyer Segmentation & Investment Profiling")

st.sidebar.subheader("Data Source")
uploaded_clients = st.sidebar.file_uploader("clients.csv", type="csv")
uploaded_properties = st.sidebar.file_uploader("properties.csv", type="csv")

clients_source = uploaded_clients if uploaded_clients is not None else "clients.csv"
properties_source = uploaded_properties if uploaded_properties is not None else "properties.csv"

try:
    df = load_and_clean(clients_source, properties_source)
except FileNotFoundError:
    st.error("Couldn't find clients.csv / properties.csv next to streamlit_app.py. "
             "Upload them using the sidebar to continue.")
    st.stop()

st.sidebar.subheader("Model Settings")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=8, value=4)

df = run_clustering(df, k)

st.sidebar.subheader("Filters")
countries = st.sidebar.multiselect("Country", sorted(df["country"].unique()))
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()))
purposes = st.sidebar.multiselect("Acquisition Purpose", sorted(df["acquisition_purpose"].unique()))
client_types = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()))

filtered = df.copy()
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if regions:
    filtered = filtered[filtered["region"].isin(regions)]
if purposes:
    filtered = filtered[filtered["acquisition_purpose"].isin(purposes)]
if client_types:
    filtered = filtered[filtered["client_type"].isin(client_types)]

if filtered.empty:
    st.warning("No clients match the current filters. Adjust filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------------

st.title("Buyer Segmentation & Investment Profiling")
st.caption("AI-driven buyer intelligence for Parcl's real estate market — K-Means clustering on client demographics, financing, and transaction behaviour.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Clients (filtered)", f"{len(filtered):,}", f"of {len(df):,} total")
kpi2.metric("Total Transaction Value", f"${filtered['total_spend'].sum():,.0f}")
kpi3.metric("Avg. Satisfaction", f"{filtered['satisfaction_score'].mean():.2f} / 5")
kpi4.metric("Segments Found", f"{filtered['segment_name'].nunique()}")

st.divider()

# --------------------------------------------------------------------------------
# Tabs: Overview | Investor Behaviour | Geographic | Segment Insights
# --------------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segmentation Overview", "💰 Investor Behaviour", "🌍 Geographic Analysis", "🔍 Segment Insights"
])

with tab1:
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader("Cluster Distribution")
        dist = filtered["segment_name"].value_counts().reset_index()
        dist.columns = ["Segment", "Count"]
        fig = px.pie(dist, names="Segment", values="Count", hole=0.45,
                     color="Segment", color_discrete_map=SEGMENT_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Clusters — PCA Projection")
        fig2 = px.scatter(filtered, x="pca_1", y="pca_2", color="segment_name",
                           color_discrete_map=SEGMENT_COLORS,
                           hover_data=["client_id", "country", "total_spend"],
                           labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Segment Sizes")
    st.dataframe(
        filtered.groupby("segment_name").size().reset_index(name="clients").sort_values("clients", ascending=False),
        use_container_width=True, hide_index=True
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Total Spend by Segment")
        spend = filtered.groupby("segment_name")["total_spend"].mean().reset_index()
        fig3 = px.bar(spend, x="segment_name", y="total_spend", color="segment_name",
                      color_discrete_map=SEGMENT_COLORS, labels={"total_spend": "Avg. Total Spend ($)"})
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Financing Behaviour (Loan Applied %)")
        loan = filtered.groupby("segment_name")["loan_applied"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        loan.columns = ["segment_name", "pct_loan"]
        fig4 = px.bar(loan, x="segment_name", y="pct_loan", color="segment_name",
                      color_discrete_map=SEGMENT_COLORS, labels={"pct_loan": "% Applied for Loan"})
        st.plotly_chart(fig4, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Acquisition Purpose by Segment")
        purpose_ct = pd.crosstab(filtered["segment_name"], filtered["acquisition_purpose"], normalize="index") * 100
        fig5 = px.bar(purpose_ct, barmode="stack", labels={"value": "% of Segment"})
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        st.subheader("Avg. Property Price vs Floor Area")
        fig6 = px.scatter(filtered, x="avg_floor_area", y="avg_price", color="segment_name",
                           color_discrete_map=SEGMENT_COLORS, size="num_purchases",
                           hover_data=["client_id"],
                           labels={"avg_floor_area": "Avg. Floor Area (sqft)", "avg_price": "Avg. Price ($)"})
        st.plotly_chart(fig6, use_container_width=True)

with tab3:
    st.subheader("Buyer Segments by Country")
    geo = filtered.groupby(["country", "segment_name"]).size().reset_index(name="count")
    fig7 = px.bar(geo, x="country", y="count", color="segment_name",
                  color_discrete_map=SEGMENT_COLORS, barmode="stack")
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Heatmap: Country vs Segment")
    pivot = geo.pivot(index="country", columns="segment_name", values="count").fillna(0)
    fig8 = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Top Regions")
    top_regions = filtered["region"].value_counts().head(15).reset_index()
    top_regions.columns = ["Region", "Clients"]
    fig9 = px.bar(top_regions, x="Clients", y="Region", orientation="h")
    fig9.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig9, use_container_width=True)

with tab4:
    st.subheader("Descriptive Statistics per Segment")
    summary = filtered.groupby("segment_name").agg(
        clients=("client_id", "count"),
        avg_age=("age", "mean"),
        pct_company=("client_type", lambda x: (x == "Company").mean() * 100),
        pct_investment=("acquisition_purpose", lambda x: (x == "Investment").mean() * 100),
        pct_loan=("loan_applied", lambda x: (x == "Yes").mean() * 100),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_purchases=("num_purchases", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_price=("avg_price", "mean"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Explore Individual Clients")
    st.dataframe(
        filtered[["client_id", "client_type", "country", "region", "age", "acquisition_purpose",
                  "loan_applied", "satisfaction_score", "num_purchases", "total_spend", "segment_name"]]
        .sort_values("total_spend", ascending=False),
        use_container_width=True, hide_index=True
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered segmented data (CSV)", csv, "segmented_clients.csv", "text/csv")

st.divider()
st.caption("Parcl Co. Limited × Unified Mentor — Data Science Internship Project")
