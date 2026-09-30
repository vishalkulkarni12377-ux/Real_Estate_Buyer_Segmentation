import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("final_buyer_segmentation.csv")

# ==========================================
# TITLE
# ==========================================

st.title("🏠 Real Estate Buyer Segmentation")

st.subheader(
    "Machine Learning Based Buyer Segmentation and Investment Profiling"
)

st.write(
    "This dashboard analyzes real-estate buyers using "
    "K-Means clustering and investment-related features."
)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔎 Dashboard Filters")

country_filter = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique()),
    default=[]
)

region_filter = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].dropna().unique()),
    default=[]
)

purpose_filter = st.sidebar.multiselect(
    "Acquisition Purpose",
    sorted(df["acquisition_purpose"].dropna().unique()),
    default=[]
)

client_type_filter = st.sidebar.multiselect(
    "Client Type",
    sorted(df["client_type"].dropna().unique()),
    default=[]
)

# ==========================================
# APPLY FILTERS
# ==========================================

filtered_df = df.copy()

if country_filter:
    filtered_df = filtered_df[
        filtered_df["country"].isin(country_filter)
    ]

if region_filter:
    filtered_df = filtered_df[
        filtered_df["region"].isin(region_filter)
    ]

if purpose_filter:
    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"].isin(purpose_filter)
    ]

if client_type_filter:
    filtered_df = filtered_df[
        filtered_df["client_type"].isin(client_type_filter)
    ]

# ==========================================
# EMPTY DATA CHECK
# ==========================================

if filtered_df.empty:
    st.warning(
        "No buyers match the selected filters. "
        "Please change the filter selection."
    )
    st.stop()

# ==========================================
# BUYER SEGMENTATION OVERVIEW
# ==========================================

st.header("📊 Buyer Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Buyers",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Total Segments",
        filtered_df["segment"].nunique()
    )

with col3:
    st.metric(
        "Total Investment",
        f"${filtered_df['total_investment'].sum():,.0f}"
    )

with col4:
    st.metric(
        "Average Investment",
        f"${filtered_df['total_investment'].mean():,.0f}"
    )

# ==========================================
# BUYERS IN EACH SEGMENT
# ==========================================

st.subheader("Number of Buyers in Each Segment")

segment_count = (
    filtered_df["segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = [
    "Segment",
    "Buyers"
]

fig_segment_count = px.bar(
    segment_count,
    x="Segment",
    y="Buyers",
    title="Number of Buyers in Each Segment"
)

st.plotly_chart(
    fig_segment_count,
    width="stretch"
)

# ==========================================
# INVESTOR BEHAVIOR DASHBOARD
# ==========================================

st.header("📈 Investor Behavior Dashboard")

# ------------------------------------------
# Average Investment
# ------------------------------------------

avg_investment = (
    filtered_df
    .groupby("segment")["total_investment"]
    .mean()
    .reset_index()
)

fig1 = px.bar(
    avg_investment,
    x="segment",
    y="total_investment",
    title="Average Investment by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "total_investment": "Average Investment"
    }
)

st.plotly_chart(
    fig1,
    width="stretch"
)

# ------------------------------------------
# Average Property Count
# ------------------------------------------

avg_property_count = (
    filtered_df
    .groupby("segment")["property_count"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    avg_property_count,
    x="segment",
    y="property_count",
    title="Average Property Count by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "property_count": "Average Properties"
    }
)

st.plotly_chart(
    fig2,
    width="stretch"
)

# ------------------------------------------
# Loan Behavior
# ------------------------------------------

loan_data = (
    filtered_df
    .groupby(["segment", "loan_applied"])
    .size()
    .reset_index(name="Buyers")
)

fig3 = px.bar(
    loan_data,
    x="segment",
    y="Buyers",
    color="loan_applied",
    barmode="group",
    title="Loan Behavior by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "loan_applied": "Loan Applied"
    }
)

st.plotly_chart(
    fig3,
    width="stretch"
)

# ------------------------------------------
# Acquisition Purpose
# ------------------------------------------

purpose_data = (
    filtered_df
    .groupby(["segment", "acquisition_purpose"])
    .size()
    .reset_index(name="Buyers")
)

fig4 = px.bar(
    purpose_data,
    x="segment",
    y="Buyers",
    color="acquisition_purpose",
    barmode="group",
    title="Acquisition Purpose by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "acquisition_purpose": "Acquisition Purpose"
    }
)

st.plotly_chart(
    fig4,
    width="stretch"
)

# ------------------------------------------
# Client Type
# ------------------------------------------

client_data = (
    filtered_df
    .groupby(["segment", "client_type"])
    .size()
    .reset_index(name="Buyers")
)

fig5 = px.bar(
    client_data,
    x="segment",
    y="Buyers",
    color="client_type",
    barmode="group",
    title="Client Type by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "client_type": "Client Type"
    }
)

st.plotly_chart(
    fig5,
    width="stretch"
)

# ------------------------------------------
# Average Property Price
# ------------------------------------------

avg_price = (
    filtered_df
    .groupby("segment")["average_sale_price"]
    .mean()
    .reset_index()
)

fig6 = px.bar(
    avg_price,
    x="segment",
    y="average_sale_price",
    title="Average Property Price by Buyer Segment",
    labels={
        "segment": "Buyer Segment",
        "average_sale_price": "Average Sale Price"
    }
)

st.plotly_chart(
    fig6,
    width="stretch"
)

# ==========================================
# GEOGRAPHIC BUYER ANALYSIS
# ==========================================

st.header("🌍 Geographic Buyer Analysis")

# ------------------------------------------
# Buyers by Country
# ------------------------------------------

country_data = (
    filtered_df["country"]
    .value_counts()
    .reset_index()
)

country_data.columns = [
    "Country",
    "Buyers"
]

fig7 = px.bar(
    country_data,
    x="Country",
    y="Buyers",
    title="Buyers by Country"
)

st.plotly_chart(
    fig7,
    width="stretch"
)

# ------------------------------------------
# Buyer Segments by Country
# ------------------------------------------

country_segment = (
    filtered_df
    .groupby(["country", "segment"])
    .size()
    .reset_index(name="Buyers")
)

fig8 = px.bar(
    country_segment,
    x="country",
    y="Buyers",
    color="segment",
    barmode="stack",
    title="Buyer Segments by Country",
    labels={
        "country": "Country",
        "segment": "Buyer Segment"
    }
)

st.plotly_chart(
    fig8,
    width="stretch"
)

# ------------------------------------------
# Buyers by Region
# ------------------------------------------

region_data = (
    filtered_df["region"]
    .value_counts()
    .reset_index()
)

region_data.columns = [
    "Region",
    "Buyers"
]

fig9 = px.bar(
    region_data,
    x="Region",
    y="Buyers",
    title="Buyers by Region"
)

st.plotly_chart(
    fig9,
    width="stretch"
)

# ==========================================
# SEGMENT SUMMARY
# ==========================================

st.header("📋 Segment Summary")

segment_summary = (
    filtered_df
    .groupby("segment")
    .agg(
        Buyers=("client_id", "count"),
        Average_Age=("age", "mean"),
        Average_Investment=("total_investment", "mean"),
        Average_Properties=("property_count", "mean"),
        Average_Sale_Price=("average_sale_price", "mean"),
        Average_Satisfaction=("satisfaction_score", "mean")
    )
    .reset_index()
)

segment_summary["Average_Age"] = (
    segment_summary["Average_Age"].round(1)
)

segment_summary["Average_Investment"] = (
    segment_summary["Average_Investment"].round(2)
)

segment_summary["Average_Properties"] = (
    segment_summary["Average_Properties"].round(2)
)

segment_summary["Average_Sale_Price"] = (
    segment_summary["Average_Sale_Price"].round(2)
)

segment_summary["Average_Satisfaction"] = (
    segment_summary["Average_Satisfaction"].round(2)
)

st.dataframe(
    segment_summary,
    width="stretch",
    hide_index=True
)

# ==========================================
# SEGMENT INSIGHTS PANEL
# ==========================================

st.header("🔍 Segment Insights Panel")

segment_options = sorted(
    filtered_df["segment"]
    .dropna()
    .unique()
    .tolist()
)

selected_segment = st.selectbox(
    "Select a Buyer Segment",
    segment_options
)

segment_df = filtered_df[
    filtered_df["segment"] == selected_segment
]

# ------------------------------------------
# Selected Segment Overview
# ------------------------------------------

st.subheader("Selected Segment Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Buyers",
        f"{len(segment_df):,}"
    )

with col2:
    st.metric(
        "Average Age",
        f"{segment_df['age'].mean():.1f}"
    )

with col3:
    st.metric(
        "Average Investment",
        f"${segment_df['total_investment'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Average Properties",
        f"{segment_df['property_count'].mean():.2f}"
    )

# ------------------------------------------
# Segment Characteristics
# ------------------------------------------

st.subheader("Segment Characteristics")

col1, col2 = st.columns(2)

with col1:

    st.write("**Acquisition Purpose**")

    purpose_summary = (
        segment_df["acquisition_purpose"]
        .value_counts()
        .reset_index()
    )

    purpose_summary.columns = [
        "Acquisition Purpose",
        "Buyers"
    ]

    st.dataframe(
        purpose_summary,
        width="stretch",
        hide_index=True
    )

with col2:

    st.write("**Loan Behavior**")

    loan_summary = (
        segment_df["loan_applied"]
        .value_counts()
        .reset_index()
    )

    loan_summary.columns = [
        "Loan Applied",
        "Buyers"
    ]

    st.dataframe(
        loan_summary,
        width="stretch",
        hide_index=True
    )

# ------------------------------------------
# Client Type Distribution
# ------------------------------------------

st.subheader("Client Type Distribution")

client_summary = (
    segment_df["client_type"]
    .value_counts()
    .reset_index()
)

client_summary.columns = [
    "Client Type",
    "Buyers"
]

st.dataframe(
    client_summary,
    width="stretch",
    hide_index=True
)

# ------------------------------------------
# Country Distribution
# ------------------------------------------

st.subheader("Country Distribution")

country_summary = (
    segment_df["country"]
    .value_counts()
    .reset_index()
)

country_summary.columns = [
    "Country",
    "Buyers"
]

st.dataframe(
    country_summary,
    width="stretch",
    hide_index=True
)

# ------------------------------------------
# Investment Characteristics
# ------------------------------------------

st.subheader("Investment Characteristics")

investment_details = pd.DataFrame({
    "Metric": [
        "Average Investment",
        "Average Sale Price",
        "Average Property Area",
        "Average Price per Sqft",
        "Average Satisfaction"
    ],
    "Value": [
        f"${segment_df['total_investment'].mean():,.2f}",
        f"${segment_df['average_sale_price'].mean():,.2f}",
        f"{segment_df['average_area'].mean():,.2f} sqft",
        f"${segment_df['average_price_per_sqft'].mean():,.2f}",
        f"{segment_df['satisfaction_score'].mean():.2f}"
    ]
})

st.dataframe(
    investment_details,
    width="stretch",
    hide_index=True
)

# ==========================================
# FILTERED BUYER DATA
# ==========================================

st.header("📄 Filtered Buyer Data")

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True
)

# ==========================================
# DOWNLOAD FILTERED DATA
# ==========================================

st.subheader("📥 Download Filtered Data")

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Buyer Data",
    data=csv,
    file_name="filtered_buyer_segmentation.csv",
    mime="text/csv"
)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.write(
    "All charts and statistics update automatically "
    "when the dashboard filters are changed."
)