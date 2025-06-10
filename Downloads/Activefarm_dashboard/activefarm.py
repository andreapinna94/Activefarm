# ------------------------------------------------
# IMPORT
# ------------------------------------------------
import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv                 #  <-- NEW
from openai import OpenAI
                            #  <-- NEW
# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Active Label Dashboard")
# ------------------------------------------------
# ENV & OPENAI (NEW)
# ------------------------------------------------
# ------------------------------------------------
# LOGO & TITLE
# ------------------------------------------------
logo = Image.open("assets/ActiveLabel_MARCHIO.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, use_container_width=True)
with col2:
    st.title("Active Label Dashboard")

# ------------------------------------------------
# DATA
# ------------------------------------------------
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path, parse_dates=["reading_timestamp"])

data = load_data("italian_shipments_dataset.csv")


# FILTERS (with select-all option)
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Checkbox per selezionare tutti i prodotti
if st.sidebar.checkbox("Select all Products", value=True):
    selected_products = list(data["product"].unique())
else:
    selected_products = st.sidebar.multiselect(
        "Product",
        options=data["product"].unique(),
        default=list(data["product"].unique())
    )

# Checkbox per selezionare tutti gli operatori
if st.sidebar.checkbox("Select all Operators", value=True):
    selected_operators = list(data["operator"].unique())
else:
    selected_operators = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique())
    )

# Checkbox per selezionare tutte le città
if st.sidebar.checkbox("Select all Cities", value=True):
    selected_cities = list(data["city"].unique())
else:
    selected_cities = st.sidebar.multiselect(
        "City",
        options=data["city"].unique(),
        default=list(data["city"].unique())
    )

filtered = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1]))
    & (data["product"].isin(selected_products))
    & (data["operator"].isin(selected_operators))
    & (data["city"].isin(selected_cities))
]
# ------------------------------------------------
# EXECUTIVE SNAPSHOT  (unchanged lines omitted)
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

compliance_pct = filtered["in_range"].mean() * 100 if not filtered.empty else 0
incident_pct = filtered["out_of_range"].mean() * 100 if not filtered.empty else 0
total_shipments = len(filtered)
cost_out_range = (
    filtered.loc[filtered["out_of_range"], "shipment_cost_eur"].sum() if not filtered.empty else 0
)
co2_saved = (
    ((0.05 - 0.01) * len(filtered) * filtered["unit_co2_emitted"].mean())
    if not filtered.empty
    else 0
)

col1.metric("% Compliant Shipments", f"{compliance_pct:.1f}%")
col2.metric("% Shipments with Incidents", f"{incident_pct:.1f}%")
col3.metric("📦 Total Shipments", f"{total_shipments}")
col4.metric("Total Waste Cost (€)", f"{cost_out_range:.2f}")
col5.metric("🌱 CO₂ Saved (kg)", f"{co2_saved:.1f}")

# --- Operational Control ---
# ------------------------------------------------
# 📌 Operational Control
# ------------------------------------------------
st.header("📌 Operational Control")

# Alert Center a tutta pagina
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert from the table below to view further details._")

alert_df = filtered[filtered['out_of_range']].sort_values(
    'reading_timestamp', ascending=False
)

if alert_df.empty:
    st.success("✅ No alerts to show.")
else:
    selection_alert_df = alert_df[["shipment_id", "reading_timestamp", "operator", "product", "severity", "city", "latitude", "longitude"]].copy()
    selection_alert_df.insert(0, "Select", False)

    edited_alert_df = st.data_editor(
        selection_alert_df.drop(columns=["latitude", "longitude"]),
        hide_index=True,
        use_container_width=True,
        height=300,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        key="alert_selector"
    )

# ------------------------------------------------
# SELECTED ALERTS (dettagli shipment sotto la tabella)
# ------------------------------------------------
selected_alerts = edited_alert_df[edited_alert_df["Select"]]

if len(selected_alerts) == 1:
    shipment_id = selected_alerts.iloc[0]["shipment_id"]
    shipment_details = filtered[filtered["shipment_id"] == shipment_id].iloc[0]

    st.markdown("---")
    st.header("📦 Shipment Details")

    # Due colonne affiancate: dettagli e mappa
    details_col, map_col = st.columns([1, 1])

    with details_col:
        status_text = "✅ In Range" if shipment_details['in_range'] else "❌ Out of Range"
        st.markdown(f"""
        **📅 Timestamp:** {shipment_details['reading_timestamp']}  
        **📍 City:** {shipment_details['city']}  
        **🚚 Operator:** {shipment_details['operator']}  
        **📦 Product:** {shipment_details['product']}  
        **⚠️ Severity:** {shipment_details['severity']}  

        ---

        **🌡️ Actual Temperature:** {shipment_details['actual_temperature']:.2f} °C  
        **🔹 Min Threshold:** {shipment_details['threshold_min_temperature']} °C  
        **🔸 Max Threshold:** {shipment_details['threshold_max_temperature']} °C  

        **💰 Shipment Cost:** € {shipment_details['shipment_cost_eur']}  
        **🍃 CO₂ Emitted:** {shipment_details['unit_co2_emitted']} kg  

        **🚦 Status:** {status_text}

        ---

        **🏷️ Label ID:** {shipment_details['label_id']}  
        **🆔 Shipment ID:** {shipment_details['shipment_id']}  
        **🔅 Exposure:** {shipment_details['exposure']:.2f}  
        """)

        if st.button("🚨 Report Issue", key=f"report_{shipment_id}"):
            st.warning("Reporting feature coming soon!")

    with map_col:
        st.subheader("📍 Shipment Location")
        shipment_map_df = pd.DataFrame({
            'latitude': [shipment_details['latitude']],
            'longitude': [shipment_details['longitude']],
            'product': [shipment_details['product']],
            'operator': [shipment_details['operator']],
            'city': [shipment_details['city']],
            'reading_timestamp': [shipment_details['reading_timestamp']]
        })

        shipment_map = px.scatter_mapbox(
            shipment_map_df,
            lat="latitude",
            lon="longitude",
            zoom=10,
            height=400,
            size_max=25,
            size=[25],
            mapbox_style="open-street-map",
            hover_data=["product", "operator", "city", "reading_timestamp"]
        )
        st.plotly_chart(shipment_map, use_container_width=True)

elif len(selected_alerts) > 1:
    st.warning("⚠️ Please select only one shipment at a time.")

# ------------------------------------------------
# MAPPA COMPLETA sotto ai dettagli
# ------------------------------------------------
st.markdown("---")
st.subheader("🗺️ Complete Shipment Map")

filtered['severity'] = filtered['severity'].fillna('None')
filtered.loc[filtered['severity'].str.strip() == '', 'severity'] = 'None'

size_map = {'High': 20, 'Medium': 10, 'None': 6}
filtered['marker_size'] = filtered['severity'].map(size_map)

fig_map = px.scatter_mapbox(
    filtered,
    lat="latitude",
    lon="longitude",
    color="severity",
    color_discrete_map={"High": "red", "Medium": "yellow", "None": "green"},
    size="marker_size",
    hover_data=["product", "operator", "severity"],
    mapbox_style="open-street-map",
    zoom=4,
    height=500
)
st.plotly_chart(fig_map, use_container_width=True)



