import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Megújuló energiaforrások", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #ffffff;
}
.metric-card {
    border-radius: 15px;
    padding: 30px 20px;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.5);
    color: white !important;
    margin-bottom: 25px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: transform 0.3s;
}
.metric-card:hover {
    transform: translateY(-5px);
}
.wind-card {
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url('https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1000&q=80') center/cover;
}
.solar-card {
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url('https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1000&q=80') center/cover;
}
.metric-title {
    font-size: 1.3rem;
    font-weight: 500;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    letter-spacing: 1px;
}
.metric-value {
    font-size: 3.5rem;
    font-weight: 800;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_data():
    conn = sqlite3.connect('energy_data.db')
    df = pd.read_sql_query("SELECT timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation FROM power_production", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    cols = ['temperature_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation']
    mask = df['timestamp'].dt.minute != 0
    df.loc[mask, cols] = np.nan
    df[cols] = df[cols].ffill()
    
    return df

df = load_data()

@st.cache_resource
def train_models(data):
    clean_data = data.dropna(subset=['solar_mw', 'wind_mw']).copy()
    
    X_solar = clean_data[['temperature_2m', 'shortwave_radiation']]
    solar_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42)
    solar_model.fit(X_solar, clean_data['solar_mw'])
    
    X_wind = clean_data[['wind_speed_10m']]
    wind_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42)
    wind_model.fit(X_wind, clean_data['wind_mw'])
    
    return solar_model, wind_model

if not df.empty:
    solar_model, wind_model = train_models(df)
    
    eval_df = df.copy()
    
    X_eval_solar = eval_df[['temperature_2m', 'shortwave_radiation']]
    X_eval_wind = eval_df[['wind_speed_10m']]
    
    df['ai_solar_mw'] = solar_model.predict(X_eval_solar)
    df['ai_wind_mw'] = wind_model.predict(X_eval_wind)
    
    last_real_time = df.dropna(subset=['solar_mw'])['timestamp'].max()

st.title("⚡ Megújuló Energia Monitor & AI Előrejelző")
st.write("Valós idejű hálózati adatok és AI jövőbeli becslések.")

real_df = df.dropna(subset=['solar_mw', 'wind_mw'])

if real_df.empty:
    st.warning("Még nincsenek adatok az adatbázisban! Futtasd le a `data_collector.py` szkriptet.")
else:
    latest_row = real_df.iloc[-1]
    latest_time = latest_row['timestamp']
    latest_solar = latest_row['solar_mw']
    latest_wind = latest_row['wind_mw']

    st.markdown(f"<div style='color: #aaaaaa; margin-bottom: 20px;'>Utolsó rögzített valós mérés: <b>{latest_time}</b></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card wind-card">
            <div class="metric-title">Szélerőművi Betáplálás (Valós)</div>
            <div class="metric-value">{round(latest_wind, 1)} MW</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card solar-card">
            <div class="metric-title">Naperőművi Betáplálás (Valós)</div>
            <div class="metric-value">{round(latest_solar, 1)} MW</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    time_filter = st.radio(
        "Válaszd ki a megjeleníteni kívánt időszakot:",
        ["Utolsó 24 óra (1 nap) + Jövő", "Utolsó 1 hét + Jövő", "Utolsó 1 hónap + Jövő", "Összes adat"],
        horizontal=True
    )

    if time_filter == "Utolsó 24 óra (1 nap) + Jövő":
        filtered_df = df[df['timestamp'] >= (last_real_time - pd.Timedelta(days=1))]
    elif time_filter == "Utolsó 1 hét + Jövő":
        filtered_df = df[df['timestamp'] >= (last_real_time - pd.Timedelta(days=7))]
    elif time_filter == "Utolsó 1 hónap + Jövő":
        filtered_df = df[df['timestamp'] >= (last_real_time - pd.Timedelta(days=30))]
    else:
        filtered_df = df

    sub_tab_solar, sub_tab_wind = st.tabs(["Naperőművek", "Szélerőművek"])

    with sub_tab_solar:
        fig_solar = go.Figure()
        fig_solar.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['solar_mw'], mode='lines', name='Valós Termelés (MW)', line=dict(color='#f39c12', width=3)))
        fig_solar.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['ai_solar_mw'], mode='lines', name='AI Predikció (MW)', line=dict(color='#3498db', width=2, dash='dash')))
        
        fig_solar.add_vline(x=last_real_time, line_width=2, line_dash="dash", line_color="rgba(255,0,0,0.5)", annotation_text="MOST (Valós adatok vége)", annotation_position="top right")
        
        fig_solar.update_layout(xaxis_title='Időpont', yaxis_title='Teljesítmény (MW)', template='plotly_dark', margin=dict(t=30))
        st.plotly_chart(fig_solar, use_container_width=True)

    with sub_tab_wind:
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['wind_mw'], mode='lines', name='Valós Termelés (MW)', line=dict(color='#2ecc71', width=3)))
        fig_wind.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['ai_wind_mw'], mode='lines', name='AI Predikció (MW)', line=dict(color='#e74c3c', width=2, dash='dash')))
        
        fig_wind.add_vline(x=last_real_time, line_width=2, line_dash="dash", line_color="rgba(255,0,0,0.5)", annotation_text="MOST (Valós adatok vége)", annotation_position="top right")
        
        fig_wind.update_layout(xaxis_title='Időpont', yaxis_title='Teljesítmény (MW)', template='plotly_dark', margin=dict(t=30))
        st.plotly_chart(fig_wind, use_container_width=True)
