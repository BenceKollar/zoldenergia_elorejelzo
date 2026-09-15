iimport streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import numpy as np

DB_PATH = Path(__file__).parent / "energy_data.db"

st.set_page_config(page_title="Megújuló energiaforrások", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#121212;color:#fff;}
.metric-card {border-radius:15px;padding:30px 20px;text-align:center;
box-shadow:0 8px 16px rgba(0,0,0,.5);color:white!important;margin-bottom:25px;}
.wind-card {background:linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.7)),
url('https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1000&q=80') center/cover;}
.solar-card {background:linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.7)),
url('https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1000&q=80') center/cover;}
.metric-title {font-size:1.3rem;font-weight:500;margin-bottom:10px;text-shadow:2px 2px 4px #000;}
.metric-value {font-size:3.5rem;font-weight:800;text-shadow:2px 2px 6px #000;margin:0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover, "
        "wind_speed_10m, shortwave_radiation FROM power_production", conn
    )
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cols = ["temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation"]
    mask = df["timestamp"].dt.minute != 0
    df.loc[mask, cols] = np.nan
    df[cols] = df[cols].ffill()
    return df


def train_models(data):
    clean = data.dropna(subset=["solar_mw", "wind_mw"]).copy()
    if len(clean) < 2:
        return None, None

    solar = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42)
    wind = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42)

    solar.fit(clean[["temperature_2m", "shortwave_radiation"]].fillna(0), clean["solar_mw"])
    wind.fit(clean[["wind_speed_10m"]].fillna(0), clean["wind_mw"])
    return solar, wind


@st.fragment(run_every="15m")
def render_app():
    # 15 percenként újra lefut az aktív oldal.
    # A cache 10 perc után lejár, ezért az új SQLite adat is beolvasódik.
    load_data.clear()
    df = load_data()

    st.title("⚡ Megújuló Energia Monitor & AI Előrejelző")
    st.write("Valós idejű hálózati adatok és AI jövőbeli becslések.")

    if df.empty:
        st.warning("Még nincsenek adatok az adatbázisban!")
        return

    solar_model, wind_model = train_models(df)
    if solar_model is None:
        st.warning("Még nincs elég valós adat az AI modellek betanításához.")
        return

    df["ai_solar_mw"] = solar_model.predict(
        df[["temperature_2m", "shortwave_radiation"]].fillna(0)
    )
    df["ai_wind_mw"] = wind_model.predict(
        df[["wind_speed_10m"]].fillna(0)
    )

    real_df = df.dropna(subset=["solar_mw", "wind_mw"])
    if real_df.empty:
        st.warning("Még nincsenek valós energiaadatok.")
        return

    latest = real_df.iloc[-1]
    last_real_time = real_df["timestamp"].max()

    st.markdown(
        f"<div style='color:#aaa;margin-bottom:20px;'>"
        f"Utolsó rögzített valós mérés: <b>{latest['timestamp']}</b></div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div class='metric-card wind-card'>"
            f"<div class='metric-title'>Szélerőművi Betáplálás (Valós)</div>"
            f"<div class='metric-value'>{latest['wind_mw']:.1f} MW</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-card solar-card'>"
            f"<div class='metric-title'>Naperőművi Betáplálás (Valós)</div>"
            f"<div class='metric-value'>{latest['solar_mw']:.1f} MW</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    time_filter = st.radio(
        "Válaszd ki a megjeleníteni kívánt időszakot:",
        ["Utolsó 24 óra (1 nap) + Jövő", "Utolsó 1 hét + Jövő",
         "Utolsó 1 hónap + Jövő", "Összes adat"],
        horizontal=True
    )

    if time_filter == "Utolsó 24 óra (1 nap) + Jövő":
        filtered = df[df["timestamp"] >= last_real_time - pd.Timedelta(days=1)]
    elif time_filter == "Utolsó 1 hét + Jövő":
        filtered = df[df["timestamp"] >= last_real_time - pd.Timedelta(days=7)]
    elif time_filter == "Utolsó 1 hónap + Jövő":
        filtered = df[df["timestamp"] >= last_real_time - pd.Timedelta(days=30)]
    else:
        filtered = df

    tab_solar, tab_wind = st.tabs(["Naperőművek", "Szélerőművek"])

    with tab_solar:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["solar_mw"],
            mode="lines", name="Valós Termelés (MW)",
            line=dict(color="#f39c12", width=3), connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["ai_solar_mw"],
            mode="lines", name="AI Predikció (MW)",
            line=dict(color="#3498db", width=2, dash="dash")
        ))
        fig.add_vline(
            x=last_real_time, line_width=2, line_dash="dash",
            line_color="rgba(255,0,0,.5)",
            annotation_text="MOST (Valós adatok vége)"
        )
        fig.update_layout(
            xaxis_title="Időpont", yaxis_title="Teljesítmény (MW)",
            template="plotly_dark", margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_wind:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["wind_mw"],
            mode="lines", name="Valós Termelés (MW)",
            line=dict(color="#2ecc71", width=3), connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["ai_wind_mw"],
            mode="lines", name="AI Predikció (MW)",
            line=dict(color="#e74c3c", width=2, dash="dash")
        ))
        fig.add_vline(
            x=last_real_time, line_width=2, line_dash="dash",
            line_color="rgba(255,0,0,.5)",
            annotation_text="MOST (Valós adatok vége)"
        )
        fig.update_layout(
            xaxis_title="Időpont", yaxis_title="Teljesítmény (MW)",
            template="plotly_dark", margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)


render_app()
