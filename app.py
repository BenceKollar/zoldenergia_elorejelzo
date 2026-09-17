import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import numpy as np
import requests

DB_PATH = Path(__file__).parent / "energy_data.db"

st.set_page_config(page_title="Megújuló energiaforrások", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#121212;color:#fff;}
.metric-card {border-radius:15px;padding:30px 20px;text-align:center;box-shadow:0 8px 16px rgba(0,0,0,.5);color:white!important;margin-bottom:25px;}
.wind-card {background:linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.7)),url('https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1000&q=80') center/cover;}
.solar-card {background:linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.7)),url('https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1000&q=80') center/cover;}
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
    weather_cols = ["temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation"]
    df[weather_cols] = df[weather_cols].ffill()
    return df


def make_model():
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )


def chronological_test(data, target, features):
    clean = data.dropna(subset=[target] + features).copy()
    if len(clean) < 20:
        return None

    split = max(int(len(clean) * 0.8), 1)
    train = clean.iloc[:split]
    test = clean.iloc[split:]
    if test.empty:
        return None

    model = make_model()
    model.fit(train[features].fillna(0), train[target])
    pred = model.predict(test[features].fillna(0))

    real_sum = np.abs(test[target].to_numpy()).sum()
    if real_sum == 0:
        return None

    wape = np.abs(test[target].to_numpy() - pred).sum() / real_sum * 100
    return {
        "accuracy": round(max(0, 100 - wape), 1),
        "wape": round(wape, 1),
        "mae": round(np.abs(test[target].to_numpy() - pred).mean(), 1),
        "test_points": len(test)
    }


def train_models(data):
    real = data.dropna(subset=["solar_mw", "wind_mw"]).copy()
    solar_features = ["temperature_2m", "shortwave_radiation"]
    wind_features = ["wind_speed_10m"]

    solar_train = real.dropna(subset=solar_features)
    wind_train = real.dropna(subset=wind_features)
    if len(solar_train) < 20 or len(wind_train) < 20:
        return None, None

    solar = make_model()
    wind = make_model()
    solar.fit(solar_train[solar_features].fillna(0), solar_train["solar_mw"])
    wind.fit(wind_train[wind_features].fillna(0), wind_train["wind_mw"])
    return solar, wind


def format_forecast_rows(future, limit=8):
    rows = future.head(limit)
    if rows.empty:
        return "Nincs elérhető jövőbeli előrejelzési pont."

    lines = []
    for _, row in rows.iterrows():
        lines.append(
            f"- {row['timestamp']:%Y-%m-%d %H:%M}: "
            f"napenergia {row['ai_solar_mw']:.1f} MW, "
            f"szélenergia {row['ai_wind_mw']:.1f} MW"
        )
    return "\n".join(lines)


def call_gemini(prompt):
    api_key = st.secrets["GEMINI_API_KEY"]
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    payload = {
        "system_instruction": {
            "parts": [{"text": (
                "Te egy energiarendszer-elemző asszisztens vagy. "
                "A kapott számokat tekintsd hiteles alkalmazásadatnak. "
                "Soha ne találj ki számot, százalékot, időpontot vagy mérési értéket. "
                "Ne számolj új százalékos eltérést saját magad. "
                "A jelenlegi valós mérés és a jövőbeli előrejelzés szigorúan különálló fogalom. "
                "A jövőbeli előrejelzést soha ne nevezd jelenlegi tényleges termelésnek. "
                "Előrejelzési hibáról csak akkor beszélj, ha a prompt ugyanarra az időpontra "
                "tényleges és előre jelzett értéket is megad. Ha nincs elég adat, mondd ki."
            )}]
        },
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API {response.status_code}: {response.text}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("A Gemini API nem adott vissza választ.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError("A Gemini válasza üres.")
    return "\n".join(text_parts)


@st.fragment(run_every="15m")
def render_app():
    load_data.clear()
    df = load_data()
    st.title("⚡ Megújuló Energia Monitor & AI Előrejelző")

    if df.empty:
        st.warning("Még nincsenek adatok az adatbázisban!")
        return

    real_df = df.dropna(subset=["solar_mw", "wind_mw"]).copy()
    if real_df.empty:
        st.warning("Még nincsenek valós energiaadatok.")
        return

    latest = real_df.iloc[-1]
    last_real_time = real_df["timestamp"].max()

    solar_model, wind_model = train_models(df)
    if solar_model is None:
        st.warning("Még nincs elég valós adat az AI modellek betanításához.")
        return

    solar_features = ["temperature_2m", "shortwave_radiation"]
    wind_features = ["wind_speed_10m"]
    future_df = df[df["timestamp"] > last_real_time].copy()

    if not future_df.empty:
        future_df["ai_solar_mw"] = solar_model.predict(future_df[solar_features].fillna(0))
        future_df["ai_wind_mw"] = wind_model.predict(future_df[wind_features].fillna(0))

    solar_eval = chronological_test(real_df, "solar_mw", solar_features)
    wind_eval = chronological_test(real_df, "wind_mw", wind_features)

    st.markdown("### 📊 Modell ellenőrzése")
    col1, col2 = st.columns(2)

    with col1:
        if solar_eval:
            st.metric("Napenergia tesztpontosság", f"{solar_eval['accuracy']}%")
            st.caption(
                f"Időalapú teszt: utolsó {solar_eval['test_points']} mérés | "
                f"WAPE: {solar_eval['wape']}% | MAE: {solar_eval['mae']} MW"
            )
        else:
            st.metric("Napenergia tesztpontosság", "Gyűjtés alatt...")

    with col2:
        if wind_eval:
            st.metric("Szélenergia tesztpontosság", f"{wind_eval['accuracy']}%")
            st.caption(
                f"Időalapú teszt: utolsó {wind_eval['test_points']} mérés | "
                f"WAPE: {wind_eval['wape']}% | MAE: {wind_eval['mae']} MW"
            )
        else:
            st.metric("Szélenergia tesztpontosság", "Gyűjtés alatt...")

    st.divider()
    st.write("A folytonos vonal valós mérés, a szaggatott vonal kizárólag jövőbeli AI-előrejelzés.")
    st.markdown(
        f"<div style='color:#aaa;margin-bottom:20px;'>"
        f"Utolsó rögzített valós mérés: <b>{last_real_time}</b></div>",
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
        ["Utolsó 24 óra + Jövő", "Utolsó 1 hét + Jövő", "Utolsó 1 hónap + Jövő", "Összes adat"],
        horizontal=True
    )

    if time_filter == "Utolsó 24 óra + Jövő":
        start_time = last_real_time - pd.Timedelta(days=1)
        filtered_real = real_df[real_df["timestamp"] >= start_time]
        filtered_future = future_df[future_df["timestamp"] >= start_time]
    elif time_filter == "Utolsó 1 hét + Jövő":
        start_time = last_real_time - pd.Timedelta(days=7)
        filtered_real = real_df[real_df["timestamp"] >= start_time]
        filtered_future = future_df[future_df["timestamp"] >= start_time]
    elif time_filter == "Utolsó 1 hónap + Jövő":
        start_time = last_real_time - pd.Timedelta(days=30)
        filtered_real = real_df[real_df["timestamp"] >= start_time]
        filtered_future = future_df[future_df["timestamp"] >= start_time]
    else:
        filtered_real = real_df
        filtered_future = future_df

    tab_solar, tab_wind = st.tabs(["Naperőművek", "Szélerőművek"])

    with tab_solar:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_real["timestamp"], y=filtered_real["solar_mw"],
            mode="lines", name="Valós termelés", line=dict(color="#f39c12", width=3)
        ))
        if not filtered_future.empty:
            fig.add_trace(go.Scatter(
                x=filtered_future["timestamp"], y=filtered_future["ai_solar_mw"],
                mode="lines", name="Jövőbeli AI-előrejelzés",
                line=dict(color="#3498db", width=2, dash="dash")
            ))
        fig.add_vline(
            x=last_real_time, line_width=2, line_dash="dash",
            line_color="rgba(255,0,0,.5)", annotation_text="MOST - itt végződik a valós adat"
        )
        fig.update_layout(xaxis_title="Időpont", yaxis_title="Teljesítmény (MW)", template="plotly_dark", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with tab_wind:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_real["timestamp"], y=filtered_real["wind_mw"],
            mode="lines", name="Valós termelés", line=dict(color="#2ecc71", width=3)
        ))
        if not filtered_future.empty:
            fig.add_trace(go.Scatter(
                x=filtered_future["timestamp"], y=filtered_future["ai_wind_mw"],
                mode="lines", name="Jövőbeli AI-előrejelzés",
                line=dict(color="#e74c3c", width=2, dash="dash")
            ))
        fig.add_vline(
            x=last_real_time, line_width=2, line_dash="dash",
            line_color="rgba(255,0,0,.5)", annotation_text="MOST - itt végződik a valós adat"
        )
        fig.update_layout(xaxis_title="Időpont", yaxis_title="Teljesítmény (MW)", template="plotly_dark", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Hálózat-elemző AI Asszisztens")
    st.write("Kérdezz a jelenlegi állapotról vagy a következő órák előrejelzéséről.")

    if "GEMINI_API_KEY" not in st.secrets:
        st.info("A chatbox használatához állítsd be a GEMINI_API_KEY-t a Streamlit Secrets-ben!")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pl.: Mit vár a modell a következő 6 órában?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                solar_acc = f"{solar_eval['accuracy']}%" if solar_eval else "nincs még elég tesztadat"
                wind_acc = f"{wind_eval['accuracy']}%" if wind_eval else "nincs még elég tesztadat"
                context = f"""
ALKALMAZÁSI ADATOK - NE TALÁLJ KI MÁS SZÁMOT.

JELENLEGI VALÓS ÁLLAPOT
- Utolsó tényleges mérés időpontja: {last_real_time}
- Napenergia tényleges termelése: {latest['solar_mw']:.1f} MW
- Szélenergia tényleges termelése: {latest['wind_mw']:.1f} MW
- Felhőzet: {latest['cloud_cover']}%
- Szélsebesség: {latest['wind_speed_10m']:.1f} km/h

IDŐALAPÚ MODELLTESZT
- Napenergia tesztpontosság: {solar_acc}
- Szélenergia tesztpontosság: {wind_acc}

JÖVŐBELI ELŐREJELZÉS
A következő értékek kizárólag a {last_real_time} UTÁNI időpontokra vonatkoznak.
Ezek még nem tényleges mérések:
{format_forecast_rows(future_df, 8)}

FONTOS:
- A jelenlegi valós állapot és a jövőbeli előrejelzés nem keverhető össze.
- A jövőbeli értékeket ne nevezd jelenlegi tényleges termelésnek.
- Ne számolj új százalékos eltérést.
- Ne állítsd, hogy egy jövőbeli előrejelzés alul- vagy túlbecsül, mert annak valódi értéke még nincs meg.
- Előrejelzési hibát csak ténylegesen lezárt, azonos időpontra vonatkozó mérés és korábbi előrejelzés alapján lehet állítani.
- Ha a kérdéshez nincs adat, mondd ki, hogy az alkalmazás nem biztosítja.

FELHASZNÁLÓ KÉRDÉSE:
{prompt}

Válaszolj röviden és szakmailag. A számokat pontosan a fenti adatokból idézd.
"""
                answer = call_gemini(context)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Rendszerhiba történt: {e}")


render_app()
