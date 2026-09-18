import sqlite3
import requests
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DB_NAME = "energy_data.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS power_production (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            solar_mw REAL,
            wind_mw REAL,
            temperature_2m REAL,
            cloud_cover REAL,
            wind_speed_10m REAL,
            shortwave_radiation REAL
        )
    ''')
    try:
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE power_production ADD COLUMN ai_solar_mw REAL;")
            cursor.execute("ALTER TABLE power_production ADD COLUMN ai_wind_mw REAL;")
            conn.commit()
            print("Új AI oszlopok sikeresen létrehozva az adatbázisban!")
    except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def fetch_and_store_data():
    energy_url = "https://api.energy-charts.info/public_power?country=hu"
    weather_url = (
        "https://api.open-meteo.com/v1/forecast?latitude=47.4979&longitude=19.0402"
        "&hourly=temperature_2m,cloud_cover,wind_speed_10m,shortwave_radiation"
        "&past_days=7&forecast_days=3"
    )

    try:
        print("Adatok lekérése a szerverekről (múlt + jövőbeli időjárás)...")
        e_res = requests.get(energy_url, timeout=30)
        e_res.raise_for_status()
        e_data = e_res.json()

        w_res = requests.get(weather_url, timeout=30)
        w_res.raise_for_status()
        w_data = w_res.json()

        timestamps = e_data.get("unix_seconds", [])
        hourly = w_data.get("hourly", {})
        w_times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        clouds = hourly.get("cloud_cover", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        radiation = hourly.get("shortwave_radiation", [])

        weather_dict = {}
        for i, t_str in enumerate(w_times):
            formatted_t = t_str.replace("T", " ") + ":00"
            weather_dict[formatted_t] = {
                "temp": temps[i] if i < len(temps) else 0.0,
                "cloud": clouds[i] if i < len(clouds) else 0.0,
                "wind": wind_speeds[i] if i < len(wind_speeds) else 0.0,
                "rad": radiation[i] if i < len(radiation) else 0.0
            }

        solar_values = []
        wind_values = []
        for production in e_data.get("production_types", []):
            name = production.get("name")
            if name == "Solar":
                solar_values = production.get("data", [])
            elif name == "Wind onshore":
                wind_values = production.get("data", [])

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        changed_count = 0

        for i, ts in enumerate(timestamps):
            dt_string = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            solar = (
                solar_values[i]
                if i < len(solar_values) and solar_values[i] is not None
                else None
            )
            wind = (
                wind_values[i]
                if i < len(wind_values) and wind_values[i] is not None
                else None
            )
            w_info = weather_dict.get(
                dt_string,
                {"temp": 0.0, "cloud": 0.0, "wind": 0.0, "rad": 0.0}
            )

            cursor.execute('''
                INSERT INTO power_production
                (timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover,
                 wind_speed_10m, shortwave_radiation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    solar_mw = COALESCE(excluded.solar_mw, power_production.solar_mw),
                    wind_mw = COALESCE(excluded.wind_mw, power_production.wind_mw),
                    temperature_2m = excluded.temperature_2m,
                    cloud_cover = excluded.cloud_cover,
                    wind_speed_10m = excluded.wind_speed_10m,
                    shortwave_radiation = excluded.shortwave_radiation
            ''', (
                dt_string, solar, wind,
                w_info["temp"], w_info["cloud"],
                w_info["wind"], w_info["rad"]
            ))
            changed_count += 1

        for t_str, w_info in weather_dict.items():
            cursor.execute('''
                INSERT INTO power_production
                (timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover,
                 wind_speed_10m, shortwave_radiation)
                VALUES (?, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    temperature_2m = excluded.temperature_2m,
                    cloud_cover = excluded.cloud_cover,
                    wind_speed_10m = excluded.wind_speed_10m,
                    shortwave_radiation = excluded.shortwave_radiation
            ''', (
                t_str,
                w_info["temp"],
                w_info["cloud"],
                w_info["wind"],
                w_info["rad"]
            ))

        conn.commit()
        print(f"Adatbázis frissítve! Módosított/kezelt sorok: {changed_count}.")
        print("AI predikciók futtatása és frissítése...")
        
        df = pd.read_sql_query("SELECT * FROM power_production", conn)
        
        train_data = df.dropna(subset=["solar_mw", "wind_mw", "temperature_2m", "wind_speed_10m"])
        
        if not train_data.empty:
            rf_solar = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_wind = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rf_solar.fit(train_data[["temperature_2m", "shortwave_radiation", "cloud_cover"]], train_data["solar_mw"])
            rf_wind.fit(train_data[["wind_speed_10m"]], train_data["wind_mw"])
            
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            jelenlegi_ido = pd.Timestamp.now()
            
            mask = (df["ai_solar_mw"].isna() | (df['timestamp_dt'] > jelenlegi_ido)) & df["temperature_2m"].notna()
            
            rows_to_predict = df[mask]
            
            if not rows_to_predict.empty:
                for index, row in rows_to_predict.iterrows():
                    pred_solar = rf_solar.predict([[row["temperature_2m"], row["shortwave_radiation"], row["cloud_cover"]]])[0]
                    pred_wind = rf_wind.predict([[row["wind_speed_10m"]]])[0]
                    
                    cursor.execute("""
                        UPDATE power_production 
                        SET ai_solar_mw = ?, ai_wind_mw = ? 
                        WHERE id = ?
                    """, (round(pred_solar, 2), round(pred_wind, 2), row["id"]))
                
                conn.commit()
                print(f"Sikeresen lementve/frissítve {len(rows_to_predict)} új AI predikció!")
            else:
                print("Minden lezárt sornál van AI predikció, jövőbeli adat pedig jelenleg nincs.")

        conn.close()

    except Exception as e:
        print(f"Hiba a letöltés során: {e}")

if __name__ == "__main__":
    init_db()
    fetch_and_store_data()
