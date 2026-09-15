import sqlite3
import requests
from datetime import datetime

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
    conn.commit()
    conn.close()

def fetch_and_store_data():
    saved_count = 0
    energy_url = "https://api.energy-charts.info/public_power?country=hu"
    weather_url = "https://api.open-meteo.com/v1/forecast?latitude=47.4979&longitude=19.0402&hourly=temperature_2m,cloud_cover,wind_speed_10m,shortwave_radiation&past_days=7&forecast_days=3"
    
    try:
        print("Adatok lekérése a szerverekről (múlt + jövőbeli előrejelzés)...")
        e_res = requests.get(energy_url)
        e_res.raise_for_status()
        e_data = e_res.json()
        
        w_res = requests.get(weather_url)
        w_res.raise_for_status()
        w_data = w_res.json()
        
        timestamps = e_data.get("unix_seconds", [])
        
        w_times = w_data.get("hourly", {}).get("time", []) 
        temps = w_data.get("hourly", {}).get("temperature_2m", [])
        clouds = w_data.get("hourly", {}).get("cloud_cover", [])
        wind_speeds = w_data.get("hourly", {}).get("wind_speed_10m", [])
        radiation = w_data.get("hourly", {}).get("shortwave_radiation", [])
        
        weather_dict = {}
        for i, t_str in enumerate(w_times):
            formatted_t = t_str.replace("T", " ") + ":00"
            weather_dict[formatted_t] = {
                "temp": temps[i] if i < len(temps) else 0.0,
                "cloud": clouds[i] if i < len(clouds) else 0.0,
                "wind": wind_speeds[i] if i < len(wind_speeds) else 0.0,
                "rad": radiation[i] if i < len(radiation) else 0.0
            }

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        if timestamps:
            solar_values, wind_values = [], []
            for production in e_data.get("production_types", []):
                name = production.get("name")
                if name == "Solar":
                    solar_values = production.get("data", [])
                elif name == "Wind onshore":
                    wind_values = production.get("data", [])

            for i, ts in enumerate(timestamps):
                dt_string = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                solar = solar_values[i] if i < len(solar_values) and solar_values[i] is not None else 0.0
                wind = wind_values[i] if i < len(wind_values) and wind_values[i] is not None else 0.0
                w_info = weather_dict.get(dt_string, {"temp": 0.0, "cloud": 0.0, "wind": 0.0, "rad": 0.0})
                
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO power_production 
                        (timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (dt_string, solar, wind, w_info["temp"], w_info["cloud"], w_info["wind"], w_info["rad"]))
                    if cursor.rowcount > 0:
                        saved_count += 1
                except Exception as sql_e:
                    print(f"Hiba beszúráskor: {sql_e}")

        for t_str, w_info in weather_dict.items():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO power_production 
                    (timestamp, solar_mw, wind_mw, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation)
                    VALUES (?, NULL, NULL, ?, ?, ?, ?)
                ''', (t_str, w_info["temp"], w_info["cloud"], w_info["wind"], w_info["rad"]))
            except:
                pass
                
        conn.commit()
        conn.close()
        print(f"Adatbázis frissítve! Múltbeli adatok és jövőbeli előrejelzések rögzítve.")
        
    except Exception as e:
        print(f"Hiba a letöltés során: {e}")

if __name__ == "__main__":
    init_db()
    fetch_and_store_data()