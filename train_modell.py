import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_solar_model():
    conn = sqlite3.connect('energy_data.db')
    df = pd.read_sql_query("SELECT timestamp, solar_mw, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation FROM power_production", conn)
    conn.close()

    if df.empty or len(df) < 24:
        print("Még nincs elég adat a tanításhoz! Futtasd többször az adatgyűjtőt.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    df = df.dropna()

    X = df[['temperature_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation', 'hour', 'month']]
    y = df['solar_mw']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Gépi tanulási modell tanítása (Random Forest)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicitions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predicitions)

    print(f"Modell sikeresen betanítva!")
    print(f"Átlagos hiba (MAE) a teszt adatokon: {mae:.2f} MW")

    # Példa predikció egy beképzelt tiszta, napos nyári délre (25°C, 0% felhő, 12. óra, június)
    summer_peak = pd.DataFrame([[25.0, 10.0, 5.0, 800.0, 12, 6]], columns=['temperature_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation', 'hour', 'month'])
    print(f"\n[Teszt] Várható termelés verőfényes délben: {model.predict(summer_peak)[0]:.2f} MW")

    night_time = pd.DataFrame([[15.0, 0.0, 3.0, 0.0, 2, 6]], columns=['temperature_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation', 'hour', 'month'])
    print(f"[Teszt] Várható termelés éjszaka: {model.predict(night_time)[0]:.2f} MW")

if __name__ == '__main__':
    train_solar_model()