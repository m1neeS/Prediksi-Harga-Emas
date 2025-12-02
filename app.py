# app_plotly_future.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi Harga Emas Masa Depan", layout="wide")
st.title("Prediksi Harga Emas dengan GRU - Future Forecast")

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("Dataset-Emas.csv")

# -------------------------------
# Preprocessing
# -------------------------------
data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True, errors='coerce')

def str_to_float(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
        return float(x)
    return x

for col in ['Terakhir','Pembukaan','Tertinggi','Terendah']:
    data[col] = data[col].apply(str_to_float)

data = data.dropna(subset=['Tanggal','Terakhir','Pembukaan','Tertinggi','Terendah'])
data = data.sort_values('Tanggal')

# -------------------------------
# Load Models & Scalers
# -------------------------------
@st.cache_resource
def load_models():
    model_uni = tf.keras.models.load_model('gru_univariate.h5')
    model_multi = tf.keras.models.load_model('gru_multivariate.h5')
    
    scaler_uni = MinMaxScaler()
    scaler_multi = MinMaxScaler()
    
    scaler_uni.fit(data[['Terakhir']])
    scaler_multi.fit(data[['Terakhir','Pembukaan','Tertinggi','Terendah']])
    
    return model_uni, model_multi, scaler_uni, scaler_multi

model_uni, model_multi, scaler_uni, scaler_multi = load_models()

# -------------------------------
# Sidebar: pilih model & tanggal
# -------------------------------
st.sidebar.header("Pengaturan")
model_choice = st.sidebar.selectbox("Pilih Model", ["Univariate", "Multivariate"])

default_date = data['Tanggal'].max().date()
selected_date = st.sidebar.date_input(
    "Tanggal terakhir data historis",
    value=default_date
)
selected_date = pd.to_datetime(selected_date)

if selected_date not in data['Tanggal'].values:
    st.warning("Tanggal tidak ada di dataset, pilih tanggal lain.")
    st.stop()

# -------------------------------
# Sliding window
# -------------------------------
window = 5
data_filtered = data[data['Tanggal'] <= selected_date]
if len(data_filtered) < window:
    st.warning(f"Data tidak cukup untuk window={window}.")
    st.stop()
last_window = data_filtered.iloc[-window:]

# -------------------------------
# Siapkan input model
# -------------------------------
if model_choice == "Univariate":
    X_input = last_window[['Terakhir']].values.astype(float)
    X_input_scaled = scaler_uni.transform(X_input).reshape(1, window, 1)
else:
    X_input = last_window[['Terakhir','Pembukaan','Tertinggi','Terendah']].values.astype(float)
    X_input_scaled = scaler_multi.transform(X_input).reshape(1, window, 4)

# -------------------------------
# Prediksi 30 hari ke depan
# -------------------------------
future_days = 5

if st.button("Prediksi 5 Hari ke Depan"):
    preds = []
    last_window_scaled = X_input_scaled.copy()

    for _ in range(future_days):
        if model_choice == "Univariate":
            pred_scaled = model_uni.predict(last_window_scaled)[0][0]
            pred = scaler_uni.inverse_transform([[pred_scaled]])[0][0]
            preds.append(pred)
            # update window
            last_window_scaled = np.roll(last_window_scaled, -1)
            last_window_scaled[0, -1, 0] = pred_scaled
        else:
            pred_scaled = model_multi.predict(last_window_scaled)[0][0]
            dummy = np.zeros((1,4))
            dummy[0,0] = pred_scaled
            pred = scaler_multi.inverse_transform(dummy)[0][0]
            preds.append(pred)
            # update window
            new_input_scaled = np.zeros((1, window, 4))
            new_input_scaled[0,:-1,:] = last_window_scaled[0,1:,:]
            # terakhir = pred_scaled + kolom lain tetap last_window_scaled terakhir
            new_input_scaled[0,-1,0] = pred_scaled
            new_input_scaled[0,-1,1:] = last_window_scaled[0,-1,1:]
            last_window_scaled = new_input_scaled

    # Tanggal prediksi
    future_dates = [selected_date + pd.Timedelta(days=i) for i in range(1, future_days+1)]

    st.subheader("Hasil Prediksi 5 Hari Ke Depan")
    pred_df = pd.DataFrame({"Tanggal": future_dates, "Prediksi": preds})
    st.dataframe(pred_df)

    # -------------------------------
    # Plot historis + future
    # -------------------------------
    plot_window = 30
    plot_data = data_filtered.iloc[-plot_window:]

    fig = go.Figure()

    # Harga historis
    if model_choice == "Univariate":
        fig.add_trace(go.Scatter(
            x=plot_data['Tanggal'], y=plot_data['Terakhir'],
            mode='lines+markers', name='Close Historis'
        ))
    else:
        fig.add_trace(go.Scatter(x=plot_data['Tanggal'], y=plot_data['Terakhir'], mode='lines+markers', name='Close'))
        fig.add_trace(go.Scatter(x=plot_data['Tanggal'], y=plot_data['Pembukaan'], mode='lines+markers', name='Open'))
        fig.add_trace(go.Scatter(x=plot_data['Tanggal'], y=plot_data['Tertinggi'], mode='lines+markers', name='High'))
        fig.add_trace(go.Scatter(x=plot_data['Tanggal'], y=plot_data['Terendah'], mode='lines+markers', name='Low'))

    # Future prediksi: gabung jadi garis
    fig.add_trace(go.Scatter(
        x=future_dates, y=preds,
        mode='lines+markers', name='Prediksi Masa Depan',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="Harga Emas Historis & Prediksi 7 Hari ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        xaxis=dict(tickangle=-45, tickformat="%d/%m"),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
