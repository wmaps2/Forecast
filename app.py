import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from pandas.tseries.frequencies import to_offset

df = None

st.title("📈 Forecast con P50 / P90")
st.write("Choose cómo proporcionar tus datos de ventas:")
file = st.file_uploader("Sube un archivo CSV con 'ds' (fecha) y 'y' (ventas)", type="csv")
gsheets_link = st.text_input("O pega un enlace de Google Sheets a una hoja")

# Add model selection dropdown
model_type = st.selectbox(
    "Elige el modelo de pronóstico:",
    ["Prophet", "Linear Regression", "ARIMA"]
)

def find_header_row(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        if 'ds' in row.values and 'y' in row.values:
            return i
    return None

if gsheets_link:
    import re
    match = re.search(r"/spreadsheets/d/([\w-]+)", gsheets_link)
    if match:
        sheet_id = match.group(1)
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        try:
            temp_df = pd.read_csv(export_url, header=None)
            header_row = find_header_row(temp_df)
            if header_row is not None:
                temp_df.columns = temp_df.iloc[header_row]
                df = temp_df.drop(range(header_row+1)).reset_index(drop=True)
            else:
                st.error("No se pudo encontrar la fila de encabezado con las columnas 'ds' y 'y' en la hoja.")
                st.stop()
        except Exception as e:
            st.error(f"Error al cargar el CSV desde Google Sheets: {e}")
    else:
        st.error("Formato de enlace de Google Sheets inválido. Usa un enlace como https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
elif file:
    temp_df = pd.read_csv(file, header=None)
    header_row = find_header_row(temp_df)
    if header_row is not None:
        temp_df.columns = temp_df.iloc[header_row]
        df = temp_df.drop(range(header_row+1)).reset_index(drop=True)
    else:
        st.error("No se pudo encontrar la fila de encabezado con las columnas 'ds' y 'y' en el archivo.")
        st.stop()

if df is not None:
    # Ensure 'ds' and 'y' columns exist anywhere in the file
    if {'ds', 'y'}.issubset(df.columns):
        df = df[['ds', 'y']]
    else:
        st.error("El CSV/la hoja debe contener columnas llamadas 'ds' (fecha) y 'y' (ventas).")
        st.stop()

    # Robustly parse dates using dateutil.parser
    from dateutil.parser import parse

    def robust_parse_date(val):
        try:
            return parse(str(val), dayfirst=True)
        except Exception:
            return pd.NaT

    df['ds'] = df['ds'].apply(robust_parse_date)
    df = df.dropna(subset=['ds'])

    st.subheader("📊 Vista previa de los datos de ventas")
    st.write(df.tail())

    # Detect frequency (daily, weekly, monthly)
    date_diffs = df['ds'].sort_values().diff().dt.days.dropna()
    freq = 'D'
    periods = 30
    if not date_diffs.empty:
        mode_diff = date_diffs.mode().iloc[0]
        if mode_diff == 7:
            freq = 'W'
            periods = 12  # 12 weeks forecast
        elif 28 <= mode_diff <= 31:
            freq = 'M'
            periods = 6   # 6 months forecast

    # Forecast logic based on selected model
    forecast = None
    forecast_future = None

    if model_type == "Prophet":
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        last_date = df['ds'].max()
        if freq == 'W':
            one_step = pd.Timedelta(days=7)
        elif freq == 'M':
            one_step = pd.DateOffset(months=1)
        else:
            one_step = to_offset("1D")
        forecast_future = forecast[forecast['ds'] > last_date + one_step]
        # Prepare data for plotting
        plot_actual_x = df['ds']
        plot_actual_y = df['y']
        plot_forecast_x = forecast_future['ds']
        plot_forecast_y = forecast_future['yhat']
        plot_p90_y = forecast_future['yhat_upper']
        plot_p10_y = forecast_future['yhat_lower']
    elif model_type == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        import numpy as np
        # Convert dates to ordinal for regression
        df_sorted = df.sort_values('ds')
        X = df_sorted['ds'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = df_sorted['y'].values
        lr = LinearRegression()
        lr.fit(X, y)
        # Forecast future dates
        last_date = df_sorted['ds'].max()
        if freq == 'W':
            future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=periods, freq='W')
        elif freq == 'M':
            future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        else:
            future_dates = pd.date_range(last_date + to_offset("1D"), periods=periods, freq='D')
        X_future = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y_pred = lr.predict(X_future)
        # Prepare data for plotting
        plot_actual_x = df_sorted['ds']
        plot_actual_y = df_sorted['y']
        plot_forecast_x = future_dates
        plot_forecast_y = y_pred
        plot_p90_y = y_pred + np.std(y)  # crude upper bound
        plot_p10_y = y_pred - np.std(y)  # crude lower bound
    elif model_type == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        df_sorted = df.sort_values('ds')
        y = df_sorted['y'].astype(float).values
        order = (1, 1, 1)
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=periods)
        y_pred = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        # Forecast future dates
        last_date = df_sorted['ds'].max()
        if freq == 'W':
            future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=periods, freq='W')
        elif freq == 'M':
            future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        else:
            future_dates = pd.date_range(last_date + to_offset("1D"), periods=periods, freq='D')
        # Prepare data for plotting
        plot_actual_x = df_sorted['ds']
        plot_actual_y = df_sorted['y']
        plot_forecast_x = future_dates
        plot_forecast_y = y_pred
        plot_p90_y = conf_int.iloc[:, 1]
        plot_p10_y = conf_int.iloc[:, 0]

    # Create Plotly chart
    fig = go.Figure()

    # 1. Actual sales (historical)
    fig.add_trace(go.Scatter(
        x=plot_actual_x,
        y=plot_actual_y,
        mode='lines+markers',
        name='Ventas Reales',
        line=dict(color='black')
    ))

    # 2. Forecast (P50)
    fig.add_trace(go.Scatter(
        x=plot_forecast_x,
        y=plot_forecast_y,
        mode='lines',
        name='Pronóstico (P50)',
        line=dict(color='blue')
    ))

    # 3. Forecast (P90)
    fig.add_trace(go.Scatter(
        x=plot_forecast_x,
        y=plot_p90_y,
        mode='lines',
        name='P90 (Estimación Alta)',
        line=dict(color='green', dash='dot')
    ))

    # 4. Forecast (P10)
    fig.add_trace(go.Scatter(
        x=plot_forecast_x,
        y=plot_p10_y,
        mode='lines',
        name='P10 (Estimación Baja)',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title="Ventas Reales + Pronóstico con Incertidumbre P10–P90",
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        legend_title="Leyenda"
    )

    st.plotly_chart(fig)
    # 6. Annotation label
    # fig.add_annotation(
    #     x=last_date,
    #     y=1.02,
    #     xref="x",
    #     yref="paper",
    #     showarrow=False,
    #     text="Inicio del Pronóstico",
    #     font=dict(color="gray")
    # )

    fig.update_layout(
        title="Ventas Reales + Pronóstico con Incertidumbre P10–P90",
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        legend_title="Leyenda"
    )

    st.plotly_chart(fig)
