import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from pandas.tseries.frequencies import to_offset

df = None

st.title("📈 Forecast con P50 / P90")
st.write("Choose how to provide your sales data:")
file = st.file_uploader("Upload a CSV file with 'ds' (date) and 'y' (sales)", type="csv")
gdrive_link = st.text_input("Or paste a Google Drive link to a CSV file")

if gdrive_link:
    import re
    match = re.search(r"/d/([\w-]+)", gdrive_link)
    if match:
        file_id = match.group(1)
        download_url = f"https://drive.google.com/uc?id={file_id}"
        try:
            df = pd.read_csv(download_url)
        except Exception as e:
            st.error(f"Failed to load CSV from Google Drive: {e}")
    else:
        st.error("Invalid Google Drive link format. Please use a link like https://drive.google.com/file/d/FILE_ID/view")
elif file:
    df = pd.read_csv(file)

if df is not None:
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    st.subheader("📊 Sales Data Preview")
    st.write(df.tail())

    # Fit Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Get last actual date and safe offset
    last_date = df['ds'].max()
    one_day = to_offset("1D")
    forecast_future = forecast[forecast['ds'] > last_date + one_day]

    # Create Plotly chart
    fig = go.Figure()

    # 1. Actual sales (historical)
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='black')
    ))

    # 2. Forecast (P50)
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        name='Forecast (P50)',
        line=dict(color='blue')
    ))

    # 3. Forecast (P90)
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        name='P90 (High Estimate)',
        line=dict(color='green', dash='dot')
    ))

    # 4. Forecast (P10)
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        name='P10 (Low Estimate)',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title="Actual Sales + Forecast with P10–P90 Uncertainty",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Legend"
    )

    st.plotly_chart(fig)
    # 6. Annotation label
    # fig.add_annotation(
    #     x=last_date,
    #     y=1.02,
    #     xref="x",
    #     yref="paper",
    #     showarrow=False,
    #     text="Forecast Starts",
    #     font=dict(color="gray")
    # )

    fig.update_layout(
        title="Actual Sales + Forecast with P10–P90 Uncertainty",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Legend"
    )

    st.plotly_chart(fig)
