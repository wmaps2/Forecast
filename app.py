import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from pandas.tseries.frequencies import to_offset

df = None

st.title("📈 Forecast con P50 / P90")
st.write("Choose how to provide your sales data:")
file = st.file_uploader("Upload a CSV file with 'ds' (date) and 'y' (sales)", type="csv")
gsheets_link = st.text_input("Or paste a Google Sheets link to a sheet")

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
                st.error("Could not find header row with 'ds' and 'y' columns in the sheet.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load CSV from Google Sheets: {e}")
    else:
        st.error("Invalid Google Sheets link format. Please use a link like https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
elif file:
    temp_df = pd.read_csv(file, header=None)
    header_row = find_header_row(temp_df)
    if header_row is not None:
        temp_df.columns = temp_df.iloc[header_row]
        df = temp_df.drop(range(header_row+1)).reset_index(drop=True)
    else:
        st.error("Could not find header row with 'ds' and 'y' columns in the file.")
        st.stop()

if df is not None:
    # Ensure 'ds' and 'y' columns exist anywhere in the file
    if {'ds', 'y'}.issubset(df.columns):
        df = df[['ds', 'y']]
    else:
        st.error("CSV/Sheet must contain columns named 'ds' (date) and 'y' (sales).")
        st.stop()

    # Safely convert 'ds' to datetime and drop invalid rows
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])

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
