"""
Coinbase Institute
Blog Post - Crypto Prices and Market Efficiency
Date: July 5th 2022
Author: Cesare Fracassi
Twitter: @CesareFracassi
Email: cesare.fracassi@coinbase.com
"""

#%% Importing required libraries
import pandas as pd
import numpy as np
!pip install yfinance
import yfinance as yf
import plotly.express as px
from datetime import datetime

#%% Reading CSV file for crypto marketcap
df_crypto = pd.read_csv("crypto_mktcap.csv")

# Procesing data in dataframe
df_crypto["Date"] = pd.to_datetime(df_crypto["Date"])
df_crypto.sort_values("Date", ascending=True, inplace=True)
df_crypto["Market Cap"] = (
    df_crypto["Market Cap"].replace("[\$,]", "", regex=True).astype(float)
)

#%% Reading bitcoin marketcap CSV
temp_df_bit = pd.read_csv("bitcoin_mktcap.csv")
temp_df_bit["Date"] = pd.to_datetime(temp_df_bit["Date"])
temp_df_bit = temp_df_bit.loc[
    (temp_df_bit["Date"] < "2013-04-28") & (temp_df_bit["Value"] != 0)
]
temp_df_bit.rename(columns={"Value": "Market Cap"}, inplace=True)

#%% Getting the S&P 500 data from yahoo finance
df_sp500 = yf.Ticker("^GSPC").history(period="max", auto_adjust=True)
df_sp500.reset_index(inplace=True)

df = df_sp500[["Date", "Close"]].merge(
    df_crypto[["Date", "Market Cap"]].append(temp_df_bit), on="Date", how="inner"
)

#%% Plotting a line chart for Marketcap against Date
figure = px.line(
    df,
    y="Market Cap",
    x="Date",
    labels={"Market Cap": "Crypto Market Cap ($)", "Date": "Year"},
)

#%% Converting dataframe to by Quarter
df_quarter = df
df_quarter["Dateq"] = df_quarter["Date"].dt.to_period("Q")

df_quarter = df_quarter.groupby("Dateq")[["Date", "Dateq", "Close", "Market Cap"]].tail(
    1
)

#%% Calculating returns for S&P 500 and  BTC
df_quarter["ret500"] = df_quarter["Close"].pct_change()
df_quarter["retbit"] = df_quarter["Market Cap"].pct_change()

date20 = df_quarter.loc[df_quarter["retbit"] < -0.2]["Date"].reset_index(drop=True)

#%% Adding a thick gray line to the figure where the date has a returns lesser than (- 20%)
for i in range(0, len(date20)):
    date20str = date20[i]
    figure.add_vline(x=date20str, line_color="#D3D3D3", line_width=5)


#%% Blog Theme Stlyes
figure.update_layout(plot_bgcolor="#FFFFFF")
figure.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    showline=True,
    linewidth=1,
    linecolor="black",
)
figure.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor="LightGrey", zeroline=False, type="log"
)
figure.update_traces(line_color="#1652f0", line_width=2)
figure.write_image("figure1.png")


df = df_sp500[["Date", "Close"]].merge(
    df_crypto[["Date", "Market Cap"]].append(temp_df_bit), on="Date", how="inner"
)
df.sort_values("Date", ascending=True, inplace=True)
df["ret_sp500"] = df["Close"].pct_change()
df["ret_crypto"] = df["Market Cap"].pct_change()


# Plotting Correlation Chart

roll_corr1y = (
    df[["Date", "ret_sp500", "ret_crypto"]]
    .set_index("Date")
    .rolling(242)
    .corr()
    .reset_index()
)
roll_corr1y = (
    pd.DataFrame(roll_corr1y.groupby("Date").last()["ret_sp500"])
    .reset_index()
    .rename(columns={"ret_sp500": "corr"})
)


figure = px.line(
    roll_corr1y,
    x="Date",
    y="corr",
    labels={"corr": "Correlation S&P500 - Crypto Market Cap", "Date": "Year"},
)

start_date = "2011-08-01"
end_date = "2022-07-01"
start_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")

# Blog Theme Stlyes
figure.update_layout(plot_bgcolor="#FFFFFF")
figure.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    showline=True,
    linewidth=1,
    linecolor="black",
    range=[start_date, end_date],
)

figure.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    zeroline=True,
    zerolinecolor="Red",
    range=[-0.2, 0.5],
)
figure.update_traces(line_color="#1652f0", line_width=1)
figure.write_image("figure2.png")
