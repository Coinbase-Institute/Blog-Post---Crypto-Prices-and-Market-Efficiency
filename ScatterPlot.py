# Import libraries
import pandas as pd
import yfinance as yf
import plotly.express as px

#%% CREATE DATASET

# 1 - IMPORT MARKETCAP FOR NASDAQ100

# Files required - first_35, second_35, third_27 (97 stocks pulled from capitaliq)
first_35 = pd.read_csv("data/first_35.csv")
second_35 = pd.read_csv("data/second_35.csv")
third_27 = pd.read_csv("data/third_27.csv")

n100_marketcap5y = (
    first_35.merge(second_35, on="Dates").merge(third_27, on="Dates").set_index("Dates")
)

n100_marketcap5y = n100_marketcap5y.unstack(level=0)

last = n100_marketcap5y.str[-1:]
n100_1 = n100_marketcap5y.loc[last == "b"].str[:-1].astype(float) * 1000000000
n100 = (
    n100_1.append(n100_marketcap5y.loc[last == "m"].str[:-2].astype(float) * 1000000)
    .to_frame("marketcap")
    .reset_index()
)
n100.rename(columns={"level_0": "id", "Dates": "Date"}, inplace=True)
n100["Date"] = pd.to_datetime(n100["Date"]).dt.date

# 2 - IMPORT BETAS FROM NASDAQ100
beta = pd.read_pickle("data/beta.pkl")
beta_n100 = beta.loc[beta["type"] == "n100"]

beta_n100["Date"] = pd.to_datetime(beta_n100["Date_utc"]).dt.date

# Keeping only required columns
beta_n100 = beta_n100[["Date", "id", "beta", "type"]]
beta_n100.sort_values(["id", "Date"])

# 3 - Merge BETA AND MARKETCAP FOR N100

beta_n100 = beta_n100.merge(n100, on=["Date", "id"], how="left")

# 4- LOAD UP CRYPTO BETA AND MARKETCAP

crypto_beta_marketcap = pd.read_csv("data/Crypto_Beta_Marketcap.csv")
crypto_beta_marketcap.info()

# 5 - IMPORT GAS,OIL MarketCap and Beta from Yahoo Finance

# Getting the OIL ETF data from yahoo finance
oil = yf.Ticker("CL=F").history(period="10y", auto_adjust=True)

# Computing the daily returns
oil = oil.sort_values(by="Date", ascending=False)
oil["ret_oil"] = (oil["Close"] - oil["Close"].shift(-1)) / oil["Close"].shift(-1)

oil = oil.dropna()

# Converting the time from eastern time to UTC
oil = oil.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
oil["Date_utc"] = pd.to_datetime(oil.index, format="%Y-%m-%d %H:%M:%S")
oil = oil.reset_index()

# Changing the date to 4pm ET, 9pm UTC
oil["Date_utc"] = oil["Date_utc"] + pd.DateOffset(hours=16)
oil.rename(columns={"Close": "Close_oil", "Volume": "Volume_oil"}, inplace=True)

oil = oil.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

# DROP 2020-04-20 20:00:00+00:00 because of negative price

oil = oil.loc[oil["Close_oil"] > 0]

#%% UPLOAD NATURAL GAS DATA


# Getting the GAS ETF data from yahoo finance
gas = yf.Ticker("NG=F").history(period="10y", auto_adjust=True)

# Computing the daily returns
gas = gas.sort_values(by="Date", ascending=False)
gas["ret_gas"] = (gas["Close"] - gas["Close"].shift(-1)) / gas["Close"].shift(-1)

gas = gas.dropna()

# Converting the time from eastern time to UTC
gas = gas.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
gas["Date_utc"] = pd.to_datetime(gas.index, format="%Y-%m-%d %H:%M:%S")
gas = gas.reset_index()

# Changing the date to 4pm ET, 9pm UTC
gas["Date_utc"] = gas["Date_utc"] + pd.DateOffset(hours=16)
gas.rename(columns={"Close": "Close_gas", "Volume": "Volume_gas"}, inplace=True)

gas = gas.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

# DROP 2020-04-20 20:00:00+00:00 because of negative price

gas = gas.loc[gas["Close_gas"] > 0]


#%% UPLOAD SP500 DATA

# To Download stock market data (S&P 500)

# Getting the S&P 500 data from yahoo finance
sp500 = yf.Ticker("^GSPC").history(period="10y", auto_adjust=True)

# Computing the daily returns
sp500 = sp500.sort_values(by="Date", ascending=False)
sp500["ret_sp500"] = (sp500["Close"] - sp500["Close"].shift(-1)) / sp500["Close"].shift(
    -1
)

sp500 = sp500.dropna()

# Converting the time from eastern time to UTC
sp500 = sp500.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
sp500["Date_utc"] = pd.to_datetime(sp500.index, format="%Y-%m-%d %H:%M:%S")
sp500 = sp500.reset_index()

# Changing the date to 4pm ET, 9pm UTC
sp500["Date_utc"] = sp500["Date_utc"] + pd.DateOffset(hours=16)
sp500 = sp500.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)


#%% Merge and Compute betas

sp500_comm = sp500.merge(gas, on="Date_utc", how="inner")
sp500_comm = sp500_comm.merge(oil, on="Date_utc", how="inner")
sp500_comm.sort_values("Date_utc", ascending=True, inplace=True)
sp500_comm.set_index("Date_utc", inplace=True)

# Compute Volatility of SP500

roll_vol1y = (
    sp500_comm[["ret_sp500", "ret_oil", "ret_gas"]].rolling(242).std().reset_index()
)
roll_vol1y.rename(
    columns={"ret_sp500": "vol_sp500", "ret_gas": "vol_gas", "ret_oil": "vol_oil"},
    inplace=True,
)

roll_corr1y_oil = sp500_comm[["ret_sp500", "ret_oil"]].rolling(242).corr().reset_index()
roll_corr1y_oil = (
    pd.DataFrame(roll_corr1y_oil.groupby("Date_utc").last()["ret_sp500"])
    .reset_index()
    .rename(columns={"ret_sp500": "corr_oil"})
)
roll_corr1y_gas = sp500_comm[["ret_sp500", "ret_gas"]].rolling(242).corr().reset_index()
roll_corr1y_gas = (
    pd.DataFrame(roll_corr1y_gas.groupby("Date_utc").last()["ret_sp500"])
    .reset_index()
    .rename(columns={"ret_sp500": "corr_gas"})
)

beta_comm = sp500_comm.merge(roll_corr1y_oil, on="Date_utc", how="inner")
beta_comm = beta_comm.merge(roll_corr1y_gas, on="Date_utc", how="inner")
beta_comm = beta_comm.merge(roll_vol1y, on="Date_utc", how="inner").dropna()
beta_comm["beta_oil"] = (
    beta_comm["corr_oil"] * beta_comm["vol_oil"] / beta_comm["vol_sp500"]
)
beta_comm["beta_gas"] = (
    beta_comm["corr_gas"] * beta_comm["vol_gas"] / beta_comm["vol_sp500"]
)

# SAVE BETA FOR GAS,OIL
beta_comm.to_csv("data/beta.csv")
beta = beta_comm

# IMPORT MARKETCAP FOR GAS AND OIL TO MERGE WITH BETA
gas_production = pd.read_csv("data/Gas Production.csv")
oil_production = pd.read_csv("data/Oil Production.csv")

# FORMATTING DF TO MERGE
month_dict = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}

df_array_production = [gas_production, oil_production]

# CONVERT COLUMN TO PROPER DATES "PULLED DATA HAS DATE FORMAT JUN-2022 ETC..."
month_year = []
for df in df_array_production:
    for index, row in df.iterrows():
        month = month_dict[row.Month[0:3]]
        year = row.Month[4:]
        # print(year,month)
        month_year.append(year + "-" + month + "-" + "11")
    df["Date"] = month_year
    month_year = []

gas["Date_str"] = gas["Date_utc"].astype(str)
oil["Date_str"] = oil["Date_utc"].astype(str)
beta["Date_str"] = beta["Date_utc"].astype(str)

beta = beta[["Date_str", "Date_utc", "beta_oil", "beta_gas"]]

df_array = [gas, oil, beta]
Date_arr = []

for df in df_array:
    for index, row in df.iterrows():
        Date_arr.append(row.Date_str[:10])

    df["Date_str"] = Date_arr
    Date_arr = []

gas_production["Date_str"] = gas_production["Date"].astype(str)
oil_production["Date_str"] = oil_production["Date"].astype(str)

gas_df = pd.merge(gas, gas_production, on="Date_str")
oil_df = pd.merge(oil, oil_production, on="Date_str")

gas_df = gas_df[["Close_gas", "Production", "Date_str", "Date_utc"]]
oil_df = oil_df[["Close_oil", "Production", "Date_str", "Date_utc"]]

gas_df["MarketCap"] = gas_df["Production"] * gas_df["Close_gas"] * 12 * 1000
oil_df["MarketCap"] = oil_df["Production"] * oil_df["Close_oil"] * 365 * 1000

merged_df = pd.merge(beta, gas_df, how="inner", on="Date_str")
merged_df = pd.merge(merged_df, oil_df, how="inner", on="Date_str")

merged_df = merged_df[
    ["Date_utc", "Date_str", "beta_oil", "beta_gas", "MarketCap_x", "MarketCap_y"]
]
merged_df.to_csv("data/Beta_Marketcap_Commodities.csv")


# 5- Put N100 and CRYPTO TOGETHER

final_df = beta_n100.append(
    crypto_beta_marketcap[["Date", "id", "beta", "marketcap", "type"]]
).reset_index()
final_df["Date"] = pd.to_datetime(final_df["Date"])
final_df = final_df.sort_values(by=["Date", "id"], ascending=[True, True])
final_df.dropna(inplace=True)

required_id = ["AMD", "NVDA", "TSLA", "AMZN", "GOOGL", "AAPL", "MSFT", "ETH", "BTC"]
final_df = final_df[final_df["id"].isin(required_id)]
final_df = final_df[["Date", "id", "beta", "type", "marketcap"]]
final_df["Date"] = final_df["Date"].astype(str)


# 6 - ADD ROWS FOR OIL AND GAS TO DATAFRAME
for index, row in merged_df.iterrows():
    date = row.Date_str
    marketcap_oil = row.MarketCap_y
    marketcap_gas = row.MarketCap_x
    beta_gas = row.beta_gas
    beta_oil = row.beta_oil
    final_df.loc[len(final_df.index)] = [
        date,
        "Gas",
        beta_gas,
        "Commodities",
        marketcap_gas,
    ]
    final_df.loc[len(final_df.index)] = [
        date,
        "Oil",
        beta_oil,
        "Commodities",
        marketcap_oil,
    ]


dfdate_tokeep = final_df.groupby(["Date"])["type"].unique().str.len() == 3
dfdate_tokeep.name = "tokeep"
final_df = final_df.merge(dfdate_tokeep, on="Date", how="inner")
final_df = final_df.loc[final_df["tokeep"] == True, :]


# Transform into monthly once date
final_df["Date"] = pd.to_datetime(final_df["Date"], format="%Y-%m-%d")
final_df = final_df.loc[final_df["Date"].dt.day == 11, :]
final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")

final_df = final_df.rename(
    columns={"type": "Type", "marketcap": "Marketcap", "beta": "Beta"}
)

final_df = final_df.replace({"n100": "NDX100", "crypto": "Cryptocurrency"})


figure = px.scatter(
    final_df,
    x="Marketcap",
    y="Beta",
    animation_frame="Date",
    animation_group="id",
    color="Type",
    hover_name="id",
    log_x=True,
    range_x=[1000000000, 5000000000000],
    range_y=[-1, 3],
    height=800,
    text="id",
    labels={"Marketcap": "Marketcap ($)", "Type": ""},
)


figure.update_traces(textposition="top right")
figure.update_layout(plot_bgcolor="#FFFFFF", font_family="sans-serif", font_size=24)
figure.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    showline=True,
    linewidth=1,
    linecolor="LightGrey",
)
figure.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor="black",
)
figure.update_traces(marker_size=22)

figure.show()
