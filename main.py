from fastapi import FastAPI
from pydantic import BaseModel
import math
import pandas as pd
from pnsea import NSE

app = FastAPI(title="NSE Options Analytics API")

nse = NSE()

# -----------------------------
# BLACK SCHOLES
# -----------------------------

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return math.exp(-x**2 / 2.0) / math.sqrt(2 * math.pi)

def calculate_greeks(S, K, T, r, sigma, option_type="CE"):

    if T <= 0 or sigma <= 0:
        return {"Delta":0,"Gamma":0,"Theta":0,"Vega":0}

    d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    if option_type == "CE":
        delta = norm_cdf(d1)
        theta = -(S*norm_pdf(d1)*sigma/(2*math.sqrt(T))) - r*K*math.exp(-r*T)*norm_cdf(d2)
    else:
        delta = norm_cdf(d1)-1
        theta = -(S*norm_pdf(d1)*sigma/(2*math.sqrt(T))) + r*K*math.exp(-r*T)*norm_cdf(-d2)

    gamma = norm_pdf(d1)/(S*sigma*math.sqrt(T))
    vega = S*norm_pdf(d1)*math.sqrt(T)/100

    return {
        "Delta":round(delta,3),
        "Gamma":round(gamma,5),
        "Theta":round(theta/365,3),
        "Vega":round(vega,3)
    }

# -----------------------------
# REQUEST MODELS
# -----------------------------

class SymbolRequest(BaseModel):
    symbol: str = "NIFTY"

class GreeksRequest(BaseModel):
    symbol: str
    strike: int
    option_type: str

# -----------------------------
# DATA FETCH
# -----------------------------

def fetch_chain(symbol):
    df, expiries, underlying = nse.options.option_chain(symbol)
    return df, expiries, underlying

# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.post("/option-chain-summary")
def option_chain_summary(req: SymbolRequest):

    df, expiries, underlying = fetch_chain(req.symbol)

    ce = df.sort_values("CE_openInterest",ascending=False).head(5)
    pe = df.sort_values("PE_openInterest",ascending=False).head(5)

    return {
        "symbol":req.symbol,
        "underlying":underlying,
        "expiries":expiries[:3],
        "top_ce":ce[["strikePrice","CE_openInterest","CE_lastPrice"]].to_dict(),
        "top_pe":pe[["strikePrice","PE_openInterest","PE_lastPrice"]].to_dict()
    }

# -----------------------------

@app.post("/pcr")
def pcr(req: SymbolRequest):

    df,_,_ = fetch_chain(req.symbol)

    ce_oi = df["CE_openInterest"].sum()
    pe_oi = df["PE_openInterest"].sum()

    pcr = pe_oi/ce_oi if ce_oi>0 else 0

    sentiment = "Neutral"

    if pcr>1.3:
        sentiment="Bullish"

    if pcr<0.7:
        sentiment="Bearish"

    return {
        "symbol":req.symbol,
        "PCR":round(pcr,2),
        "sentiment":sentiment
    }

# -----------------------------

@app.post("/max-pain")
def max_pain(req: SymbolRequest):

    df,_,underlying = fetch_chain(req.symbol)

    strikes = df["strikePrice"].tolist()

    min_pain=float("inf")
    max_pain_strike=strikes[0]

    for strike in strikes:

        pain=0

        for _,row in df.iterrows():

            if strike>row["strikePrice"]:
                pain+=(strike-row["strikePrice"])*row["CE_openInterest"]

            if strike<row["strikePrice"]:
                pain+=(row["strikePrice"]-strike)*row["PE_openInterest"]

        if pain<min_pain:
            min_pain=pain
            max_pain_strike=strike

    return {
        "symbol":req.symbol,
        "max_pain":max_pain_strike,
        "spot":underlying
    }

# -----------------------------

@app.post("/support-resistance")
def support_resistance(req: SymbolRequest):

    df,_,underlying = fetch_chain(req.symbol)

    resistance = df.sort_values("CE_openInterest",ascending=False).head(3)
    support = df.sort_values("PE_openInterest",ascending=False).head(3)

    return {
        "symbol":req.symbol,
        "spot":underlying,
        "resistance":resistance[["strikePrice","CE_openInterest"]].to_dict(),
        "support":support[["strikePrice","PE_openInterest"]].to_dict()
    }

# -----------------------------

@app.post("/greeks")
def greeks(req: GreeksRequest):

    df,_,underlying = fetch_chain(req.symbol)

    row=df[df["strikePrice"]==req.strike]

    if row.empty:
        return {"error":"Strike not found"}

    row=row.iloc[0]

    iv=row[f"{req.option_type}_impliedVolatility"]/100

    greeks=calculate_greeks(
        underlying,
        req.strike,
        7/365,
        0.07,
        iv,
        req.option_type
    )

    return {
        "symbol":req.symbol,
        "strike":req.strike,
        "type":req.option_type,
        "greeks":greeks
    }
