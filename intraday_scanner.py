"""
Intraday Scanner (NSE) - roz ke liye nayi high-volume stocks + entry / stop-loss / target / quantity

Run:
    pip install streamlit pandas requests yfinance
    streamlit run intraday_scanner.py

Kaise kaam karta hai:
 1. Pichhle din ki NSE bhavcopy se sabse liquid stocks (turnover) ki list banata hai (fixed list nahi).
 2. Un stocks ka aaj ka 5-minute data laata hai.
 3. Signal tabhi deta hai jab: price VWAP ke sahi side ho + Opening Range (pehle 15 min) breakout ho
    + aaj ka volume usi time ke normal volume se zyada ho.
 4. Stop-loss ATR se, target 1:2 risk-reward, aur quantity aapke capital aur risk % se nikalta hai.
"""
import io
import math
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

st.set_page_config(page_title="Intraday Scanner", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def liquid_universe(top_n: int, min_price: float, max_price: float):
    """Latest available bhavcopy -> most liquid EQ stocks by turnover."""
    d = date.today()
    for _ in range(10):
        try:
            r = requests.get(URL.format(d=d.strftime("%d%m%Y")), headers=HEADERS, timeout=20)
            if r.status_code == 200 and len(r.content) > 1000:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                df["SERIES"] = df["SERIES"].astype(str).str.strip()
                df = df[df["SERIES"] == "EQ"].copy()
                for c in ["CLOSE_PRICE", "TURNOVER_LACS"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df[(df["CLOSE_PRICE"] >= min_price) & (df["CLOSE_PRICE"] <= max_price)]
                df = df.sort_values("TURNOVER_LACS", ascending=False).head(top_n)
                return df["SYMBOL"].str.strip().tolist(), d
        except Exception:
            pass
        d -= timedelta(days=1)
    return [], None


@st.cache_data(ttl=120, show_spinner=False)
def intraday_data(symbols: tuple):
    tickers = [s + ".NS" for s in symbols]
    return yf.download(
        tickers, period="5d", interval="5m", group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )


def analyse(sym: str, raw: pd.DataFrame, capital: float, risk_pct: float, allow_short: bool):
    try:
        d = raw[sym + ".NS"].dropna()
    except Exception:
        return None
    if d.empty:
        return None
    d = d.copy()
    d["day"] = d.index.date
    last_day = d["day"].iloc[-1]
    td, prev = d[d["day"] == last_day], d[d["day"] < last_day]
    n = len(td)
    if n < 4 or prev.empty:
        return None

    tp = (td["High"] + td["Low"] + td["Close"]) / 3
    vwap = float((tp * td["Volume"]).cumsum().iloc[-1] / max(td["Volume"].cumsum().iloc[-1], 1))
    or_high, or_low = float(td["High"].iloc[:3].max()), float(td["Low"].iloc[:3].min())
    price = float(td["Close"].iloc[-1])

    vol_now = float(td["Volume"].sum())
    same_time = prev.groupby("day")["Volume"].apply(lambda g: g.iloc[:n].sum())
    rvol = vol_now / same_time.mean() if same_time.mean() > 0 else 0

    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if not atr or math.isnan(atr):
        return None

    prev_close = float(prev["Close"].iloc[-1])
    chg = (price / prev_close - 1) * 100

    side = None
    if price > vwap and price > or_high and rvol >= 1.5:
        side = "BUY"
    elif allow_short and price < vwap and price < or_low and rvol >= 1.5:
        side = "SELL (short)"
    if side is None:
        return None

    sl_dist = 1.5 * atr
    sl = price - sl_dist if side == "BUY" else price + sl_dist
    tgt = price + 2 * sl_dist if side == "BUY" else price - 2 * sl_dist
    qty = int(min((capital * risk_pct / 100) // sl_dist, capital // price))
    if qty < 1:
        return None

    return {
        "Stock": sym, "Signal": side, "Price": round(price, 2), "Chg %": round(chg, 2),
        "VWAP": round(vwap, 2), "Rel Vol": round(rvol, 1),
        "Entry": round(price, 2), "Stop-loss": round(sl, 2), "Target": round(tgt, 2),
        "Qty": qty, "Capital used": round(qty * price), "Max loss": round(qty * sl_dist),
        "Candle time": td.index[-1].strftime("%H:%M"),
    }


st.title("⚡ Intraday Scanner (NSE)")
st.warning(
    "Yeh sirf research tool hai, investment advice nahi. Intraday me zyadatar retail traders "
    "paisa kho dete hain (SEBI studies). Sirf utna paisa lagayein jo aap kho sakte hain."
)

with st.sidebar:
    st.header("Aapka setup")
    capital = st.number_input("Trading capital (₹)", 5000, 10_000_000, 50_000, 5000)
    risk_pct = st.slider("Risk per trade (% of capital)", 0.25, 2.0, 1.0, 0.25)
    allow_short = st.checkbox("Short (SELL) signals bhi dikhao", value=False)
    top_n = st.slider("Kitne liquid stocks scan karein", 30, 150, 60, 10)
    min_price, max_price = st.slider("Price range (₹)", 20, 5000, (50, 3000))
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

symbols, uni_date = liquid_universe(top_n, min_price, max_price)
if not symbols:
    st.error("NSE se universe nahi mil paya. Internet check karke dobara try karein.")
    st.stop()

with st.spinner(f"{len(symbols)} stocks ka 5-min data la rahe hain..."):
    raw = intraday_data(tuple(symbols))

rows = [r for s in symbols if (r := analyse(s, raw, capital, risk_pct, allow_short))]
res = pd.DataFrame(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Universe date (bhavcopy)", str(uni_date))
c2.metric("Stocks scanned", len(symbols))
c3.metric("Signals abhi", len(res))

if res.empty:
    st.info("Abhi koi setup nahi bana. Bina setup ke trade na karna bhi ek achhi strategy hai. "
            "Market 9:30 AM ke baad refresh karein.")
else:
    res = res.sort_values("Rel Vol", ascending=False).head(10)
    st.dataframe(res, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", res.to_csv(index=False), f"intraday_{date.today()}.csv", "text/csv")

st.subheader("Risk rules (zaroor follow karein)")
st.markdown(
    "- Ek din me maximum **2-3 trades**, aur har trade me stop-loss **lagana zaroori**.\n"
    "- Din ka loss capital ke **2-3%** tak pahunche to us din trading band.\n"
    "- Pehle 15 min (9:15-9:30) me entry na lein; sab positions **3:15 PM tak square off**.\n"
    "- Pehle 2-4 hafte **paper trading** (bina paise) se test karein.\n"
    "- Yahoo ka NSE data thoda delayed ho sakta hai; real-time ke liye broker API (Upstox / Angel One / Zerodha) use karein."
)
