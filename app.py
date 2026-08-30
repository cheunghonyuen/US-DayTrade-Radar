import argparse, os, time
from pathlib import Path
import pandas as pd
import requests

DEFAULT_SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AMD', 'AVGO', 'NFLX', 'PLTR', 'HOOD', 'COIN', 'MSTR', 'RIVN', 'LCID', 'SOFI', 'MARA', 'RIOT', 'RKLB', 'ASTS', 'IONQ', 'RGTI', 'QBTS', 'SMCI', 'ARM', 'MU', 'INTC', 'QCOM', 'AMAT', 'LRCX', 'TSM', 'ASML', 'CRM', 'ORCL', 'NOW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'PANW', 'ZS', 'OKTA', 'MDB', 'UBER', 'LYFT', 'ABNB', 'DASH', 'SHOP', 'RBLX', 'SNAP', 'PINS', 'RDDT', 'GME', 'AMC', 'CVNA', 'AFRM', 'UPST', 'PYPL', 'XYZ', 'CELH', 'CAVA', 'HIMS', 'TMDX', 'TEM', 'CRDO', 'APP', 'DUOL', 'AXON', 'VST', 'CEG', 'NNE', 'OKLO', 'SMR', 'CCJ', 'F', 'GM', 'NIO', 'LI', 'XPEV', 'BABA', 'PDD', 'JD', 'NKE', 'SBUX', 'WMT', 'COST', 'TGT', 'JPM', 'BAC', 'C', 'GS', 'MS', 'XOM', 'CVX', 'OXY', 'SLB', 'UNH', 'LLY', 'NVO', 'MRNA', 'BNTX', 'PFE', 'BA', 'LMT', 'GE', 'CAT', 'DE', 'DIS', 'PARA', 'RKT', 'OPEN', 'W', 'CHWY', 'ETSY', 'ROKU', 'DKNG', 'PATH', 'AI', 'SOUN', 'BBAI', 'SERV', 'ACHR', 'JOBY']
BASE = "https://api.tiingo.com/tiingo/equity/intraday"

def fetch_symbol(symbol, start, end, token, freq):
    url = f"{BASE}/{symbol}/prices"
    params = {
        "startDate": start,
        "endDate": end,
        "resampleFreq": freq,
        "columns": "open,high,low,close,volume",
        "afterHours": "false",
        "forceFill": "false",
        "token": token,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise ValueError(f"{symbol} response missing date")
    df = df.rename(columns={"date":"timestamp"})
    df["symbol"] = symbol
    return df[["timestamp","symbol","open","high","low","close","volume"]]

def preflight(df):
    x=df.copy()
    x["timestamp"]=pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    return {
        "rows": int(len(x)),
        "symbols": int(x["symbol"].nunique()),
        "start_utc": str(x["timestamp"].min()),
        "end_utc": str(x["timestamp"].max()),
        "bad_timestamp": int(x["timestamp"].isna().sum()),
        "duplicates": int(x.duplicated(["symbol","timestamp"]).sum()),
        "bad_price_rows": int(((x["open"]<=0)|(x["high"]<=0)|(x["low"]<=0)|(x["close"]<=0)|(x["high"]<x["low"])).sum()),
        "negative_volume_rows": int((x["volume"]<0).sum()),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--freq", default="5min", choices=["1min","5min","15min"])
    p.add_argument("--out", default="sip_data.csv")
    p.add_argument("--cache-dir", default="sip_cache")
    p.add_argument("--limit", type=int, default=0)
    args=p.parse_args()

    token=os.getenv("TIINGO_TOKEN")
    if not token:
        raise RuntimeError("TIINGO_TOKEN is not set.")

    symbols=DEFAULT_SYMBOLS[:args.limit] if args.limit else DEFAULT_SYMBOLS
    cache=Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames=[]
    failed=[]

    for i,symbol in enumerate(symbols,1):
        fp=cache/f"{symbol}_{args.start}_{args.end}_{args.freq}.csv"
        print(f"[{i}/{len(symbols)}] {symbol} ...", flush=True)
        try:
            if fp.exists():
                df=pd.read_csv(fp)
                print(f"  cache {len(df):,} bars", flush=True)
            else:
                df=fetch_symbol(symbol,args.start,args.end,token,args.freq)
                if df.empty:
                    print("  no data", flush=True)
                    continue
                df.to_csv(fp,index=False)
                print(f"  {len(df):,} bars", flush=True)
                time.sleep(0.15)
            frames.append(df)
        except Exception as e:
            failed.append((symbol,str(e)))
            print(f"  FAILED: {e}", flush=True)

    if not frames:
        raise RuntimeError("No data downloaded.")

    data=pd.concat(frames,ignore_index=True)
    data=data.drop_duplicates(["symbol","timestamp"]).sort_values(["symbol","timestamp"])
    rpt=preflight(data)

    print("\nPREFLIGHT")
    for k,v in rpt.items():
        print(f"{k}: {v}")
    print(f"requested_symbols: {len(symbols)}")
    print(f"loaded_symbols: {data['symbol'].nunique()}")
    print(f"failed_symbols: {len(failed)}")

    if rpt["bad_timestamp"] or rpt["duplicates"] or rpt["bad_price_rows"] or rpt["negative_volume_rows"]:
        raise RuntimeError("Preflight failed.")

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.out,index=False)
    print(f"Saved: {args.out}")

if __name__=="__main__":
    main()
