
import argparse, json
from pathlib import Path
import pandas as pd
from src.engine import backtest

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--config", default="config.json")
p.add_argument("--outdir", default="outputs")
args = p.parse_args()

cfg = json.loads(Path(args.config).read_text())
df = pd.read_csv(args.data)
trades, metrics = backtest(df, cfg)

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
trades.to_csv(out/"trades.csv", index=False)
(out/"metrics.json").write_text(json.dumps(metrics, indent=2))

print(json.dumps(metrics, indent=2))
print(f"Saved: {out/'trades.csv'}")
