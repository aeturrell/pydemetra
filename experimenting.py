import pandas as pd
import pydemetra as jd

df = pd.read_parquet("/Users/arthur.turrell/Downloads/point_estimates_q_on_q.parquet")

df.index = df.index + pd.tseries.offsets.MonthBegin() - pd.DateOffset(months=1)

df.index = df.index.to_period()
df["London"] = df["London"].abs()

spec = jd.x13_spec("rsa5c")
# Force log transformation (no automatic detection)
spec["regarima"]["transform"]["fn"] = "LOG"

# Disable transitory component (TC) outlier detection
spec["regarima"]["outlier"]["outliers"] = [
    o for o in spec["regarima"]["outlier"]["outliers"] if o["type"] != "TC"
]

# Disable trading day regressors (suitable for quarterly data)
spec["regarima"]["regression"]["td"]["td"] = "TD_NONE"
spec["regarima"]["regression"]["td"]["auto"] = "AUTO_NO"

result = jd.x13(df["London"], spec)

sa = result["result"]["final"]["d11final"]  # seasonally adjusted series
trend = result["result"]["final"]["d12final"]  # trend


results_vanilla = jd.x13(df["London"], "rsa5c")
results_vanilla["result"]["final"]["d12final"]
