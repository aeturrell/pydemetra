import pandas as pd
import pydemetra as jd

df = pd.read_parquet("/Users/arthur.turrell/Downloads/point_estimates_q_on_q.parquet")

df.index = df.index + pd.tseries.offsets.MonthBegin() - pd.DateOffset(months=1)

df.index = df.index.strftime("%d-%m-%Y")



# df.to_excel("~/Desktop/sa_test.xlsx")
# df.to_csv("~/Desktop/sa_test.csv")



# Create a SARIMA model specification
model = jd.sarima_model(period=12, phi=[0.8], d=1, theta=[-0.6], bd=1, btheta=[-0.5])
print(model)
# SARIMA(1,1,1)(0,1,1)[12]

# Build a calendar with holidays
cal = jd.national_calendar(days=[
    jd.fixed_day(month=1, day=1),       # New Year
    jd.fixed_day(month=12, day=25),      # Christmas
    jd.easter_day(offset=-2),            # Good Friday
])

# Generate trading day regressors
td_regs = jd.td(frequency=12, start=2000, length=120)

# Run seasonality tests (requires JVM)
result = jd.seasonality_qs(df["London"], period=12)
