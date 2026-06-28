"""Run the multibagger backtest on a sample. PYTHONPATH=. python this_file"""
import sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(__file__))
from backtest import backtest_company

# Score AS OF early 2021; measure forward return to mid-2023 (~2.5 years).
AS_OF = date(2021, 3, 1)
FWD_END = date(2023, 9, 1)

names = {
    "NVDA": "0001045810", "AAPL": "0000320193", "CELH": "0001341766",
    "SMCI": "0001375365", "AMD": "0000002488",  "INTC": "0000050863",
    "F":    "0000037996", "KO":   "0000021344",
}

rows = []
for tk, cik in names.items():
    try:
        comp, piotroski, fwd = backtest_company(tk, cik, AS_OF, FWD_END)
        rows.append((tk, comp, piotroski, fwd))
        fwds = "{:+.0%}".format(fwd) if fwd is not None else "n/a"
        print("{:5s} score {:5.1f} | piotroski {}/9 | fwd return {}".format(tk, comp, piotroski, fwds))
    except Exception as e:
        print("{:5s} ERROR {}: {}".format(tk, type(e).__name__, e))

valid = [(c, f) for _, c, _, f in rows if f is not None]
if len(valid) >= 4:
    valid.sort(reverse=True)
    half = len(valid) // 2
    top = [f for c, f in valid[:half]]
    bot = [f for c, f in valid[half:]]
    print()
    print("Top-half avg fwd return:    {:+.0%}".format(sum(top)/len(top)))
    print("Bottom-half avg fwd return: {:+.0%}".format(sum(bot)/len(bot)))
    print("Spread (top - bottom):      {:+.0%}".format(sum(top)/len(top) - sum(bot)/len(bot)))
    print("POSITIVE spread = the score had predictive lift on this sample.")
