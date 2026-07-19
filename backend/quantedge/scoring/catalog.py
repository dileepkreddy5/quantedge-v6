"""The signal catalog — the declarative heart of the platform.

Every leaf signal is ONE dict here, never a function. Adding signal #2001 = adding
one row. The engine (compute.py + rollup.py) reads this catalog; it never changes.

Structure: INTELLIGENCES -> categories -> signals. Each level carries a weight
(relative, normalized at rollup). Each signal spec:
  id, label         identity
  field             which computed metric it reads (from the feature dict)
  weight            relative weight within its category
  higher_is_better  direction
  good, great       absolute anchor points (fallback when no peers)
  floor/cap(+_score) hard guardrails that override the percentile
  peer_key          key in peer 'factors' to rank against (optional)
  evidence          human string describing the source
"""

CATALOG = {
  "financial_statement": {
    "label": "Financial Statement Intelligence",
    "weight": 1.0,
    "categories": {
      "revenue": {
        "label": "Revenue Intelligence", "weight": 10,
        "signals": [
          {"id":"rev_growth_yoy","label":"Revenue growth YoY","field":"rev_growth_yoy",
           "weight":3,"higher_is_better":True,"good":0.05,"great":0.20,
           "floor":-0.05,"floor_score":25,"peer_key":"rev_growth_yoy",
           "evidence":"latest annual revenue vs prior year, 10-K"},
          {"id":"rev_cagr_3y","label":"Revenue 3y CAGR","field":"rev_cagr_3y",
           "weight":3,"higher_is_better":True,"good":0.05,"great":0.15,
           "peer_key":"rev_cagr_3y","evidence":"3-year revenue CAGR, 10-K"},
          {"id":"rev_stability","label":"Revenue stability","field":"rev_stability",
           "weight":2,"higher_is_better":True,"good":0.6,"great":0.9,
           "evidence":"1 - coefficient of variation of YoY growth"},
          {"id":"rev_predictability","label":"Growth persistence","field":"rev_growth_persistence",
           "weight":2,"higher_is_better":True,"good":0.0,"great":0.02,
           "evidence":"slope of growth trend across available years"},
        ],
      },
      "profitability": {
        "label": "Profitability Intelligence", "weight": 12,
        "signals": [
          {"id":"roic_wacc_spread","label":"ROIC - WACC spread","field":"roic_wacc_spread",
           "weight":5,"higher_is_better":True,"good":0.0,"great":0.15,
           "floor":0.0,"floor_score":30,"peer_key":"roic",
           "evidence":"NOPAT/invested capital minus WACC (value-creation test)"},
          {"id":"gross_margin","label":"Gross margin","field":"gross_margin",
           "weight":3,"higher_is_better":True,"good":0.30,"great":0.60,
           "peer_key":"gross_margin","evidence":"gross profit / revenue, TTM"},
          {"id":"operating_margin","label":"Operating margin","field":"operating_margin",
           "weight":2,"higher_is_better":True,"good":0.10,"great":0.25,
           "peer_key":"operating_margin","evidence":"operating income / revenue, TTM"},
          {"id":"margin_stability","label":"Margin stability","field":"gross_margin_stability",
           "weight":2,"higher_is_better":True,"good":0.6,"great":0.95,
           "evidence":"1 - CoV of gross margin across cycle (moat proxy)"},
        ],
      },
    },
  },
}
