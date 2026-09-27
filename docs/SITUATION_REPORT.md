# Situation Report — per-ticker, 90-day (concept, 2026-09-27)

The question: "What state is this company in right now, what has happened to
companies in this state before, and what is missing from the picture?"
It is a research view assembled from existing engines. It predicts nothing.

## Sections (all from existing computed data; each names its source)
1. PRICE STATE (90d) — Pattern Lab state vector: trend, slope, drawdown, 52w
   position, momentum 5/20/60, vol percentile/direction, volume trend, regime,
   multi-scale alignment. Source: /patterns/analogs state_vector.
2. WHAT FOLLOWED THIS STATE — analog distributions (20d + 60d shape) with base
   rates and SPY excess; conditional split for the ticker's own regime and 52w
   position. Source: /patterns/analogs, /patterns/conditions.
3. IF IT IS OFF ITS HIGHS — rebound stage (drawdown, trough age, bounce, slope)
   and the measured recovery base rate for its drawdown bucket. Source:
   rebound artifact + _RECOVERY_BASE_RATE. Absent if drawdown < 30%.
4. REPORTED RESULTS — last 2 quarters: revenue, margins, CapEx, R&D, buybacks,
   each with filing date (PIT). Source: XBRL capital-allocation events +
   bulk revenue. PRIMARY.
5. COMPANY INTELLIGENCE (90d) — material 8-Ks, insider open-market net,
   13F snapshot, attention vs filings. Source: /intel timeline. PRIMARY/SECONDARY.
6. NEWS TONE — FinBERT score on Polygon headlines already in the analysis;
   article counts vs baseline. SECONDARY; thin for small caps, stated.
7. NO SOURCE — analyst estimates / expected results, guidance, transcripts:
   listed explicitly as unavailable. Never inferred.
8. STATISTICAL NOTE — every distribution shows n (non-overlapping episodes),
   base rate, period; "DESCRIPTIVE RESULT" on the panel.

## Delivery
Pattern Lab mode "SITUATION REPORT"; one endpoint /patterns/situation/{ticker}
that assembles the above from internal calls (cache-first). No new models.
