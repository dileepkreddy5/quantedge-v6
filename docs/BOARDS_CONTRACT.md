# The Four Boards — contract (concept, 2026-09-27)

Rule: each board answers ONE question the others cannot, over a stated
universe, and displays its own measured track record beside its rows. A
board without a track record says "not yet measured" — never nothing.

| Board | Question | Universe | Ranking inputs | Track record to show |
|---|---|---|---|---|
| MULTIBAGGER | Growing fastest with quality (fundamental momentum) | all XBRL filers (~5,150) | quarterly YoY growth, Piotroski, margin trend, accruals, debt trend, volume confirmation | forward 6m/12m return distribution of each past nightly top-100 cohort vs universe; hit rate of top-10 |
| REBOUND | Quality names far below prior highs that stopped falling (mean reversion) | 500+ session names | drawdown, trough age, bounce, 20d slope, quality join | recovery-to-high odds by drawdown bucket (measured: 13.6% within 1y at 35-50%); stage transition rates |
| ASCENT | Climbing size tiers on sustained price+volume strength (trend) | all priced | relative strength, tier crossing, volume confirmation, persistence | 3-month persistence: share of climbers still above entry tier; excess vs SPY |
| SCREENER | The user's own filter over signals/horizons (interactive, not a board) | any analyzed | user-chosen | per-horizon validation badge on every forecast column |

Overlap handling: a name may appear on several boards; each row states why it
qualifies HERE. Remove: duplicated columns across boards, decorative scores,
any number without a stated rule.

Cohort tracking (new, shared): every nightly top-N per board is stored with
its date; a job fills forward returns at 1/3/6/12 months; boards display the
resulting base rates. This is the same outcome-filling pattern the signals
table uses. Until 6 months of cohorts exist, boards show "measured over N
weeks so far".

Build order: cohort store + outcome filler (shared) -> Rebound (smallest) ->
Ascent -> Multibagger -> Screener validation badges.
