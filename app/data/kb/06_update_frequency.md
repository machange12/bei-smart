# How Often Forecasts and Data Are Updated

The prices behind AgriPulse come from monthly sources: WFP and KAMIS market surveys, KNBS diesel pump prices, and ERA5 rainfall totals are all published on a monthly cycle by their respective agencies. The historical price record used by the system currently runs through August 2026.

AgriPulse itself works from a periodic snapshot of this data rather than a continuously live feed. Forecasts, anomaly flags, and volatility figures are produced by re-running the modelling pipeline against the latest available data, not recalculated with every new price tick, because the underlying sources themselves only publish monthly at best. This means a forecast you see reflects the most recent snapshot the pipeline was run against, and a newly reported price for a given month will not appear in AgriPulse until the pipeline is next re-run with updated data.

In practice, this matches how the underlying markets actually report: there is no meaningful "daily" price to track for most of these series, so a monthly refresh cycle is the natural cadence rather than a limitation.
