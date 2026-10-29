# How Often Forecasts and Data Are Updated

The prices behind AgriPulse come from monthly sources: KAMIS (the Ministry of Agriculture's market information system), FEWS NET, and WFP market surveys, plus supporting data such as KNBS prices and fuel and rainfall figures, all published on a monthly cycle by their respective agencies. The price record runs up to the latest month these sources have published; the banner at the top of every page shows the current "data as of" month.

AgriPulse refreshes itself automatically on the 5th of every month. A scheduled job pulls the latest KAMIS and WFP prices, rebuilds the combined price history, re-runs the anomaly check, refits every forecast, and publishes the updated numbers to the app — no one has to do it by hand. Between refreshes the numbers stay fixed rather than changing with every new price tick, because the underlying sources themselves only publish monthly at best. A newly reported price will therefore appear in AgriPulse after the next monthly refresh, and a source that publishes late will show up a month later.

In practice, this matches how the underlying markets actually report: there is no meaningful "daily" price to track for most of these series, so a monthly refresh cycle is the natural cadence rather than a limitation.
