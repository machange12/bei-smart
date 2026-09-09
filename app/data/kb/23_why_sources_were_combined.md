# Why Multiple Data Sources Were Combined

No single data source covers the full period and geography AgriPulse needs. WFP's historical record breaks in 2022 when its methodology changed; KAMIS picks up from there but did not exist as a comparable source before. FAO GIEWS, KNBS, and ERA5 each contribute one piece — food-security context, diesel prices, rainfall — that no price dataset carries on its own.

Combining sources is not free of risk. One clear example: the market of Eldoret appeared under two different names in the raw data. A join on exact date found zero overlap between them, at first suggesting two different markets. They were the same market — one source recorded observations on the 1st of the month, the other on the 15th, so dates never matched exactly even though both referred to the same monthly reading. The fix was to align records by the month they represent, not the literal date stamped on them.

The general lesson carried into how AgriPulse handles all its sources: multi-source time series need to be joined on the period they describe, not on matching dates.
