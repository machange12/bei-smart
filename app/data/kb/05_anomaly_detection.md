# How Price Anomalies Are Flagged

AgriPulse flags a price as an anomaly by comparing the latest observation against the normal range (the interquartile range) of that same series' own history — not against a national price or another market.

Building this required three safeguards. Without them, the detector flagged a Nakuru maize reading from 2012 on a series that had ended fourteen years earlier, produced negative price floors on series whose range had collapsed to zero after gaps were filled in, and flagged commodities never actually monitored by the system. With those guards in place, AgriPulse currently monitors 55 commodity-market series. Three are presently flagged as a SPIKE: Beans (mixed) in Kitui, and Maize in both Marsabit and Taita Taveta.

A SPIKE label means that specific market's own recent price sits outside its own normal range — it does not mean the price is unusually high compared with the rest of Kenya. A price that looks high nationally can be perfectly normal for a market that always runs high, and vice versa.
