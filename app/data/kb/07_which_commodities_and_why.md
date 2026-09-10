# Which Commodities Are Modelled, and Why Only Five

The underlying dataset contains dozens of tracked commodities, but only five currently meet the bar for a reliable forecast: Maize, Sorghum, Maize meal, and two bean varieties (Beans (mixed) and Beans (dry)). To qualify, a series needs at least 60 monthly observations and a test error no worse than 25%.

Potatoes came close but failed, both in the original WFP-based pipeline and again after the system was rebuilt on FEWS NET as its primary source. The median error across tested potato markets was 38%, with one market at 42%. Irish potato prices in Kenya swing on seasonal gluts that do not repeat on a stable annual cycle, and that irregularity is exactly what these models struggle with — they lean on seasonality being roughly the same shape each year. That the same commodity failed under two different data sources is itself informative: the problem is the price behaviour, not a gap in either source.

Other commodities are excluded either for lacking sufficient continuous history in a mainstream market, or for existing only in refugee settlement data where prices follow humanitarian distribution schedules rather than ordinary supply and demand. Adding a commodity that cannot be forecast reliably would give a false sense of precision, so it is left out rather than shown with a misleading number attached.
