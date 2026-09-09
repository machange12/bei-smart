# Data Source: ERA5 Rainfall

Rainfall data in AgriPulse comes from ERA5, a monthly reanalysis dataset accessed through the World Bank's Climate Change Knowledge Portal, with gaps in the 2026 record filled in from NASA POWER. Figures are monthly totals aggregated by region.

Rainfall was tested as a direct model input, lagged three and six months to reflect the time between planting and harvest. It did not improve forecasts, and in Prophet's case made them noticeably worse. Two reasons stood out. First, a single regional rainfall figure is a coarse measure — Rift Valley alone spans climatically different areas from Turkana to Narok, so one number says little about conditions on the farms actually supplying a given market. Second, the models' own seasonal terms were already picking up most of the annual price cycle that rainfall drives, so adding it directly supplied the same information twice rather than anything new.

Rainfall remains useful as background context for understanding why prices move, even though it did not earn a place as a direct model input.
