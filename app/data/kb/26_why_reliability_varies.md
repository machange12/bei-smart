# Why Some Forecasts Are More Reliable Than Others

Reliability differs series by series because the underlying price behaviour differs series by series. In the current system, 16 of 27 live series test at under 10% error and carry a high-confidence label, 7 test between roughly 11% and 15% (medium), and 4 test above 20% (low).

Several things drive this spread. Shorter or noisier series give a model less to learn from. Heavily interpolated series — where gaps in the original record were filled in with straight lines — can look deceptively easy to predict, because a model partly scored on reproducing a straight line it was handed will appear more accurate than it really is; one series had 41 of its 205 months filled this way. And some commodities, notably sorghum and maize meal, are simply erratic enough that no single model family fits them well, which is why those two are served by an ensemble of all three models rather than one selected winner.

Confidence labels exist so this variation is visible rather than hidden behind a uniform-looking number.
