# Why Some Forecasts Are More Reliable Than Others

Reliability differs series by series because the underlying price behaviour differs series by series. In the current system, roughly half of the live series (about 140 of around 270) test at under 10% error and carry a high-confidence label, around 110 test between 10% and 20% (medium), and only about a dozen test above 20% (low).

Two things drive this spread. First, shorter or noisier series give a model less to learn from — heavily interpolated series, where gaps in the original record were filled in with straight lines, can look deceptively easy to predict, because a model partly scored on reproducing a straight line it was handed will appear more accurate than it really is. Second, how current a series is matters: staleness itself is treated as a reason to trust a number less. A series whose latest price is more than 12 months old is not forecast at all, the confidence rules would cap an older series at medium or low regardless of its measured error, and the Forecast page only lists series with data from the last six months.

Confidence labels exist so this variation is visible rather than hidden behind a uniform-looking number.
