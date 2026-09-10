# Why Some Forecasts Are More Reliable Than Others

Reliability differs series by series because the underlying price behaviour differs series by series. In the current system, 47 of 103 live series test at under 10% error and carry a high-confidence label, 42 test between roughly 10% and 20% (medium), and 14 test above 20% (low).

Two things drive this spread. First, shorter or noisier series give a model less to learn from — heavily interpolated series, where gaps in the original record were filled in with straight lines, can look deceptively easy to predict, because a model partly scored on reproducing a straight line it was handed will appear more accurate than it really is. Second, confidence is also capped by how current a series is: a series that is otherwise well-fitting but more than 12 months stale is capped at medium confidence, and one more than 24 months stale is capped at low, regardless of its measured error — staleness itself is treated as a reason to trust a number less.

Confidence labels exist so this variation is visible rather than hidden behind a uniform-looking number.
