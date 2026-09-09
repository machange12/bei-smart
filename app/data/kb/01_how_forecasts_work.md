# How AgriPulse Forecasts Are Made

AgriPulse serves 27 live forecast series covering 6 commodities across 12 markets, each with forecasts at three horizons: 3, 6, and 12 months. Every series is modelled on its own rather than pooled together, because a single model trained across all series learns an average that fits none of them well.

Three model families are used: Prophet, SARIMA, and Chronos, a pre-trained foundation model that needs no fitting. For each series, all three are tested on a held-out 12-month window, and the one that performs best is selected to serve live forecasts. Two commodities, sorghum and maize meal, are served by a three-model ensemble instead, because on these short, erratic series no single model is confident enough to trust alone.

The forecasts you see are pre-computed from this pipeline, not generated on the spot. Each one carries the model that produced it, its measured test error, and a confidence label, so the number comes with a sense of how much to trust it.
