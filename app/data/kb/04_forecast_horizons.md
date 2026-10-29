# Understanding Forecast Horizons (3, 6, 12 Months)

AgriPulse produces forecasts at three horizons: three, six, and twelve months ahead. Accuracy is not the same at every horizon. In the original multi-horizon test, typical error rose only gently, from around 13% at three months to around 15–16% at twelve months. The models have become more accurate overall since then (typical test error is now around 9–10%), but the same pattern holds: error grows slowly, not sharply, as the horizon stretches out.

The rise is modest rather than steep, and that is informative in itself. It means these models are mostly capturing a series' overall price level and its seasonal pattern — the annual rhythm of harvest and lean season — rather than short-term momentum from the last month or two of data. A model built around recent lags would degrade much faster as the horizon stretches out, because small early errors compound. Here, most of the forecast's value comes from knowing where a commodity's price level and seasonal cycle sit, which changes slowly.

Practically, this means a three-month forecast is somewhat more precise than a twelve-month one, but not dramatically so — a twelve-month forecast is still a reasonable planning horizon, just with a wider band around the central estimate.
