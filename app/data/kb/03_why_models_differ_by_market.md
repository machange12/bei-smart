# Why Different Markets Use Different Models

AgriPulse does not use one model for every market. Testing showed that no single approach wins everywhere, so each of the 103 live series is assigned whichever model performed best on its own twelve-month test window — or, for series with fewer than 84 months of history, Chronos is used automatically, since it is the only candidate that needs no training data.

Where a series is long enough for a real contest, Prophet is picked where it shows visible regime shifts — sudden trend changes — since it explicitly detects those changepoints. SARIMA is picked for smoother, more persistent series. Chronos, a foundation model that requires no training, is picked for noisy series where fitting parameters to local observations tends to overfit. In practice Chronos ends up serving the large majority of series, simply because most markets have shorter histories than the 84-month bar the other two need for a fair validation contest.
