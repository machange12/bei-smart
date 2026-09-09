# Why Different Markets Use Different Models

AgriPulse does not use one model for every market. Testing showed that no single approach wins everywhere, so each of the 27 live series is assigned whichever model performed best on its own twelve-month test window.

Prophet is picked where a series shows visible regime shifts — sudden trend changes — since it explicitly detects those changepoints. SARIMA is picked for smoother, more persistent series, and it takes most of the Eastern and arid-county retail markets, which move less sharply than city wholesale markets. Chronos, a foundation model that requires no training, is picked for noisy series where fitting parameters to a couple of hundred local observations tends to overfit; it takes most of the busy wholesale markets like Nairobi and Kisumu.

Sorghum and maize meal break this pattern. On these two commodities, no single model has a firm grip, so averaging all three (an ensemble) proved more stable than picking one — it cut Sorghum's error from 23.4% to 14.4% and maize meal's from 13.0% to 6.9%.
