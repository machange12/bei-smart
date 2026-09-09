# What Actually Drives Food Prices in This Data

Three external factors were tested as explicit inputs to the forecasting models: rainfall, diesel price, and the Geopolitical Risk Index. All three carry genuine information about food prices, but adding them directly to the models mostly did not improve forecasts.

Rainfall, lagged three and six months to match the planting-to-harvest cycle, made Prophet worse — in the worst case, nearly doubling its error. Two reasons: rainfall is measured as a single regional average, which says little about conditions on the specific farms supplying a specific market, and Prophet's own seasonality term was already capturing most of the annual pattern rainfall would explain. Diesel price behaved similarly when tested with SARIMA, moving the error from 11.54% to 11.52% — essentially nothing.

The one exception was the Geopolitical Risk Index in a single-series SARIMA test, where adding it improved accuracy from 20.2% to 17.8%. That refinement was not carried into the deployed system. The broader lesson: these drivers are real, but the models were already extracting most of their signal from price history alone.
