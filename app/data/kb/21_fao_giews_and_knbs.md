# Data Sources: FAO GIEWS and KNBS

Two further sources support AgriPulse without being the primary price record. FAO GIEWS, the Global Information and Early Warning System run by the UN Food and Agriculture Organization, provides market price and food-security context useful for cross-checking coverage in food-insecure areas; it is not one of the price feeds stitched into the forecasting data.

KNBS, the Kenya National Bureau of Statistics, supplies monthly observed diesel pump prices for Nairobi, covering 2006 through 2026. This was brought in specifically to test whether transport cost — which feeds into market prices with roughly a one- to two-month lag — could improve forecasts as an explicit model input. It was tested alongside rainfall and the Geopolitical Risk Index; on its own it did not meaningfully change forecast accuracy when added to SARIMA, though the underlying idea that transport cost affects price remains sound.

KNBS also contributes a small number of retail price records to the price history (for example, dry beans in Nakuru). Otherwise both sources are used for context and testing rather than as the core series that forecasts are built from.
