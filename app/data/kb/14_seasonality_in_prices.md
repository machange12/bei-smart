# Why Food Prices Follow an Annual Cycle

Food prices in Kenya move in a repeating annual pattern layered on top of a longer-term trend. Prices typically ease after harvest, when supply is plentiful, and tighten in the lean months before the next harvest arrives. A seasonal decomposition of the Nairobi maize series confirmed this annual cycle is real structure, not just an artefact of the long-term price trend running through it.

This seasonality is a large part of why forecasting models work at all on these series: knowing the time of year tells you a great deal about which direction price is likely to move, independent of anything else happening in the market.

It is also why adding rainfall as an explicit input to the models mostly did not help. Prophet's built-in yearly seasonality term already absorbs most of the annual price cycle that rainfall drives, so supplying rainfall data on top gave the model the same information twice rather than something new — extra noise without extra signal.
