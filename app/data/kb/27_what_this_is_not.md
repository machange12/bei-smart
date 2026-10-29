# What AgriPulse Is Not

AgriPulse is not a trading or investment signal, and it is not a real-time price feed — it runs on periodic monthly data snapshots, not live ticks. It is also not exhaustive on the commodity side: it forecasts around twenty of the roughly fifty commodities in its data — mostly grains and pulses such as maize, beans, sorghum, millet and rice — leaving most vegetables, fruit, meat, dairy, sugar and more outside its scope. Geographic coverage is broad (all eight regions, about 80 markets) but still not every market in the country, and refugee settlement markets are deliberately excluded.

The price classifier deserves a candid note too. Tested against held-out data, it correctly labels a price as cheap, average, or expensive 61.5% of the time. That is actually a little below what you'd get by always guessing "average," the most common label, which is right 66.1% of the time on its own. Where the classifier earns its place is in catching genuine cheap and expensive outliers that a majority-guess would always miss entirely — it is meaningfully better at that specific job, just not at overall accuracy.

AgriPulse is best understood as a supporting reference: a way to see typical patterns, flag unusual readings, and get a general sense of direction — not a substitute for checking an actual market yourself.
