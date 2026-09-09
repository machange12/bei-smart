# The 2022 Price Shock

Kenya's food prices saw a sharp shock around 2022, part of a wider pattern visible in the Geopolitical Risk Index — a monthly measure built from global newspaper coverage of geopolitical tension — which spikes in 2014, 2020, and 2022. These are precisely the periods where every model tested performed worst.

An earlier, related spike in August 2021 was driven by drought combined with rising fuel costs. Neither AgriPulse's price history, rainfall data, nor fuel price data contained any advance signal of it. Every approach tested against this kind of event — SARIMA, Prophet, a small convolutional network, an LSTM, and the Chronos foundation model — missed it in the same way, because none of them had any input capable of anticipating it before it happened.

This is a structural limit, not a flaw specific to one model. A shock driven by geopolitics, war, or a sudden policy change shows up in price data only after the fact. AgriPulse's forecasts should be read as reflecting normal seasonal and trend behaviour, with the understanding that an unprecedented shock can override them without warning.
