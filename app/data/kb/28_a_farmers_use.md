# How a Farmer Might Use AgriPulse

A farmer selling maize in an Eastern retail market — Kitui, Makueni, Meru, or Isiolo — can check AgriPulse's forecast page before deciding when to sell rather than selling immediately at harvest. These particular retail series are among the system's strongest: several test under 7% error and carry a high-confidence label, because SARIMA fits their smoother, more persistent price pattern well.

Before relying on a number, check three things: the confidence label (high is worth more weight than low), the model behind it, and the width of the forecast band. A high-confidence three-month forecast is reasonable input for near-term selling decisions; a low-confidence twelve-month forecast — more common on volatile wholesale commodities like sorghum — should be treated as a rough direction only.

It is also worth checking the Alerts page for the same market: if the current price is already flagged as a spike, that changes the calculation for selling now versus waiting, independent of what the longer-range forecast says.
