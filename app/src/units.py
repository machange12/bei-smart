"""Price unit per commodity: almost everything is KES/kg; livestock is per animal."""

# Mirrors PER_HEAD in scripts/build_chained_all.py (rows there carry unit="HEAD").
PER_HEAD = {"Goats"}


def unit_label(commodity: str | None, language: str = "en") -> str:
    if commodity in PER_HEAD:
        return "kichwa" if language == "sw" else "head"
    return "kg"
