"""AgriPulse entry point (Streamlit Cloud runs this file): the page router.

st.navigation gives every page a proper title and icon in the sidebar
instead of the file name ("streamlit app"), and fixes the URL of each page.
"""

import streamlit as st

st.navigation({
    "": [
        st.Page("home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/1_Forecast.py", title="Forecast", icon="📈", url_path="Forecast"),
        st.Page("pages/2_Alerts.py", title="Alerts & map", icon="🚨", url_path="Alerts"),
        st.Page("pages/6_Sell.py", title="Where to sell", icon="🚚", url_path="Sell"),
    ],
    "More": [
        st.Page("pages/4_Classify.py", title="Cheap or expensive?", icon="🏷️", url_path="Classify"),
        st.Page("pages/5_Coming_Soon.py", title="Coming soon", icon="⏳", url_path="Coming-soon"),
    ],
}).run()
