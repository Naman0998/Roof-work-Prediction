import streamlit as st
import joblib
import pandas as pd
from feature_engineering import FastFeatureEnricher, FeatureSelector

# --- Load stacking model + metadata ---
saved = joblib.load(r"catboost_roof_model.pkl")
stacking_model = saved["catboost_model"]
#stacking_model = saved["stacking_model"]
selected_features = saved["selected_features"]
enricher = saved["enricher"]

# --- Define State → County mapping ---
state_county_map = {
    "NY": ["Brooklyn", "Queens", "Staten Island", "Westchester", "Nassau", "Suffolk",
           "Bronx", "Manhattan", "Dutchess County", "Rockland County", "Ulster County",
           "Putnam County", "Orange County"],
    "CT": ["Fairfield", "Hartford", "New Haven", "Middlesex", "New London", "Litchfield",
           "Windham", "Tolland"],
    "MA": ["Bristol", "Worcester", "Norfolk", "Plymouth", "Barnstable", "Berkshire",
           "Franklin", "Hampden", "Hampshire", "Middlesex", "Essex"],
    "ME": ["York County, ME", "Cumberland County, ME", "Kennebec", "Oxford", "Androscoggin",
           "Sagadahoc", "Penobscot", "Piscataquis", "Waldo", "Hancock", "Lincoln"],
    "MD": ["Baltimore County", "Anne Arundel County", "Prince George", "Montgomery County",
           "Howard County", "Carroll County", "Harford County", "Cecil County",
           "Wicomico County", "Talbot County", "Caroline County", "Dorchester County",
           "Frederick County", "Queen Anne's County"],
    "RI": ["Providence", "Newport"],
    "PA": ["Philadelphia County", "Delaware County", "Chester County", "Lancaster",
           "Berks County", "Lehigh", "Northampton County", "Bucks County", "Monroe County",
           "Dauphin County", "York County", "Adams County", "Luzerne County", "Lackawanna County"],
    "NJ": ["Bergen", "Hudson", "Union", "Morris", "Camden", "Passaic", "Somerset County",
           "Sussex County", "Ocean", "Burlington", "Gloucester", "Atlantic", "Cape May",
           "Warren", "Hunterdon", "Mercer", "Salem"],
    "NH": ["Rockingham", "Hillsborough", "Cheshire", "Merrimack", "Strafford", "Grafton",
           "Sullivan", "Belknap"],
    "DE": ["New Castle County", "Kent County", "Sussex"],
}

# High-risk counties
high_risk_counties = [
    'Queens', 'Fairfield', 'Westchester', 'Staten Island', 'Hartford',
    'New Haven', 'Middlesex', 'Brooklyn', 'Suffolk', 'Nassau'
]

# Rule-based fallback
def rule_based_prediction(inputs):
    roof_age = inputs["roof_age_num"]
    layers = inputs["roof_layers_num"]

    if roof_age >= 16:
        return 1, "Rule fired: Roof Age ≥ 16 years"
    elif layers >= 4:
        return 1, "Rule fired: Layers ≥ 4"
    else:
        return None, "No rule fired"

# --- Streamlit UI ---
st.title("🏠 Roof Work Prediction (Hybrid Rules + Catboost Model)")

roof_age = st.selectbox("Roof Age", ["0-5 years", "6-10 years", "11-15 years", 
                                     "16-20 years", "Above 20 years", "Unknown"])
layers = st.selectbox("Roof Layers", ["Unknown", "1", "2", "3", "4", "5"])
roof_type = st.selectbox("Roof Type", ["Pitched roof", "Unknown", "Flat roof", 
                                       "Asphalt - Pitched", "Metal - Pitched"])
state = st.selectbox("State", list(state_county_map.keys()))

# Show counties only for that state
counties = state_county_map.get(state, [])
county = st.selectbox("County", counties)

# Map inputs (raw)
age_map = {
    "0-5 years": 2.5, "6-10 years": 8, "11-15 years": 12,
    "16-20 years": 18, "Above 20 years": 25, "Unknown": 15
}
roof_age_num = age_map[roof_age]
layers_num = 1 if layers == "Unknown" else int(layers)

inputs = {
    "How old is the roof?": roof_age,
    "Number of Roof Layers": layers,
    "Type of Roofing": roof_type,
    "State": state,
    "County": county,
    "roof_age_num": roof_age_num,
    "roof_layers_num": layers_num,
    "high_risk_county": int(county in high_risk_counties)  # 🚨 New feature
}

if st.button("Predict"):
    # Rule check
    rule_pred, rule_msg = rule_based_prediction(inputs)

    if rule_pred is not None:
        st.success(f"✅ Prediction: Roof Work Needed ({rule_msg})")
    else:
        # --- ML Stacking Model Prediction ---
        df = pd.DataFrame([inputs])

        # Apply feature engineering + feature selection
        X_fe = enricher.transform(df)
        X_sel = X_fe[selected_features]

        # Get probability + calibrated decision
        probs = stacking_model.predict_proba(X_sel)[:, 1][0]

        # 🔥 Use threshold from training (Stacking ~0.232 from your output)
        threshold = 0.232
        pred = int(probs >= threshold)

        if pred == 1:
            st.success(f"✅ Prediction: Roof Work Needed (Stacking Model, {probs:.2f} confidence)")
        else:
            st.info(f"❌ Prediction: Roof Work Not Needed (Stacking Model, {probs:.2f} confidence)")

st.write("Inputs:", inputs)

