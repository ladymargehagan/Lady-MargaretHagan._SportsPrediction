import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

filename = '/Users/margehagan/Downloads/fb_player_prediction_ensemble_model.pkl'
loaded_model = joblib.load(open(filename, 'rb'))

# feature names 
columns = [
    'potential', 'shooting', 'attacking_crossing', 'skill_fk_accuracy',
    'movement_reactions', 'wage_eur', 'value_eur', 'international_reputation',
    'physic', 'age', 'goalkeeping_diving', 'goalkeeping_handling',
    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
    'combined_passing', 'combined_dribbling', 'combined_mentality'
]

# configuration for streamlit webpage
def main():
    st.title("FIFA Player Prediction")
    html_temp = """
    <div style="background-color:#025246; padding:10px;">
    <h2 style="color:white; text-align:center;">FIFA Player Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for all features
    potential = st.text_input("Potential", "0")
    shooting = st.text_input("Shooting", "0")
    attacking_crossing = st.text_input("Attacking Crossing", "0")
    skill_fk_accuracy = st.text_input("Skill FK Accuracy", "0")
    movement_reactions = st.text_input("Movement Reactions", "0")
    wage_eur = st.text_input("Wage (in EUR)", "0")
    value_eur = st.text_input("Value (in EUR)", "0")
    international_reputation = st.text_input("International Reputation", "0")
    physic = st.text_input("Physic", "0")
    age = st.text_input("Age", "0")
    goalkeeping_diving = st.text_input("Goalkeeping Diving", "0")
    goalkeeping_handling = st.text_input("Goalkeeping Handling", "0")
    goalkeeping_kicking = st.text_input("Goalkeeping Kicking", "0")
    goalkeeping_positioning = st.text_input("Goalkeeping Positioning", "0")
    goalkeeping_reflexes = st.text_input("Goalkeeping Reflexes", "0")
    combined_passing = st.text_input("Passing Capabilities", "0")
    combined_dribbling = st.text_input("Dribbling Capabilities", "0")
    combined_mentality = st.text_input("Combined Mentality", "0")

    if st.button("Predict"):
        features = [[
            int(potential), int(shooting), int(attacking_crossing), int(skill_fk_accuracy),
            int(movement_reactions), int(wage_eur), int(value_eur), int(international_reputation),
            int(physic), int(age), int(goalkeeping_diving), int(goalkeeping_handling),
            int(goalkeeping_kicking), int(goalkeeping_positioning), int(goalkeeping_reflexes),
            int(combined_passing), int(combined_dribbling), int(combined_mentality)
        ]]
        
        df = pd.DataFrame(features, columns=columns)

        prediction = loaded_model.predict(df)
        
        output = prediction[0]
        st.success(f'Predicted value: {output}')
    
if __name__ == '__main__':
    main()
