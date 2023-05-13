import streamlit as st
import numpy as np
import pickle

loaded_model = pickle.load(open('F:/projet_ia/model.sav', 'rb'))


def cancer_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        return "Votre Vin est mauvais"
    elif (prediction[0] == 1):
        return "Votre vin est bon"
    else:
        return "Erreur"


def main():
    st.title('Vin Prediction Web App')
    volatile_acidity = st.text_input('volatile_acidity')
    citric_acid = st.text_input('citric_acid')
    residual_sugar = st.text_input('residual_sugar')
    chlorides = st.text_input('chlorides')
    free_sulfur_dioxide = st.text_input('free_sulfur_dioxide')
    pH = st.text_input('pH')
    alcohol = st.text_input('alcohol')
    diagnosis = ''

    if st.button("Vin Test"):
        diagnosis = cancer_prediction([volatile_acidity, citric_acid, residual_sugar,
                                       chlorides, free_sulfur_dioxide, pH, alcohol])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
