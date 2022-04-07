# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Male", 1:"Female"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

    st.set_page_config(page_title="Death Prediction")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://previews.123rf.com/images/phonlamaiphoto/phonlamaiphoto1903/phonlamaiphoto190300024/118104573-chat-bot-concept-with-3d-rendering-humanoid-robot-with-headset-work-on-computer-notebook.jpg")

    with overview:
        st.title("Death prediction")

    with left:
        pclass_radio = st.radio( "Class", list(pclass_d.keys()), index=2, format_func=lambda x : pclass_d[x]);
        sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

    with right:
        age_slider = st.slider("Wiek", value=1.0, min_value=1.0, max_value=80.0, step=0.01)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0.0, max_value=512.0, step=0.0001)

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
