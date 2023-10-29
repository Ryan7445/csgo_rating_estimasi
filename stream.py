import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
cs_model = pickle.load(open('estimasi_rating_player.sav','rb'))
image = Image.open('banner.jpg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Rating CS GO Players')

col1, col2,col3=st.columns(3)
with col1:
    damage_per_round = st.number_input('Jumlah damage per-round :')
with col2:
    kills_per_death  = st.number_input('Jumlah kill per-death :')
with col3:
    kills_per_round  = st.number_input('Jumlah kills per-round :')
with col1:
    saved_teammates_per_round = st.number_input('Teammates Save per-round :')
with col2:
    opening_kill_ratio = st.number_input('Opening kill Ratio :')
with col3:
    opening_kill_rating = st.number_input('opening kill rating :')
with col1:
    four_kill_rounds = st.number_input('4 Kill Rounds :')
with col2:
    five_kill_rounds = st.number_input('5 Kill Rounsa :')

#code untuk estimasi
cs_est=''

#membuat button
with col1:
    if st.button('Estimasi Rating Player'):
        cs_pred = cs_model.predict([[damage_per_round,kills_per_death,kills_per_round,saved_teammates_per_round,opening_kill_ratio,opening_kill_rating,four_kill_rounds,five_kill_rounds]])

        st.success(f'Estimasi Rating Player : {cs_pred[0]:.2f}')