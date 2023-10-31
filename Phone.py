import pickle
import streamlit as st

model = pickle.load(open('Phone.sav', 'rb'))

st.title('Estimasi Phone')

battery_power = st.number_input('Masukan Persentase batrei', step=0,
                                max_value=100, min_value=1)
blue = st.number_input(
    'Masukan Speed')
clock_speed = st.number_input(
    'Masukan Jumlah Sim', step=0, max_value=2, min_value=1)
dual_sim = st.number_input(
    'Masukan Internal Memory', step=0, max_value=100000000, min_value=1024)
fc = st.number_input('Masukan Ram', step=0, max_value=9000, min_value=1)
four_g = st.number_input('Masukan Waktu Display',
                         step=0, max_value=100, min_value=1)
int_memory = st.number_input(
    'Masukan Brightes', step=0, max_value=100, min_value=1)
m_dep = st.number_input(
    'Masukan Harga', step=0, max_value=10000000, min_value=1)


predict = ''

if st.button(' Estimasi Car PRICE'):
    predict = model.predict(
        [[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep]]
    )
    st.write('Estimasi Phone PRICE: ', predict)
    st.write('Estimasi Phone PRICE: ', predict*2000)
