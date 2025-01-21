import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
with open('insurance_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Judul aplikasi
st.title('Prediksi Biaya Asuransi')

# Informasi Pembuat
st.sidebar.header('Informasi Pembuat')
st.sidebar.write('Nama: SALMAN ALFARITSI')
st.sidebar.write('NIM: 2021230055')

# Input fitur
age = st.number_input('Umur', min_value=18, max_value=64, value=30)
sex = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Perempuan' if x==0 else 'Laki-laki')
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0)
children = st.number_input('Jumlah Anak', min_value=0, max_value=5, value=0)
smoker = st.selectbox('Perokok', [0, 1], format_func=lambda x: 'Tidak' if x==0 else 'Ya')

# Tombol prediksi
if st.button('Prediksi Biaya Asuransi'):
    # Siapkan input
    input_data = np.array([[age, sex, bmi, children, smoker]])
    
    # Scaling input
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    
    # Tampilkan hasil
    st.success(f'Estimasi Biaya Asuransi: ${prediction[0]:,.2f}')

# Footer
st.markdown('---')
st.markdown('© 2024 Aplikasi Prediksi Biaya Asuransi')
