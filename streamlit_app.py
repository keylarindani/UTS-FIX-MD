import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile

# Load model & encoder
with open("best_model_rf (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder (1).pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("Prediksi Pembatalan Booking Hotel")

# Ambil input user
lead_time = st.number_input("Lead Time (hari)", min_value=0)
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
no_of_children = st.number_input("Jumlah Anak", min_value=0)
no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0)
meal_plan = st.selectbox("Paket Makanan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
parking = st.selectbox("Butuh Parkir?", [0, 1])
room_type = st.selectbox("Tipe Kamar", [
    "Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4",
    "Room_Type 5", "Room_Type 6", "Room_Type 7"
])
market_segment = st.selectbox("Segment Pasar", [
    "Offline", "Online", "Corporate", "Aviation", "Complementary"
])
repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
canceled_before = st.number_input("Pembatalan Sebelumnya", min_value=0)
not_canceled_before = st.number_input("Booking Sukses Sebelumnya", min_value=0)
avg_price = st.number_input("Harga Rata-Rata Kamar", min_value=0.0)
special_request = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# Buat DataFrame input
input_dict = {
    'lead_time': [lead_time],
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'type_of_meal_plan': [meal_plan],
    'required_car_parking_space': [parking],
    'room_type_reserved': [room_type],
    'market_segment_type': [market_segment],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [canceled_before],
    'no_of_previous_bookings_not_canceled': [not_canceled_before],
    'avg_price_per_room': [avg_price],
    'no_of_special_requests': [special_request]
}

input_df = pd.DataFrame(input_dict)

# Lakukan encoding hanya pada kolom kategorikal
cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
encoded_input = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_cols))

# Gabung dengan kolom numerik
input_numerik = input_df.drop(columns=cat_cols).reset_index(drop=True)
final_input = pd.concat([input_numerik, encoded_df], axis=1)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(final_input)[0]
    if pred == 1:
        st.error("❌ Booking kemungkinan AKAN DIBATALKAN.")
    else:
        st.success("✅ Booking kemungkinan TIDAK akan dibatalkan.")
