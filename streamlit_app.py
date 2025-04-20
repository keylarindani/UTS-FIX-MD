import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
with open("best_model_rf (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder (1).pkl", "rb") as f:
    encoder = pickle.load(f)

# Judul
st.markdown("<h1 style='text-align: center; color: #3366cc;'>Prediksi Pembatalan Booking Hotel 🏨</h1>", unsafe_allow_html=True)
st.markdown("---")

# Form input user
st.subheader("📝 Masukkan Informasi Booking:")

col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time (hari)", min_value=0)
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
    no_of_children = st.number_input("Jumlah Anak", min_value=0)
    no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
    no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0)
    avg_price = st.number_input("Harga Rata-Rata Kamar", min_value=0.0)

with col2:
    meal_plan = st.selectbox("Paket Makanan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room_type = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    market_segment = st.selectbox("Segment Pasar", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])
    parking = st.selectbox("Butuh Parkir?", [0, 1])
    repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
    special_request = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# Tambahan fitur historis
canceled_before = st.number_input("Pembatalan Sebelumnya", min_value=0)
not_canceled_before = st.number_input("Booking Sukses Sebelumnya", min_value=0)

# Submit button
if st.button("🔮 Prediksi"):
    # Buat dataframe
    input_df = pd.DataFrame([[
        lead_time, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
        avg_price, meal_plan, room_type, market_segment, parking, repeated_guest,
        canceled_before, not_canceled_before, special_request
    ]], columns=[
        'lead_time', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
        'no_of_week_nights', 'avg_price_per_room', 'type_of_meal_plan', 'room_type_reserved',
        'market_segment_type', 'required_car_parking_space', 'repeated_guest',
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
        'no_of_special_requests'
    ])

    # Encoding kolom kategorik
    for col in input_df.select_dtypes(include='object').columns:
        le = encoder[col]
        input_df[col] = le.transform(input_df[col])

    # Prediksi
    pred = model.predict(input_df)[0]

    # Output hasil prediksi
    st.markdown("---")
    if pred == 1:
        st.error("❌ Booking **DIBATALKAN**")
    else:
        st.success("✅ Booking **TIDAK DIBATALKAN**")
