import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd

# Load model CNN
def load_model():
    return tf.keras.models.load_model("model_skripsinew.h5")

model = load_model()

# Prediksi
def model_prediction(image_data):
    image = Image.open(image_data).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return np.argmax(prediction)

# Daftar nama kelas ikan
class_name = [
    'Bandeng', 'Bawal', 'Cupang', 'Gabus', 'Gurame',
    'Ikan Cere', 'Ikan Mas', 'Kakap', 'Lele', 'Mujair',
     'Nila','Patin'
]

# Informasi edukatif
fish_info = {
    'Bandeng': {'Nama Ilmiah': 'Chanos chanos', 'Ciri-ciri': 'Tubuh memanjang, sisik besar berkilau, ekor bercabang.',
                'Habitat': 'Perairan payau dan laut dangkal.', 'Kegunaan': 'Konsumsi, budidaya tambak.'},
    'Bawal': {'Nama Ilmiah': 'Colossoma macropomum', 'Ciri-ciri': 'Badan lebar dan pipih, warna keperakan dengan sirip gelap.',
              'Habitat': 'Sungai dan danau air tawar.', 'Kegunaan': 'Konsumsi, sering dibudidayakan.'},
    'Betok': {'Nama Ilmiah': 'Anabas testudineus', 'Ciri-ciri': 'Tubuh kecil bersisik kasar, bisa bernapas di udara.',
              'Habitat': 'Sawah, rawa, perairan dangkal.', 'Kegunaan': 'Konsumsi lokal.'},
    'Cupang': {'Nama Ilmiah': 'Betta splendens', 'Ciri-ciri': 'Warna cerah, sirip panjang mengembang.',
               'Habitat': 'Air tenang seperti kolam dan selokan.', 'Kegunaan': 'Ikan hias, kadang aduan.'},
    'Gabus': {'Nama Ilmiah': 'Channa striata', 'Ciri-ciri': 'Tubuh panjang, kepala seperti ular.',
              'Habitat': 'Sungai, rawa, danau.', 'Kegunaan': 'Konsumsi, pengobatan luka tradisional.'},
    'Gurame': {'Nama Ilmiah': 'Osphronemus goramy', 'Ciri-ciri': 'Badan pipih, sisik kasar, sirip panjang.',
               'Habitat': 'Sungai dan kolam air tenang.', 'Kegunaan': 'Konsumsi favorit.'},
    'Ikan Cere': {'Nama Ilmiah': 'Rasbora spp.', 'Ciri-ciri': 'Ikan kecil, tubuh ramping, warna keperakan.',
                  'Habitat': 'Sungai kecil dan danau.', 'Kegunaan': 'Ikan hias, umpan mancing.'},
    'Ikan Mas': {'Nama Ilmiah': 'Cyprinus carpio', 'Ciri-ciri': 'Tubuh besar, bersisik kuning keemasan.',
                 'Habitat': 'Danau, kolam, sungai lambat.', 'Kegunaan': 'Konsumsi, lomba mancing.'},
    'Jade Perch': {'Nama Ilmiah': 'Scortum barcoo', 'Ciri-ciri': 'Tubuh oval, warna abu-abu kehijauan.',
                   'Habitat': 'Sungai dan perairan tenang Australia.', 'Kegunaan': 'Konsumsi, omega-3 tinggi.'},
    'Kakap': {'Nama Ilmiah': 'Lutjanus spp.', 'Ciri-ciri': 'Tubuh panjang, warna merah atau abu.',
              'Habitat': 'Muara dan perairan pantai.', 'Kegunaan': 'Konsumsi restoran.'},
    'Lele': {'Nama Ilmiah': 'Clarias batrachus', 'Ciri-ciri': 'Tubuh licin, berkumis, tanpa sisik.',
             'Habitat': 'Kolam, rawa, sungai.', 'Kegunaan': 'Konsumsi, budidaya masif.'},
    'Mujair': {'Nama Ilmiah': 'Oreochromis mossambicus', 'Ciri-ciri': 'Tubuh gepeng, warna abu atau gelap.',
               'Habitat': 'Danau dan sungai air tawar.', 'Kegunaan': 'Konsumsi rakyat.'},
    'Nila': {'Nama Ilmiah': 'Oreochromis niloticus', 'Ciri-ciri': 'Mirip mujair, warna terang dengan garis gelap.',
             'Habitat': 'Kolam dan sungai.', 'Kegunaan': 'Konsumsi dan ekspor.'},
    'Patin': {'Nama Ilmiah': 'Pangasius spp.', 'Ciri-ciri': 'Tubuh licin, putih keabu-abuan, tanpa sisik.',
              'Habitat': 'Sungai besar seperti Mekong.', 'Kegunaan': 'Konsumsi, industri fillet.'}
}

# Statistik
STAT_FILE = "statistik_deteksi.json"

def load_statistics():
    if os.path.exists(STAT_FILE):
        with open(STAT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_statistics(stats):
    try:
        with open(STAT_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        st.error(f"Gagal menyimpan statistik: {e}")


def update_statistics(ikan_nama):
    stats = load_statistics()
    stats[ikan_nama] = stats.get(ikan_nama, 0) + 1
    save_statistics(stats)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Pilih Halaman", ["Home", "Informasi Web", "Riwayat Deteksi", "Fish Recognition"])

# Halaman Home
if app_mode == "Home":
    st.header("🎣 DETEKSI JENIS IKAN AIR TAWAR")
    st.image("bawal14.jpg", use_column_width=True)
    st.markdown("""
    Selamat Datang di Web Deteksi Jenis Ikan Air Tawar 🎣

    **Cara Kerja:**
    1. Pergi ke halaman **Fish Recognition**
    2. Unggah gambar ikan
    3. Sistem memproses gambar & menampilkan jenis ikan dan informasi edukatif

    **Keunggulan:**
    - Menggunakan model CNN
    - Mudah digunakan
    - Tersedia statistik deteksi

    Klik halaman **Fish Recognition** untuk memulai.
    """)

# Halaman About
elif app_mode == "Informasi Web":
    st.header("Tentang Proyek")
    st.markdown("""
    Proyek ini mengembangkan sistem klasifikasi jenis ikan air tawar berbasis deep learning (CNN) dan diimplementasikan ke antarmuka web menggunakan Streamlit.

    Diharapkan aplikasi ini dapat dimanfaatkan untuk edukasi, budidaya, dan identifikasi ikan dengan mudah.
    """)

# Halaman Statistik
elif app_mode == "Riwayat Deteksi":
    st.header("📊 Statistik Deteksi Ikan")
    stats = load_statistics()

    if stats:
        sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
        df_stats = pd.DataFrame.from_dict(sorted_stats, orient='index', columns=['Jumlah Deteksi'])
        st.bar_chart(df_stats)
        st.markdown("### Rincian Deteksi:")
        for ikan, jumlah in sorted_stats.items():
            st.write(f"**{ikan}**: {jumlah} kali terdeteksi")

        if st.button("🔄 Reset Statistik"):
            save_statistics({})
            st.success("Statistik berhasil direset.")
    else:
        st.info("Anda belum melakukan deteksi gambar jenis ikan air.")

# Halaman Fish Recognition
elif app_mode == "Fish Recognition":
    st.header("📷 Fish Recognition")

    upload_option = st.radio("Pilih metode input gambar:", ["Unggah Gambar", "Gunakan Kamera"])

    if upload_option == "Unggah Gambar":
        test_image = st.file_uploader("Unggah Gambar Ikan", type=["jpg", "png", "jpeg"])
    else:
        test_image = st.camera_input("Ambil gambar menggunakan kamera")

    if test_image is not None:
        st.image(test_image, caption="Gambar yang Diuji", use_column_width=True)

        if st.button("🔍 Prediksi"):
            st.write("Sedang memproses...")
            st.snow()
            result_index = model_prediction(test_image)

            if result_index < len(class_name):
                fish_name = class_name[result_index]
                st.success(f"Model memprediksi ini adalah ikan **{fish_name}**.")
                update_statistics(fish_name)

                info = fish_info.get(fish_name)
                if info:
                    st.markdown("### ℹ️ Informasi Edukatif")
                    st.write(f"**Nama Ilmiah:** {info['Nama Ilmiah']}")
                    st.write(f"**Ciri-ciri:** {info['Ciri-ciri']}")
                    st.write(f"**Habitat Asli:** {info['Habitat']}")
                    st.write(f"**Kegunaan:** {info['Kegunaan']}")
                else:
                    st.info("Informasi detail belum tersedia.")
            else:
                st.error("Terjadi kesalahan prediksi.")
