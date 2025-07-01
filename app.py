import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import base64
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import json
import pandas as pd # type: ignore

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(main_bg_file, sidebar_bg_file):
    main_bg = get_base64(main_bg_file)
    sidebar_bg = get_base64(sidebar_bg_file)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{main_bg}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("data:image/png;base64,{sidebar_bg}");
            background-position: center; 
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("image/background.png", "image/sidebar.png")

# Load model CNN
def load_model():
    return tf.keras.models.load_model("model_paling_baru.h5")

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
    'Ikan Mas', 'Kakap', 'Lele', 'Mujair', 'Nila', 'Patin'
]

# Informasi edukatif
fish_info = {
    'Bandeng': {'Nama Ilmiah': 'Chanos chanos', 'Ciri-ciri': 'Tubuh memanjang, sisik besar berkilau, ekor bercabang.',
                'Habitat': 'Perairan payau dan laut dangkal.', 'Kegunaan': 'Konsumsi, budidaya tambak.'},
    'Bawal': {'Nama Ilmiah': 'Colossoma macropomum', 'Ciri-ciri': 'Badan lebar dan pipih, warna keperakan dengan sirip gelap.',
              'Habitat': 'Sungai dan danau air tawar.', 'Kegunaan': 'Konsumsi, sering dibudidayakan.'},
    'Cupang': {'Nama Ilmiah': 'Betta splendens', 'Ciri-ciri': 'Warna cerah, sirip panjang mengembang.',
               'Habitat': 'Air tenang seperti kolam dan selokan.', 'Kegunaan': 'Ikan hias, kadang aduan.'},
    'Gabus': {'Nama Ilmiah': 'Channa striata', 'Ciri-ciri': 'Tubuh panjang, kepala seperti ular.',
              'Habitat': 'Sungai, rawa, danau.', 'Kegunaan': 'Konsumsi, pengobatan luka tradisional.'},
    'Gurame': {'Nama Ilmiah': 'Osphronemus goramy', 'Ciri-ciri': 'Badan pipih, sisik kasar, sirip panjang.',
               'Habitat': 'Sungai dan kolam air tenang.', 'Kegunaan': 'Konsumsi favorit.'},
    'Ikan Mas': {'Nama Ilmiah': 'Cyprinus carpio', 'Ciri-ciri': 'Tubuh besar, bersisik kuning keemasan.',
                 'Habitat': 'Danau, kolam, sungai lambat.', 'Kegunaan': 'Konsumsi, lomba mancing.'},
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
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Pilih Halaman", ["Home", "Informasi Web", "Riwayat Deteksi", "Fish Recognition"])


# Halaman Home
if app_mode == "Home":
    ##st.header("🎣 DETEKSI JENIS IKAN AIR TAWAR")
    st.image("image/judul.png", use_column_width=True)
    st.markdown("""
    🐟 Selamat Datang di Deteksi Ikan Air Tawar

    **Kenali Ikan Air Tawar dengan Mudah dan Menyenangkan!**
    Air tawar menyimpan banyak kekayaan hayati—termasuk beragam jenis ikan yang unik dan menarik. Tapi… apakah kamu 
    bisa membedakan ikan nila, gurame, atau lele hanya dari fotonya? Nah, di sinilah peran website ini!
    
    Kami hadir untuk membantu kamu mengenali ikan air tawar hanya dengan mengunggah gambar. 
    Sistem ini menggunakan teknologi CNN (Convolutional Neural Network) yang bisa mendeteksi dan mengenali 
    jenis ikan secara otomatis dan akurat.

    **Apa Saja yang Bisa Kamu Lakukan di Sini?:**
    - 🎯 Deteksi Cepat Jenis Ikan Air Tawar
    - 📚 Dapatkan Informasi Edukatif tentang Ikan
    - 🧠 Belajar Sambil Praktik, Seru dan Interaktif!

    Yuk, mulai eksplorasi dunia ikan air tawar bersama teknologi! Klik **Fish Recognition** dan 
    unggah gambar ikanmu atau 
    kamu bisa mengambil gambar ikan mu secara langsung 🎉
    """)

# Halaman About
elif app_mode == "Informasi Web":
    #st.header("Tentang Proyek")
    st.image("image/tentang_web.png", use_column_width=True)
    st.markdown("""
    **ℹ️ Tentang Website Ini**
    **Mengenal Ikan Air Tawar Lewat Teknologi**
    
    Website ini dibuat sebagai sarana edukatif dan interaktif untuk membantu pengguna mengenali 
    jenis-jenis ikan air tawar melalui gambar. 
    Dengan menggabungkan ilmu biologi dan kecerdasan buatan, kami ingin mempermudah proses identifikasi ikan yang 
    biasanya memerlukan pengetahuan khusus.
    
    **🎯 Tujuan Kami**
    - Membantu pelajar, mahasiswa, dan masyarakat umum mengenal ikan air tawar dengan cara yang praktis
    - Meningkatkan kesadaran akan keanekaragaman hayati air tawar
    - Mendorong pemanfaatan teknologi dalam bidang perikanan dan pendidikan
    
    **⚙️ Teknologi yang Digunakan**
    Website ini menggunakan **Convolutional Neural Network (CNN)**, salah satu metode dalam Deep Learning, untuk mengenali pola visual pada gambar ikan. 
    Dengan data pelatihan yang cukup, sistem ini dapat memprediksi jenis ikan dengan tingkat akurasi yang baik.

    
    
    """)

# Halaman Statistik
elif app_mode == "Riwayat Deteksi":
    #st.header("📊 Statistik Deteksi Ikan")
    st.image("image/riwayat_deteksi.png",  use_column_width=True)
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
    #st.header("📷 Fish Recognition")
    st.image("image/fish_recog.png",  use_column_width=True)

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
