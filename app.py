import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import base64
import streamlit as st  # type: ignore
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import json
import pandas as pd  # type: ignore

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

# Load models
def load_model1():
    return tf.keras.models.load_model("model_paling_baru.h5")  # model ikan air tawar

def load_model2():
    return tf.keras.models.load_model("model_07_juli.h5")  # model gambar lain

model1 = load_model1()
model2 = load_model2()

# Label model 1 (ikan air tawar)
class_name = [
    'Bandeng', 'Bawal', 'Cupang', 'Gabus', 'Gurame',
    'Ikan Mas', 'Kakap', 'Lele', 'Mujair', 'Nila', 'Patin'
]
# Label model 2
class_name2 = [ 
    'Bandeng', 'Bawal', 'Cupang', 'Gabus', 'Gurame',
    'Ikan Mas', 'Kakap', 'Lele', 'Model tidak mempelajari gambar ini', 'Nila', 'Patin'
]

def model1_prediction(image_data):
    image = Image.open(image_data).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model1.predict(image_array)
    pred_index = int(np.argmax(prediction))
    pred_conf = float(np.max(prediction))
    return pred_index, pred_conf

def model2_prediction(image_data, threshold=0.7):
    image = Image.open(image_data).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model2.predict(image_array)
    pred_index = int(np.argmax(prediction))
    pred_conf = float(np.max(prediction))
    if pred_conf < threshold:
        return None, pred_conf
    else:
        return pred_index, pred_conf

# Info ikan air tawar
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
    st.image("image/judul.png", use_column_width=True)
    st.markdown("""
    <div style="
        position: relative;
        background: #b6d8fa;
        padding: 32px 28px 32px 28px;
        border-radius: 24px;
        box-shadow: 0 4px 24px rgba(80,80,80,0.08);
        margin: 0 auto 32px auto;
        max-width: 680px;
        min-width: 200px;
        ">
        <span style="font-size:1.18rem; color:#1a2444; font-weight:500;">
        🐟 <b>Selamat Datang di Deteksi Ikan Air Tawar</b><br><br>
        Kenali Ikan Air Tawar dengan Mudah dan Menyenangkan! Air tawar menyimpan banyak kekayaan hayati—termasuk 
        beragam jenis ikan yang unik dan menarik. Tapi… apakah kamu 
        bisa membedakan ikan nila, gurame, atau lele hanya dari fotonya? Nah, di sinilah peran website ini!<br><br>
        Kami hadir untuk membantu kamu mengenali ikan air tawar hanya dengan mengunggah gambar.<br>
        Sistem ini menggunakan teknologi CNN (Convolutional Neural Network) yang bisa mendeteksi dan mengenali 
        jenis ikan air tawar.<br><br>
        <b>Apa Saja yang Bisa Kamu Lakukan di Sini?:</b>
        <ul>
            <li>🎯 Deteksi Cepat Jenis Ikan Air Tawar</li>
            <li>📚 Dapatkan Informasi Edukatif tentang Ikan</li>
            <li>🧠 Belajar Sambil Praktik, Seru dan Interaktif!</li>
        </ul>
        Yuk, mulai eksplorasi dunia ikan air tawar bersama teknologi! Klik <b>Fish Recognition</b> dan unggah gambar ikanmu atau kamu bisa mengambil gambar ikan mu secara langsung 🎉
        </span>
    </div>
    """, unsafe_allow_html=True)

# Halaman About
elif app_mode == "Informasi Web":
    st.image("image/tentang_web.png", use_column_width=True)
    st.markdown("""
    <div style="
        position: relative;
        background: #b6d8fa;
        padding: 32px 28px 32px 28px;
        border-radius: 24px;
        box-shadow: 0 4px 24px rgba(80,80,80,0.08);
        margin: 0 auto 32px auto;
        max-width: 680px;
        min-width: 200px;
    ">
        <span style="font-size:1.13rem; color:#1a2444;">
            <b>ℹ️ Tentang Website Ini: Mengenal Ikan Air Tawar Lewat Teknologi</b><br><br>
            Website ini dibuat sebagai sarana edukatif dan interaktif untuk membantu pengguna mengenali 
            jenis-jenis ikan air tawar melalui gambar.<br>
            Dengan menggabungkan ilmu biologi dan kecerdasan buatan, kami ingin mempermudah proses identifikasi 
            ikan yang 
            biasanya memerlukan pengetahuan khusus.<br><br>
            <b>⚙️ Teknologi yang Digunakan</b><br>
            Website ini menggunakan <b>Convolutional Neural Network (CNN)</b>, salah satu metode dalam Deep Learning 
            untuk mengenali pola visual pada gambar ikan.<br>
            Dengan data pelatihan yang cukup, sistem ini dapat memprediksi jenis ikan dengan tingkat akurasi yang baik.
        </span>
    </div>
    """, unsafe_allow_html=True)

# Halaman Statistik
elif app_mode == "Riwayat Deteksi":
    st.image("image/riwayat_deteksi.png",  use_column_width=True)
    stats = load_statistics()

    if stats:
        # Convert ke list of dict agar aman diproses DataFrame
        list_stats = [
            {"Nama Ikan": ikan, "Jumlah Deteksi": jumlah}
            for ikan, jumlah in stats.items()
        ]
        df_stats = pd.DataFrame(list_stats)
        df_stats = df_stats.sort_values(by="Jumlah Deteksi", ascending=False).reset_index(drop=True)
        df_stats.index = df_stats.index + 1

        st.bar_chart(df_stats.set_index("Nama Ikan"))
        st.markdown("### Rincian Deteksi:")
        st.table(df_stats)

        if st.button("🔄 Reset Deteksi"):
            save_statistics({})
            st.success("Deteksi berhasil direset.")
    else:
        st.info("Anda belum melakukan deteksi gambar jenis ikan air tawar.")

# Halaman Fish Recognition
elif app_mode == "Fish Recognition":
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
            st.balloons()
            result1, conf1 = model1_prediction(test_image)
            fish_name = class_name[result1]
            THRESHOLD = 0.75

            if (fish_name in fish_info) and (conf1 > THRESHOLD):
                st.success(f"Gambar yang diunggah adalah ikan air tawar **{fish_name}** (confidence: {conf1:.2f}).")
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
                # Confidence rendah/meragukan, langsung model2
                result2, conf2 = model2_prediction(test_image, threshold=0.7)
                if result2 is not None and result2 < len(class_name2):
                    nonfish_name = class_name2[result2]
                    if nonfish_name == "Model tidak mempelajari gambar ini":
                        st.warning("Model tidak mempelajari gambar ini.")
                        update_statistics(nonfish_name)
                        st.info("Maaf, model kami belum mempelajari gambar ini.")
                    else:
                        st.warning(f"Gambar yang diunggah adalah **{nonfish_name}** (confidence: {conf2:.2f}).")
                        update_statistics(nonfish_name)
                        info = fish_info.get(nonfish_name)
                        if info:
                            st.markdown("### ℹ️ Informasi Edukatif")
                            st.write(f"**Nama Ilmiah:** {info['Nama Ilmiah']}")
                            st.write(f"**Ciri-ciri:** {info['Ciri-ciri']}")
                            st.write(f"**Habitat Asli:** {info['Habitat']}")
                            st.write(f"**Kegunaan:** {info['Kegunaan']}")
                else:
                    st.error("Gambar tidak dikenali. Silakan gunakan gambar lain.")
