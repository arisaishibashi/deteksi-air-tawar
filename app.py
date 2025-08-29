import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import base64
import json
from typing import Tuple, Dict, Any, List

import streamlit as st  # type: ignore
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import pandas as pd  # type: ignore

# ============== KONSTAN / KONFIGURASI ==============
MODEL_PATH = "model_24juli.h5"
STAT_FILE = "statistik_deteksi.json"
UNKNOWN_CLASS = "Model tidak mempelajari gambar ini"
CONF_THRESHOLD = 0.60  # Kalibrasi berdasarkan validasi Anda

# Path gambar UI
IMG_BG_MAIN = "image/background.png"
IMG_BG_SIDEBAR = "image/sidebar.png"
IMG_JUDUL = "image/judul.png"
IMG_TENTANG = "image/tentang_web.png"
IMG_RIWAYAT = "image/riwayat_deteksi.png"
IMG_RECOG = "image/fish_recog.png"

# Daftar nama kelas ikan (harus konsisten dengan urutan output model)
class_name: List[str] = [
    'Bandeng', 'Bawal', 'Cupang', 'Gabus', 'Gurame',
    'Ikan Mas', 'Kakap', 'Lele', 'Model tidak mempelajari gambar ini',
    'Mujair', 'Nila', 'Patin'
]

# Informasi edukatif
fish_info: Dict[str, Dict[str, str]] = {
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
    # UNKNOWN_CLASS tidak perlu entry edukatif
}

# ============== UTILITAS UI ==============
def get_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(main_bg_file: str, sidebar_bg_file: str) -> None:
    try:
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
    except Exception as e:
        st.warning(f"Gagal memasang background: {e}")

# ============== MODEL & PREDIKSI ==============
@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari '{MODEL_PATH}': {e}")
        st.stop()

model = load_model()

def preprocess_image(image_data, target_size: Tuple[int, int]=(224, 224)) -> np.ndarray:
    """Baca gambar -> RGB -> resize -> skala 0-1 -> tambah dimensi batch"""
    image = Image.open(image_data).convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def model_prediction(image_data) -> Tuple[int, float, np.ndarray]:
    """
    Return:
      - top_index (int)
      - top_confidence (float)
      - probs penuh (np.ndarray shape (num_classes,))
    """
    x = preprocess_image(image_data)
    preds = model.predict(x, verbose=0)  # (1, num_classes)
    preds = preds[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    return top_idx, top_conf, preds

# ============== STATISTIK ==============
def load_statistics() -> Dict[str, int]:
    if os.path.exists(STAT_FILE):
        try:
            with open(STAT_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # pastikan int
                    return {k: int(v) for k, v in data.items()}
        except Exception:
            pass
    return {}

def save_statistics(stats: Dict[str, int]) -> None:
    try:
        with open(STAT_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        st.error(f"Gagal menyimpan statistik: {e}")

def update_statistics(ikan_nama: str) -> None:
    stats = load_statistics()
    stats[ikan_nama] = int(stats.get(ikan_nama, 0)) + 1
    save_statistics(stats)

# ============== APP START ==============
st.set_page_config(page_title="Deteksi Ikan Air Tawar", page_icon="🐟", layout="centered")
set_background(IMG_BG_MAIN, IMG_BG_SIDEBAR)

# Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Informasi Web", "Riwayat Deteksi", "Fish Recognition"]
)

# ============== HALAMAN: HOME ==============
if app_mode == "Home":
    if os.path.exists(IMG_JUDUL):
        st.image(IMG_JUDUL, use_column_width=True)

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
        Yuk, mulai eksplorasi dunia ikan air tawar bersama teknologi! Klik <b>Fish Recognition</b> dan unggah gambar ikanmu atau ambil foto langsung 🎉
        </span>
    </div>
    """, unsafe_allow_html=True)

# ============== HALAMAN: INFORMASI WEB ==============
elif app_mode == "Informasi Web":
    if os.path.exists(IMG_TENTANG):
        st.image(IMG_TENTANG, use_column_width=True)
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
            ikan yang biasanya memerlukan pengetahuan khusus.<br><br>
            <b>⚙️ Teknologi yang Digunakan</b><br>
            Website ini menggunakan <b>Convolutional Neural Network (CNN)</b>, salah satu metode dalam Deep Learning 
            untuk mengenali pola visual pada gambar ikan.<br>
            Dengan data pelatihan yang cukup, sistem ini dapat memprediksi jenis ikan dengan tingkat akurasi yang baik.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ============== HALAMAN: RIWAYAT DETEKSI ==============
elif app_mode == "Riwayat Deteksi":
    if os.path.exists(IMG_RIWAYAT):
        st.image(IMG_RIWAYAT, use_column_width=True)

    stats = load_statistics()

    if stats:
        # Convert ke list of dict agar aman diproses DataFrame
        list_stats = [{"Nama Ikan": ikan, "Jumlah Deteksi": jumlah} for ikan, jumlah in stats.items()]
        df_stats = pd.DataFrame(list_stats)
        df_stats = df_stats.sort_values(by="Jumlah Deteksi", ascending=False).reset_index(drop=True)
        df_stats.index = df_stats.index + 1

        # Bar chart
        try:
            st.bar_chart(df_stats.set_index("Nama Ikan"))
        except Exception as e:
            st.warning(f"Gagal memuat grafik: {e}")

        st.markdown("### Rincian Deteksi:")
        st.table(df_stats)

        if st.button("🔄 Reset Deteksi"):
            save_statistics({})
            st.success("Deteksi berhasil direset.")
    else:
        st.info("Anda belum melakukan deteksi gambar jenis ikan air tawar.")

# ============== HALAMAN: FISH RECOGNITION ==============
elif app_mode == "Fish Recognition":
    if os.path.exists(IMG_RECOG):
        st.image(IMG_RECOG, use_column_width=True)

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

            try:
                result_index, confidence, probs = model_prediction(test_image)
            except Exception as e:
                st.error(f"Gagal memproses gambar: {e}")
                st.stop()

            # Nama kelas hasil prediksi top-1
            try:
                fish_name = class_name[result_index]
            except Exception:
                st.error("Output model tidak sesuai jumlah kelas.")
                st.stop()

            # Logika keputusan (gerbang ganda: label unknown ATAU confidence rendah)
            is_unknown_by_label = (fish_name == UNKNOWN_CLASS)
            is_unknown_by_threshold = (confidence < CONF_THRESHOLD)

            if is_unknown_by_label or is_unknown_by_threshold:
                st.error("Maaf, gambar tidak dikenali atau model belum mempelajari gambar ini.")
                update_statistics(UNKNOWN_CLASS)
            else:
                st.success(f"Model memprediksi ini adalah ikan **{fish_name}** (keyakinan {confidence:.2%})")
                update_statistics(fish_name)

                # Tampilkan Top-3 prediksi (opsional, untuk transparansi)
                try:
                    topk = min(3, len(class_name))
                    top_idx = np.argsort(probs)[-topk:][::-1]
                    top_rows = [{"Kelas": class_name[i], "Probabilitas": f"{probs[i]:.2%}"} for i in top_idx]
                    st.markdown("**Top-3 Prediksi:**")
                    st.table(pd.DataFrame(top_rows))
                except Exception:
                    pass

                # Informasi edukatif
                info = fish_info.get(fish_name)
                if info:
                    st.markdown("### ℹ️ Informasi Edukatif")
                    st.write(f"**Nama Ilmiah:** {info['Nama Ilmiah']}")
                    st.write(f"**Ciri-ciri:** {info['Ciri-ciri']}")
                    st.write(f"**Habitat Asli:** {info['Habitat']}")
                    st.write(f"**Kegunaan:** {info['Kegunaan']}")
                else:
                    st.info("Informasi detail belum tersedia.")
