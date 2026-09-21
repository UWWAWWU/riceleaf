from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "riceleaf.joblib"

CLASS_NAMES = {
    "blast": "Blast",
    "blight": "Blight",
    "tungro": "Tungro",
}

st.set_page_config(
    page_title="RiceLeaf AI",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7faf7; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #123d29 0%, #237a45 65%, #6fae4d 100%);
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(18, 61, 41, .16);
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p { margin: .55rem 0 0; color: #e9f5eb; font-size: 1.05rem; }
    .result-card {
        padding: 1.25rem 1.4rem;
        border: 1px solid #d9e8dc;
        border-radius: 18px;
        background: white;
        margin-bottom: 1rem;
    }
    .result-label { color: #55705d; font-size: .9rem; margin-bottom: .2rem; }
    .result-name { color: #174d2f; font-size: 2rem; font-weight: 750; }
    .small-note { color: #607266; font-size: .9rem; }
    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #9abca2;
        border-radius: 18px;
        padding: .6rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfeae1;
        border-radius: 16px;
        padding: .8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH.name}")

    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "classes", "feature_names", "image_size"}
    missing = required_keys.difference(bundle)
    if missing:
        raise ValueError(f"Isi model tidak lengkap: {sorted(missing)}")
    if len(bundle["feature_names"]) != 42:
        raise ValueError("Model tidak menggunakan 42 fitur yang diharapkan.")
    return bundle


def resize_with_padding(image, target_size):
    if image is None or image.size == 0:
        raise ValueError("Gambar tidak valid.")

    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x_start = (target_width - new_width) // 2
    y_start = (target_height - new_height) // 2
    canvas[
        y_start:y_start + new_height,
        x_start:x_start + new_width,
    ] = resized
    return canvas


def create_leaf_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid_hue = (hue <= 100) | (hue >= 170)
    mask = (
        valid_hue
        & (saturation >= 30)
        & (value >= 20)
        & (value <= 245)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError(
            "Objek daun tidak ditemukan. Gunakan foto daun yang jelas dan "
            "tidak terlalu jauh."
        )

    leaf_contour = max(contours, key=cv2.contourArea)
    minimum_area = image.shape[0] * image.shape[1] * 0.002
    if cv2.contourArea(leaf_contour) < minimum_area:
        raise ValueError("Area daun terlalu kecil pada gambar.")

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [leaf_contour], -1, 255, cv2.FILLED)
    return clean_mask, leaf_contour


def preprocess_diseased_leaf(prepared_image, leaf_mask):
    hsv = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 100, 20], dtype=np.uint8)
    upper = np.array([20, 255, 200], dtype=np.uint8)
    disease_mask = cv2.inRange(hsv, lower, upper)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )
    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    segmented = cv2.bitwise_and(
        prepared_image,
        prepared_image,
        mask=disease_mask,
    )
    grayscale = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(
        grayscale,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    return blurred, disease_mask


def run_image_preprocessing(image, image_size):
    prepared = resize_with_padding(image, image_size)
    leaf_mask, leaf_contour = create_leaf_mask(prepared)
    preprocessed, disease_mask = preprocess_diseased_leaf(
        prepared,
        leaf_mask,
    )
    canny_edges = cv2.Canny(preprocessed, 50, 150)
    marked = prepared.copy()
    marked[disease_mask != 0] = (0, 0, 255)
    return {
        "prepared_image": prepared,
        "leaf_mask": leaf_mask,
        "leaf_contour": leaf_contour,
        "preprocessed_image": preprocessed,
        "disease_mask": disease_mask,
        "canny_edges": canny_edges,
        "marked_image": marked,
    }


def normalized_histogram(image, channel, mask, bins, value_range):
    histogram = cv2.calcHist(
        [image],
        [channel],
        mask,
        [bins],
        value_range,
    )
    return cv2.normalize(
        histogram,
        None,
        alpha=1,
        norm_type=cv2.NORM_L1,
    ).flatten().astype(np.float32)


def extract_features(image, image_size):
    artifacts = run_image_preprocessing(image, image_size)
    prepared = artifacts["prepared_image"]
    leaf_mask = artifacts["leaf_mask"]
    leaf_contour = artifacts["leaf_contour"]
    disease_mask = artifacts["disease_mask"]
    preprocessed = artifacts["preprocessed_image"]
    edges = artifacts["canny_edges"]

    hsv = cv2.cvtColor(prepared, cv2.COLOR_BGR2HSV)
    color_features = np.concatenate(
        [
            normalized_histogram(hsv, 0, leaf_mask, 8, [0, 180]),
            normalized_histogram(hsv, 1, leaf_mask, 8, [0, 256]),
            normalized_histogram(hsv, 2, leaf_mask, 8, [0, 256]),
        ]
    )

    image_area = float(prepared.shape[0] * prepared.shape[1])
    image_perimeter = float(2 * (prepared.shape[0] + prepared.shape[1]))
    leaf_features = np.array(
        [
            cv2.contourArea(leaf_contour) / image_area,
            cv2.arcLength(leaf_contour, True) / image_perimeter,
        ],
        dtype=np.float32,
    )

    disease_contours, _ = cv2.findContours(
        disease_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    disease_contours = [
        contour
        for contour in disease_contours
        if cv2.contourArea(contour) >= 2
    ]
    disease_areas = [cv2.contourArea(contour) for contour in disease_contours]
    disease_features = np.array(
        [
            np.count_nonzero(disease_mask) / image_area,
            len(disease_contours),
            (max(disease_areas) / image_area) if disease_areas else 0.0,
        ],
        dtype=np.float32,
    )

    intensity_histogram = normalized_histogram(
        preprocessed,
        0,
        disease_mask,
        8,
        [0, 256],
    )
    disease_pixels = preprocessed[disease_mask > 0]
    if disease_pixels.size:
        disease_mean = float(disease_pixels.mean() / 255.0)
        disease_std = float(disease_pixels.std() / 255.0)
        laplacian = cv2.Laplacian(preprocessed, cv2.CV_32F)
        laplacian_variance = float(
            laplacian[disease_mask > 0].var() / (255.0 ** 2)
        )
    else:
        disease_mean = 0.0
        disease_std = 0.0
        laplacian_variance = 0.0

    intensity_statistics = np.array(
        [disease_mean, disease_std, laplacian_variance],
        dtype=np.float32,
    )
    edge_contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    edge_features = np.array(
        [
            np.count_nonzero(edges) / image_area,
            len(edge_contours),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            color_features,
            leaf_features,
            disease_features,
            intensity_histogram,
            intensity_statistics,
            edge_features,
        ]
    ).astype(np.float32)
    if features.shape != (42,) or not np.isfinite(features).all():
        raise ValueError("Ekstraksi fitur menghasilkan data yang tidak valid.")
    return features, artifacts


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File tidak dapat dibaca sebagai gambar.")
    return image


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


st.markdown(
    """
    <section class="hero">
      <h1>RiceLeaf AI</h1>
      <p>Klasifikasi penyakit daun padi menggunakan Random Forest, fitur warna,
      area penyakit, dan Canny edge.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    model_bundle = load_model_bundle()
except Exception as error:
    st.error(f"Model gagal dimuat: {error}")
    st.stop()

upload_column, information_column = st.columns([1.45, 1], gap="large")
with upload_column:
    st.subheader("Unggah foto daun")
    uploaded_file = st.file_uploader(
        "Pilih gambar JPG, JPEG, atau PNG",
        type=["jpg", "jpeg", "png"],
        help="Gunakan foto daun padi yang jelas dengan pencahayaan cukup.",
    )
with information_column:
    st.subheader("Informasi model")
    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Akurasi uji", "94%")
    metric_2.metric("Jumlah fitur", "42")
    st.caption(
        "Kelas model: Blast, Blight, dan Tungro. Hasil merupakan prediksi "
        "model dan bukan pengganti pemeriksaan ahli pertanian."
    )

if uploaded_file is None:
    st.info("Unggah satu foto daun padi untuk memulai klasifikasi.")
    st.stop()

try:
    original_image = decode_uploaded_image(uploaded_file)
    features, artifacts = extract_features(
        original_image,
        tuple(model_bundle["image_size"]),
    )
    model = model_bundle["model"]
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    classes = [str(class_name) for class_name in model.classes_]
    prediction_index = int(np.argmax(probabilities))
    predicted_class = classes[prediction_index]
    predicted_score = float(probabilities[prediction_index])
except Exception as error:
    st.error(f"Gambar tidak dapat diproses: {error}")
    st.stop()

leaf_pixels = max(1, int(np.count_nonzero(artifacts["leaf_mask"])))
disease_pixels = int(np.count_nonzero(artifacts["disease_mask"]))
disease_coverage = disease_pixels / leaf_pixels

st.divider()
result_column, score_column = st.columns([1, 1.25], gap="large")
with result_column:
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">Hasil klasifikasi</div>
          <div class="result-name">{CLASS_NAMES.get(predicted_class, predicted_class.title())}</div>
          <div class="small-note">Skor model: {predicted_score:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_metric_1, result_metric_2 = st.columns(2)
    result_metric_1.metric("Keyakinan model", f"{predicted_score:.2%}")
    result_metric_2.metric("Cakupan area", f"{disease_coverage:.2%}")

with score_column:
    st.subheader("Skor setiap kelas")
    score_data = pd.DataFrame(
        {
            "Kelas": [CLASS_NAMES.get(name, name.title()) for name in classes],
            "Skor": probabilities,
        }
    ).sort_values("Skor", ascending=False)
    st.bar_chart(score_data.set_index("Kelas"), color="#2b8a4b")
    score_data["Skor"] = score_data["Skor"].map(lambda value: f"{value:.2%}")
    st.dataframe(score_data, hide_index=True, use_container_width=True)

st.subheader("Hasil pemrosesan citra")
image_columns = st.columns(5, gap="small")
visualizations = [
    (bgr_to_rgb(original_image), "Gambar asli"),
    (bgr_to_rgb(artifacts["prepared_image"]), "Normalisasi"),
    (artifacts["preprocessed_image"], "Preprocessing"),
    (artifacts["canny_edges"], "Canny edge"),
    (bgr_to_rgb(artifacts["marked_image"]), "Area penyakit"),
]
for column, (image, caption) in zip(image_columns, visualizations):
    column.image(image, caption=caption, use_container_width=True)

with st.expander("Cara kerja model"):
    st.write(
        "Gambar dinormalisasi menjadi 256 × 256 piksel. Sistem memisahkan "
        "objek daun, mendeteksi area berwarna yang terindikasi penyakit, "
        "menghasilkan Canny edge, lalu mengekstrak 42 fitur. Random Forest "
        "menggunakan seluruh fitur tersebut untuk menentukan kelas prediksi."
    )

st.caption("RiceLeaf AI · Random Forest · Accuracy testing 94%")
