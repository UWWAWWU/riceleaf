from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
import segmentation_models_pytorch as smp
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


st.set_page_config(
    page_title="RiceLeaf AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pt"
SEGMENTER_PATH = MODEL_DIR / "segmenter.pt"


st.markdown(
    """
    <style>
      .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 2.1rem 2.4rem; border-radius: 24px;
             background: linear-gradient(120deg,#124f35,#2f8f46,#79b84a);
             color: white; box-shadow: 0 14px 38px rgba(20,80,45,.17);}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2.35rem;}
      .hero p {margin: 0; opacity: .95; font-size: 1.02rem;}
      .result {padding: 1.1rem 1.3rem; border: 1px solid #dce8df;
               border-radius: 16px; background: #f7fbf8; margin-bottom: 1rem;}
      [data-testid="stMetric"] {background: white; border: 1px solid #e0e9e2;
               border-radius: 15px; padding: 1rem;}
    </style>
    <div class="hero">
      <h1>RiceLeaf AI</h1>
      <p>Identifikasi dan visualisasi penyakit tanaman padi berbasis deep learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def stop_if_models_missing():
    missing = [p.name for p in (METADATA_PATH, CLASSIFIER_PATH, SEGMENTER_PATH) if not p.is_file()]
    if missing:
        st.error(
            "File model belum lengkap. Ekstrak isi `riceleaf_bundle.zip` ke folder "
            f"`model/`. File yang belum ditemukan: {', '.join(missing)}"
        )
        st.stop()


stop_if_models_missing()
META = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
CLASSES = META["classes"]
LABELS = META["labels"]
IMAGE_SIZE = int(META["image_size"])
MEAN = META["mean"]
STD = META["std"]
TEMPERATURE = float(META.get("temperature", 1.0))
CONFIDENCE_THRESHOLD = float(META.get("confidence_threshold", 1.01))
SEGMENTATION_THRESHOLD = float(META.get("segmentation_threshold", 0.5))
PADDING_RGB = tuple(META.get("padding_rgb", [124, 116, 104]))


def make_classifier(name):
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    else:
        raise ValueError(f"Arsitektur classifier tidak didukung: {name}")
    return model


def make_segmenter():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    classifier = make_classifier(META["architecture"])
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=True))
    classifier.eval()

    segmenter = make_segmenter()
    segmenter.load_state_dict(torch.load(SEGMENTER_PATH, map_location="cpu", weights_only=True))
    segmenter.eval()
    return classifier, segmenter


def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    return ImageOps.exif_transpose(image).convert("RGB")


def letterbox(image, size=IMAGE_SIZE):
    ratio = min(size / image.width, size / image.height)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), PADDING_RGB)
    left, top = (size - width) // 2, (size - height) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, width, height)


NORMALIZE = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)


def preprocess(image):
    square, box = letterbox(image)
    return NORMALIZE(square), box


def target_layer_for(model, architecture):
    if architecture == "densenet121":
        return model.features[-1]
    if architecture == "convnext_tiny":
        return model.features[-1]
    if architecture == "efficientnet_v2_s":
        return model.features[-1]
    raise ValueError(f"Arsitektur Grad-CAM++ tidak didukung: {architecture}")


def infer(image, classifier, segmenter):
    tensor, (left, top, width, height) = preprocess(image)
    batch = tensor.unsqueeze(0)

    with torch.inference_mode():
        logits = classifier(batch)
        probabilities = (logits / TEMPERATURE).softmax(dim=1)[0].numpy()

    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])

    with GradCAMPlusPlus(
        model=classifier,
        target_layers=[target_layer_for(classifier, META["architecture"])],
    ) as cam:
        heatmap = cam(
            input_tensor=batch,
            targets=[ClassifierOutputTarget(prediction)],
        )[0]

    heatmap = heatmap[top : top + height, left : left + width]
    heatmap = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_LINEAR)
    original = np.asarray(image)
    gradcam = show_cam_on_image(
        original.astype(np.float32) / 255.0,
        heatmap,
        use_rgb=True,
    )

    with torch.inference_mode():
        mask_probability = segmenter(batch).sigmoid()[0, 0].numpy()
    mask_probability = mask_probability[top : top + height, left : left + width]
    mask_probability = cv2.resize(
        mask_probability, image.size, interpolation=cv2.INTER_LINEAR
    )
    mask = mask_probability >= SEGMENTATION_THRESHOLD
    overlay = original.copy()
    overlay[mask] = (
        0.60 * original[mask] + 0.40 * np.array([255, 40, 40])
    ).astype(np.uint8)

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    all_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    # Tampilkan tepi hanya di sekitar area penyakit hasil segmentasi. Sedikit
    # dilatasi mempertahankan tepi lesi yang berada tepat di batas mask.
    disease_region = cv2.dilate(
        mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1
    ).astype(bool)
    canny = np.zeros_like(all_edges)
    canny[disease_region] = all_edges[disease_region]
    return prediction, confidence, probabilities, gradcam, mask, overlay, canny


st.write("")
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Unggah foto tanaman padi")
    uploaded = st.file_uploader(
        "Pilih satu gambar JPG, JPEG, atau PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
with right:
    st.subheader("Informasi model")
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Kelas penyakit", len(CLASSES))
    metric_b.metric("Ukuran input", f"{IMAGE_SIZE} × {IMAGE_SIZE}")
    st.caption("Kelas: Bacterial Blight, Blast, Brown Spot, dan Tungro.")

if uploaded is None:
    st.info("Unggah satu foto tanaman padi untuk memulai prediksi.")
    st.stop()

try:
    image = read_uploaded_image(uploaded)
except Exception:
    st.error("Gambar tidak dapat dibaca. Gunakan file JPG, JPEG, atau PNG yang valid.")
    st.stop()

classifier, segmenter = load_models()
with st.spinner("Menganalisis gambar..."):
    prediction, confidence, probabilities, gradcam, mask, overlay, canny = infer(
        image, classifier, segmenter
    )

st.divider()
st.subheader("Hasil analisis")
result_a, result_b, result_c = st.columns(3)
result_a.metric("Prediksi", LABELS[prediction])
result_b.metric("Skor model", f"{confidence:.2%}")
result_c.metric("Cakupan area prediksi", f"{mask.mean():.2%}")

table = pd.DataFrame({"Kelas": LABELS, "Skor": probabilities})
table = table.sort_values("Skor", ascending=False).reset_index(drop=True)
table["Skor"] = table["Skor"].map(lambda value: f"{value:.2%}")
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Visualisasi")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Gambar asli", "Grad-CAM++", "Mask penyakit", "Overlay", "Canny edge"]
)
with tab1:
    st.image(image, use_container_width=True)
with tab2:
    st.image(gradcam, caption="Area yang memengaruhi keputusan model identifikasi", use_container_width=True)
with tab3:
    st.image(mask.astype(np.uint8) * 255, caption="Mask prediksi penyakit", use_container_width=True)
with tab4:
    st.image(overlay, caption="Area prediksi penyakit ditandai merah", use_container_width=True)
with tab5:
    st.image(
        canny,
        caption="Tepi yang berada di dalam area penyakit hasil segmentasi",
        use_container_width=True,
    )

st.caption("RiceLeaf AI • Model identifikasi dan segmentasi citra tanaman padi")
