import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image
from deepforest import main
from mobile_sam import sam_model_registry, SamPredictor

CHECKPOINT_PATH = "mobile_sam.pt"
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

@st.cache_resource
def load_models():
    if not os.path.exists(CHECKPOINT_PATH):
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    m = main.deepforest()
    m.load_model(model_name="weecology/deepforest-tree", revision="main")
    sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
    predictor = SamPredictor(sam)
    return m, predictor

m, predictor = load_models()

def analyze_forest(image, confidence_threshold, meters_per_pixel):
    detections = m.predict_image(image=image)
    confident = detections[detections['score'] >= confidence_threshold]
    predictor.set_image(image)
    total_area = 0
    overlay = image.copy()
    for _, row in confident.iterrows():
        box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0]
        total_area += mask.sum() * (meters_per_pixel ** 2)
        colored = np.zeros_like(overlay)
        colored[mask] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
    img_h, img_w = image.shape[0], image.shape[1]
    ground_area_m2 = (img_h * meters_per_pixel) * (img_w * meters_per_pixel)
    canopy_pct = (total_area / ground_area_m2) * 100 if ground_area_m2 > 0 else 0
    summary = {
        "trees_detected": int(len(confident)),
        "total_canopy_area_m2": round(total_area, 2),
        "canopy_cover_pct": round(canopy_pct, 1),
    }
    return overlay, summary

st.set_page_config(page_title="Tree Crown Detection & Canopy Area Estimator")
st.title("Tree Crown Detection & Canopy Area Estimator")
st.markdown("""
### Known limitations — read before trusting the output
- **Undercounts dense, closed canopy.** Adjacent trees are often merged into a single detection in overlapping-crown forest.
- **No single "correct" confidence threshold** — lower values catch more real trees but add false positives; higher values miss real trees in dense scenes.
- **Resolution must be accurate** — a wrong meters/pixel value silently produces a wrong area.
- **Segmentation uses MobileSAM**, a compressed model chosen to fit free-tier hosting memory limits — mask precision may be lower than the full SAM model tested during development.
- **Not validated against ground-truth counts.**
""")

uploaded = st.file_uploader("Upload forest imagery (RGB)", type=["png", "jpg", "jpeg"])
threshold = st.slider("Detection confidence threshold", 0.05, 0.9, 0.3, 0.05)
resolution = st.number_input("Image resolution (meters per pixel)", value=0.10, format="%.3f")

if uploaded is not None:
    image = np.array(Image.open(uploaded).convert("RGB"))
    with st.spinner("Running detection and segmentation..."):
        overlay, summary = analyze_forest(image, threshold, resolution)
    st.image(overlay, caption="Detected crowns (green = segmented)", use_column_width=True)
    st.subheader("Results")
    st.write(f"**Trees detected:** {summary['trees_detected']}")
    st.write(f"**Total canopy area:** {summary['total_canopy_area_m2']} m²")
    st.write(f"**Canopy cover:** {summary['canopy_cover_pct']}%")
    st.write(f"Threshold: {threshold} | Resolution: {resolution} m/pixel")
