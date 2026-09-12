import gradio as gr
import numpy as np
import cv2
import os
import urllib.request
from deepforest import main
from segment_anything import sam_model_registry, SamPredictor

# --- Download SAM checkpoint if not already present ---
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)

# --- Load models once at startup, not per-request ---
m = main.deepforest()
m.load_model(model_name="weecology/deepforest-tree", revision="main")

sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)

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
    summary = (
        f"Trees detected: {len(confident)}\n"
        f"Total canopy area: {round(total_area, 2)} m²\n"
        f"Canopy cover: {round(canopy_pct, 1)}%\n"
        f"Threshold: {confidence_threshold} | Resolution: {meters_per_pixel} m/pixel"
    )
    return overlay, summary

LIMITATIONS = """
### Known limitations — read before trusting the output
- **Undercounts dense, closed canopy.** In overlapping-crown forest, adjacent trees are often merged into a single detection. Tree count and canopy area are more reliable in open/sparse canopy than dense forest.
- **No single "correct" confidence threshold.** Lower thresholds catch more real trees but add false positives on bare ground/shadow; higher thresholds miss real trees in dense scenes. Move the slider and compare — don't trust one number.
- **Resolution must be accurate.** You must know your image's actual meters/pixel value. A wrong value silently produces a wrong area — this tool cannot detect that for you.
- **Not validated against ground-truth counts.** Numbers shown are model output, not field-verified measurements.
"""

demo = gr.Interface(
    fn=analyze_forest,
    inputs=[
        gr.Image(type="numpy", label="Forest imagery (RGB)"),
        gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Detection confidence threshold"),
        gr.Number(value=0.1, label="Image resolution (meters per pixel)")
    ],
    outputs=[
        gr.Image(label="Detected crowns (green = segmented)"),
        gr.Textbox(label="Results")
    ],
    title="Tree Crown Detection & Canopy Area Estimator",
    description=LIMITATIONS
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))