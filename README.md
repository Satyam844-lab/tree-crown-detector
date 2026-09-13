# Tree Crown Detection & Canopy Area Estimator

Detects individual tree crowns in high-resolution RGB aerial/satellite imagery, segments each detected crown, and estimates canopy area and percent cover. Built to explore what an automated, honestly-scoped canopy assessment tool can and can't do — relevant to use cases like carbon-market canopy verification, where overconfident numbers are worse than admitted uncertainty.

**Live demo:** https://tree-crown-detector-ejhvugksijnxndvpv3bpwh.streamlit.app/

---

## What it does

1. Upload an RGB forest image (ideally ~10cm/pixel resolution or finer)
2. **DeepForest** (a pretrained RetinaNet model, `weecology/deepforest-tree`) detects candidate tree crowns as bounding boxes, each with a confidence score
3. Boxes below a user-adjustable confidence threshold are discarded
4. **MobileSAM** segments each remaining box into a precise crown-shaped mask (not just the rectangular box)
5. Each mask's pixel area is converted to real-world area using a user-supplied meters-per-pixel value, and summed into a total canopy area and cover percentage

The confidence threshold and resolution are exposed as adjustable inputs, not hardcoded — see *Why no single threshold* below for why.

---

## What we actually found (tested, not assumed)

**Confidence drops sharply in dense canopy.** On an open/sparse scrub test image, the top detection confidence was 0.80. On a dense, closed-canopy image (NEON airborne RGB, 10cm/pixel), the top confidence was only 0.69, with most detections between 0.17–0.52. The model is visibly less certain about anything in dense canopy — this isn't noise, it's a real, repeatable pattern.

**No single confidence threshold works across scene types.** Tested on the same dense-canopy image:

| Threshold | Trees kept | Canopy area | Canopy cover | Verdict |
|---|---|---|---|---|
| 0.5 | 5 / 59 | 95.4 m² | 3.8% | Clearly wrong — image is visually near-full canopy |
| 0.3 | 44 / 59 | 1040.2 m² | 41.6% | Plausible, likely still an undercount |
| 0.15 | 59 / 59 | 1248.5 m² | 49.9% | Recovers more real trees, likely reintroduces some false positives |

On the open scrub image, by contrast, 0.5 correctly filtered out false positives on bare dirt/shadow without losing real detections. **The "right" threshold depends entirely on canopy density — there is no default that's correct everywhere.** The app defaults to 0.3 as a reasoned compromise, biased toward not missing real canopy (an undercount is arguably the costlier error for something like carbon accounting) — but the slider is there so a user isn't stuck trusting one hidden number.

**Dense canopy produces merged-crown detections.** Visual inspection of bounding boxes on the dense-canopy image showed boxes spanning what were clearly 2–4 distinct treetops, alongside real coverage gaps (canopy with no box at all). This is a known, structural limitation of box-based crown detectors in closed canopy, not a bug — there's no visible boundary between adjacent crowns for the model to use. One segmented "crown" came back at 62 m² (~9m radius) against a typical single-crown size of 15–23 m² in the same image — almost certainly a merged cluster, not one tree.

**Segmentation-based area is more defensible than box-based area**, since a bounding box always overestimates a round crown's true footprint. One validated single-crown mask came back at 22.86 m² (~2.7m radius) — a plausible size for a mature tree crown, confirming the pixel→m² conversion is sound at the individual-crown level. It does not fix the merged-crown undercount described above.

---

## Known limitations (read before trusting the output)

- **Undercounts dense, closed canopy** due to merged bounding boxes. Reliability is meaningfully lower here than on open/sparse canopy.
- **No single correct confidence threshold** — it is scene-dependent, and the true tree count for the dense test image likely lies somewhere between the 0.3 and 0.15 results above, not at either number exactly.
- **Resolution (meters/pixel) must be entered accurately by the user.** The tool has no way to verify this value; a wrong number produces a wrong area silently, with no visible error.
- **Segmentation uses MobileSAM**, a compressed model chosen specifically to fit free-tier hosting memory limits (see *Deployment notes*). Mask precision has not been rigorously benchmarked against full SAM — expect somewhat lower quality, not quantified here.
- **Not validated against any ground-truth count.** All figures are model output, not field-verified or manually cross-checked measurements.
- **Tested on two scene types only** — one open scrub image (Florida) and one dense closed-canopy image (NEON, temperate forest). Performance on other forest types, sensors, or coarser-resolution imagery is untested and unknown.
- **Only tested on small (500×500px) crops.** Full-size raw tiles (e.g. a 10,000×10,000px NEON tile) were not run end-to-end through the deployed app; behavior on very large uploads is unverified.

---

## Deployment notes (an honest account, since it's part of the story)

The original pipeline used full SAM (`vit_b`, 375MB checkpoint) and ran successfully in Google Colab (which provides ~12GB+ RAM). Deploying it to permanent, free, public hosting was not straightforward:

- **Render (free tier, 512MB RAM):** crashed with an explicit out-of-memory error during model loading/inference.
- **Streamlit Community Cloud (free tier, ~1GB RAM):** the UI and models loaded fine, but crashed with no traceback (consistent with an OS-level memory kill) immediately after running inference on an uploaded image.
- **Hugging Face Spaces:** as of a recent platform policy change, creating a new Gradio/Docker (compute) Space now requires a paid plan; Static Spaces can't run Python at all. Ruled out.
- **Fix:** swapped full SAM for **MobileSAM** (Tiny-ViT encoder, ~40MB checkpoint vs. 375MB) — same API, far smaller memory footprint. This resolved the Streamlit Cloud crash.
- **Render, re-tested with MobileSAM:** still failed with the same explicit out-of-memory error, even with the smaller model. This narrows down the real constraint: the baseline cost of PyTorch + torchvision + DeepForest + even the smallest SAM variant sits somewhere between 512MB and 1GB. Render's fixed 512MB ceiling is simply below that floor; Streamlit Cloud's ~1GB is just above it.

This is disclosed here rather than hidden because it's a real constraint of running vision models on free infrastructure, and the fix (a smaller model) is itself a legitimate engineering trade-off worth being explicit about — not a workaround to gloss over. The deployed link below runs on Streamlit Community Cloud, the platform that actually worked.

---

## Tech stack

- **Detection:** DeepForest 2.1.0 (`weecology/deepforest-tree`, pretrained RetinaNet)
- **Segmentation:** MobileSAM (Tiny-ViT encoder)
- **UI:** Streamlit (deployed) / Gradio (alternate version)
- **Test imagery:** DeepForest's bundled sample (`OSBS_029.png`) and NEON AOP RGB camera imagery (data product DP3.30010.001, 10cm/pixel)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py   # or: python app.py  for the Gradio version
```

The MobileSAM checkpoint downloads automatically on first run.
