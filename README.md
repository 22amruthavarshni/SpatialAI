# SpatialAI

**3D Scene Understanding & Reconstruction** — an end-to-end Spatial AI pipeline that turns a single handheld phone video into a reconstructed, reasoning-capable 3D scene. Object detection, monocular depth estimation, camera pose estimation, dense 3D reconstruction, and multi-modal fusion, combined into one coordinate system and one queryable scene graph.

Built entirely on a free-tier Google Colab T4 GPU, from a 34-second video of a desk.

**[→ View the interactive results dashboard](https://22amruthavarshni.github.io/SpatialAI/spatialai_dashboard.html)**

---

## What this is

Most computer vision portfolio projects stop at a single model — detection, or depth, or a point cloud. SpatialAI wires four independent, well-established computer vision techniques into one pipeline so the *output of one becomes the input of the next*, and adds a fusion layer that resolves the coordinate-system mismatches between them. The result is a structured, queryable understanding of a real-world scene: not just "here's a 3D model," but "here's the laptop, here's the chair, and here's how far apart they are."

## Pipeline

| Phase | What it does | Model / Tool |
|---|---|---|
| 0 — Setup & Capture | Frame extraction from a slow handheld arc video (36 frames @ 1fps) | OpenCV |
| 1 — Object Detection | Per-frame bounding boxes, class, confidence | YOLOv8 (nano → medium) |
| 2 — Depth Estimation | Per-pixel relative depth from a single image | MiDaS (small) |
| 3 — Camera Pose Estimation | Solves camera position + orientation for every frame via Structure from Motion, plus a sparse 3D point cloud | COLMAP |
| 4 — Dense Reconstruction | Trains a renderable 3D scene from the sparse point cloud + camera poses | 3D Gaussian Splatting (INRIA reference implementation) |
| 5 — Fusion | Unprojects 2D detections into COLMAP's 3D coordinate space using depth + pose, with scale alignment between MiDaS and COLMAP | NumPy, custom projection math |
| 6 — Scene Graph & Dashboard | Robust per-object clustering (median + MAD outlier filtering), reliability labeling, pairwise distances | Python, Plotly |

## Key results

- **36/36 camera frames registered** by COLMAP, mean reprojection error **0.39px**
- **5,234** sparse 3D points reconstructed; **630** image pairs matched (exhaustive matching)
- Gaussian Splatting: **PSNR 37.3 dB** at 3,000 iterations, visibly sharper geometry at the full 30,000-iteration run
- MiDaS→COLMAP scale alignment computed from **138,677** point observations (scale factor ≈ 0.0151)
- Final scene graph: **6 fused objects**, reliability-labeled, with pairwise 3D distances

## Known limitations (and why they're there)

This project deliberately documents its own failure modes rather than hiding them:

- **YOLO misclassifications**: COCO's 80-class vocabulary has no "mousepad" category, so a mouse + mousepad were consistently detected as "book." Fixed partially via confidence thresholding and a larger model (nano → medium); the rest is a documented, expected limitation of closed-vocabulary detectors.
- **Sparse reconstruction gaps**: flat, low-texture surfaces (the desk, the chair's frame) produced almost no SIFT keypoints and stayed sparse in Phase 3 — expected behavior for feature-based SfM, resolved (mostly) by dense reconstruction in Phase 4.
- **Relative vs. metric depth**: MiDaS outputs relative depth, not real-world units. A single global scale factor (median-based) was used to align it with COLMAP's coordinate system; per-region scale fitting is a noted future improvement.
- **Fusion reliability**: the scene graph labels each object `high` / `medium` / `low` reliability rather than presenting every position as equally trustworthy. Chair and book positions are flagged `low` — traced directly back to the texture and classification issues above, not unexplained noise.

## Tech stack

Python · PyTorch · OpenCV · Ultralytics YOLOv8 · MiDaS (Intel ISL) · COLMAP · 3D Gaussian Splatting · NumPy · Plotly · Google Colab (T4 GPU)

## Repository structure

```
SpatialAI/
├── notebooks/              # Phase-by-phase Colab notebooks
│   ├── 01_frame_extraction.ipynb
│   ├── 02_object_detection.ipynb
│   ├── 03_depth_estimation.ipynb
│   ├── 04_camera_pose_estimation.ipynb
│   ├── 05_gaussian_splatting.ipynb
│   └── 06_scene_fusion.ipynb
├── outputs/
│   ├── detections/          # YOLO detections (detections.json — class, confidence, box per frame)
│   ├── depth_maps/          # MiDaS depth maps (raw arrays + visualizations)
│   ├── camera_poses/        # COLMAP sparse reconstruction (cameras, images, points3D)
│   └── reconstruction/      # Gaussian Splatting trained models
├── scene_graph/
│   └── scene_graph.json     # Final fused object graph
├── docs/
│   └── spatialai_dashboard.html   # Self-contained interactive results dashboard
└── README.md
```

## Reproducing this

1. Record a slow ~30 second handheld video in an arc around a static, texture-rich scene (60–80% overlap between frames matters more than frame count).
2. Run the notebooks in `notebooks/` in order — each is self-contained and mounts Google Drive for persistence across sessions.
3. Phases 1–2 run comfortably on a free Colab T4. Phase 4 (Gaussian Splatting) benefits from the GPU but will also run, more slowly, on CPU-only COLMAP steps.
4. Open `docs/spatialai_dashboard.html` directly in a browser to explore the final reconstruction and scene graph — no server required.

## Author

**Amrutha Varshni G R** — B.E. AI & ML, Dayananda Sagar Academy of Technology and Management
[GitHub](https://github.com/22amruthavarshni) · [LinkedIn](https://www.linkedin.com/in/amrutha-varshni22052005/) · [Portfolio](https://22amruthavarshni.github.io/portfolio/)
