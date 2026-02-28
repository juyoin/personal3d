# 2D → 3D Browser Reconstruction App

This project implements an end-to-end, modular Python pipeline that transforms a single RGB image into an interactive 3D point-cloud model in the browser.

## Pipeline logic

1. **Image preprocessing** (`app/pipeline/preprocess.py`)
   - Loads user image and resizes to manageable resolution.
   - Normalizes depth map values to `[0,1]` for stable geometric scaling.

2. **Semantic foreground segmentation** (`app/pipeline/segmentation.py`)
   - Primary path: `facebook/detr-resnet-50-panoptic` identifies scene segments.
   - Chooses the dominant high-confidence segment as primary foreground subject.
   - Fallback path: HSV + morphology + largest connected component heuristic.
   - Model loading is lazy and failure-tolerant (no startup crash if model download/runtime is unavailable).

3. **Monocular depth estimation** (`app/pipeline/depth.py`)
   - Primary path: `Intel/dpt-large` infers dense depth from one image.
   - Fallback path: Sobel-gradient pseudo-depth when model/runtime is unavailable.

4. **3D projection** (`app/pipeline/pointcloud.py`)
   - Converts pixel coordinates `(u,v)` + normalized depth `d` into 3D coordinates `(x,y,z)`.
   - Keeps points only within segmentation mask.
   - Attaches RGB colors to each 3D point.

5. **Export + serving** (`app/pipeline/exporters.py`, `app/main.py`)
   - Writes point cloud to JSON + debug mask/depth images.
   - FastAPI endpoint returns generated artifact URLs.

6. **Interactive visualization** (`app/static/app.js`)
   - Three.js renders point cloud in WebGL canvas.
   - OrbitControls enables mouse-based rotate, pan, and zoom for 360° inspection.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## API

- `POST /api/reconstruct`
  - Form field: `image`
  - Returns URLs for point cloud JSON, segmentation mask, depth map, and metadata.

- `GET /api/reconstruct?image=<reference>`
  - Processes an existing image by filename/path (searches `app/samples`, `app/uploads`, project root, or absolute path).

## Quick query mode

You can process a server-local image directly from the URL:

```bash
http://localhost:8000/?image=your_file.png
```

For convenience, place files in `app/samples/`.

## Transformation summary: 2D pixels → navigable 3D scene

- **2D input**: image pixels `(u, v, RGB)`.
- **Subject isolation**: segmentation generates binary mask `M(u,v)`.
- **Geometry inference**: depth estimator produces scalar depth `D(u,v)`.
- **Projection**: each masked pixel mapped to point `P = (x(u), y(v), z(D))`.
- **Color binding**: assign original RGB to each `P`.
- **Render**: WebGL displays colored points with interactive camera controls.
