from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from app.pipeline.pipeline import ImageTo3DPipeline

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

config = PipelineConfig(output_dir=BASE_DIR / "outputs")
pipeline = ImageTo3DPipeline(config)

app = FastAPI(title="2D to 3D Reconstruction Studio")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=config.output_dir), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_query": request.query_params.get("image", "")},
    )


def _resolve_image_query(image_ref: str) -> Path:
    candidate = unquote(image_ref).strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="Missing image query parameter")

    raw = Path(candidate)
    if raw.is_absolute() and raw.exists():
        return raw

    search_roots = [UPLOAD_DIR, SAMPLES_DIR, Path.cwd(), BASE_DIR]
    for root in search_roots:
        path = (root / candidate).resolve()
        if path.exists() and path.is_file():
            return path

    raise HTTPException(
        status_code=404,
        detail=(
            f"Image '{candidate}' not found. Place files in app/samples or app/uploads, "
            "or provide an existing absolute path."
        ),
    )


@app.post("/api/reconstruct")
async def reconstruct(image: UploadFile = File(...)) -> JSONResponse:
    file_id = uuid.uuid4().hex
    ext = Path(image.filename or "upload.png").suffix or ".png"
    image_path = UPLOAD_DIR / f"{file_id}{ext}"

    image_bytes = await image.read()
    image_path.write_bytes(image_bytes)

    output = pipeline.run(image_path=image_path, stem=file_id)

    return JSONResponse(
        {
            "pointCloudUrl": f"/outputs/{output.point_cloud_json.name}",
            "maskUrl": f"/outputs/{output.mask_image.name}",
            "depthUrl": f"/outputs/{output.depth_image.name}",
            "label": output.label,
            "confidence": output.confidence,
            "pointCount": output.point_count,
        }
    )


@app.get("/api/reconstruct")
def reconstruct_from_query(image: str = Query(..., description="Path or filename to an existing image")) -> JSONResponse:
    image_path = _resolve_image_query(image)
    output = pipeline.run(image_path=image_path, stem=uuid.uuid4().hex)

    return JSONResponse(
        {
            "pointCloudUrl": f"/outputs/{output.point_cloud_json.name}",
            "maskUrl": f"/outputs/{output.mask_image.name}",
            "depthUrl": f"/outputs/{output.depth_image.name}",
            "label": output.label,
            "confidence": output.confidence,
            "pointCount": output.point_count,
            "sourceImage": image,
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": DEFAULT_CONFIG.depth_model}
