from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from app.pipeline.pipeline import ImageTo3DPipeline

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

config = PipelineConfig(output_dir=BASE_DIR / "outputs")
pipeline = ImageTo3DPipeline(config)

app = FastAPI(title="2D to 3D Reconstruction Studio")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=config.output_dir), name="outputs")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": DEFAULT_CONFIG.depth_model}
