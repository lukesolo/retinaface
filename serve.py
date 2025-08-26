from pathlib import Path
import io

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse


@serve.deployment
class TFMnistModel:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request):
        # Accept either multipart/form-data file upload (key: "file")
        # or raw bytes in the request body
        content_type = starlette_request.headers.get("content-type", "")
        upload = None
        image = None
        if "multipart/form-data" in content_type:
            try:
                form = await starlette_request.form()
                upload = form.get("file")
            except Exception as e:  # pragma: no cover - runtime parsing issue
                return JSONResponse({"error": f"Invalid multipart form: {e}"}, status_code=400)

        try:
            if upload is not None:
                image = Image.open(upload.file)
            else:
                body = await starlette_request.body()
                if not body:
                    return JSONResponse({"error": "Empty request body"}, status_code=400)
                image = Image.open(io.BytesIO(body))
            image = image.convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse({"error": "Unsupported or corrupt image"}, status_code=415)

        image_np = np.asarray(image, dtype=np.float32)
        # Add batch dimension for the model: (1, H, W, 3)
        image_np = image_np[None, ...]

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(image_np, training=False)

        # Step 3: tensorflow output -> web output
        if isinstance(prediction, (list, tuple)):
            shapes = [tuple(int(d) if d is not None else -1 for d in p.shape) for p in prediction]
        else:
            shapes = [tuple(int(d) if d is not None else -1 for d in prediction.shape)]

        return JSONResponse({"prediction_shapes": shapes, "file": str(self.model_path)})
    

model_path = Path("retinaface_full_model.h5").absolute()
app = TFMnistModel.bind(model_path=model_path)

serve.run(app, route_prefix="/")

image_path = Path("/home/gennady/dataset/superbet/with_external_id/selfies/66ba3aa6e65ec7aad2766611.jpg")
with image_path.open("rb") as f:
    resp = httpx.post(
        "http://localhost:8000/",
        files={"file": (image_path.name, f, "image/jpeg")},
        timeout=60,
    )
    try:
        print(resp.status_code, resp.json())
    except Exception:
        print(resp.status_code, resp.text)
