from pathlib import Path
import io
import os

import httpx
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from retinaface.commons import postprocess


@serve.deployment
class TFMnistModel:
    def __init__(self, model_path: str):
        # Ensure legacy keras behavior if needed (before importing tensorflow)
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        import tensorflow as tf

        self.model_path = model_path
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)

        # Constants copied from RetinaFace.detect_faces
        self._feat_stride_fpn = [32, 16, 8]
        self._anchors_fpn = {
            "stride32": np.array(
                [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
            ),
            "stride16": np.array(
                [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
            ),
            "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
        }
        self._num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    def _prepare_image_for_size(self, image_rgb: Image.Image, size: int):
        # Use PIL to resize and keep RGB only
        resized: Image.Image = image_rgb.resize((size, size), Image.BILINEAR)
        rgb_np = np.asarray(resized, dtype=np.float32)
        im_tensor = rgb_np[None, ...]
        im_info = (rgb_np.shape[0], rgb_np.shape[1])
        im_scale = 1.0
        return im_tensor, im_info, im_scale, resized

    def _postprocess(self, net_out, im_info, im_scale, threshold: float = 0.9) -> dict:
        # Logic copied from RetinaFace.detect_faces after `net_out = model(im_tensor)`
        nms_threshold = 0.4
        decay4 = 0.5

        proposals_list = []
        scores_list = []
        landmarks_list = []

        # Ensure numpy arrays
        net_out = [elt.numpy() if hasattr(elt, "numpy") else elt for elt in net_out]
        sym_idx = 0
        for _, s in enumerate(self._feat_stride_fpn):
            scores = net_out[sym_idx]
            scores = scores[:, :, :, self._num_anchors[f"stride{s}"] :]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = self._num_anchors[f"stride{s}"]
            K = height * width
            anchors_fpn = self._anchors_fpn[f"stride{s}"]
            anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
            proposals = postprocess.bbox_pred(anchors, bbox_deltas)

            proposals = postprocess.clip_boxes(proposals, im_info[:2])

            if s == 4 and decay4 < 1.0:
                scores *= decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            # proposals[:, 0:4] /= im_scale
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3] // A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
            landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            # landmarks[:, :, 0:2] /= im_scale
            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list) if len(proposals_list) > 0 else np.empty((0, 0))
        if proposals.shape[0] == 0:
            return {}

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = postprocess.cpu_nms(pre_det, nms_threshold)

        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        landmarks = landmarks[keep]

        resp: dict = {}
        for idx, face in enumerate(det):
            label = "face_" + str(idx + 1)
            resp[label] = {}
            resp[label]["score"] = float(face[4])
            resp[label]["facial_area"] = list(face[0:4].astype(int))
            resp[label]["landmarks"] = {
                "right_eye": list(landmarks[idx][0]),
                "left_eye": list(landmarks[idx][1]),
                "nose": list(landmarks[idx][2]),
                "mouth_right": list(landmarks[idx][3]),
                "mouth_left": list(landmarks[idx][4]),
            }
        return resp

    @staticmethod
    def _draw_detections(image_rgb: Image.Image, detections: dict, color: tuple) -> Image.Image:
        annotated = image_rgb.copy()
        draw = ImageDraw.Draw(annotated)
        for _, info in detections.items():
            x1, y1, x2, y2 = info["facial_area"]
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            lm = info["landmarks"]
            for key in ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left"):
                px, py = lm[key]
                r = 2
                draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=color)
        return annotated

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

        sizes = [1024, 640, 320]
        threshold = 0.9

        outputs_dir = Path("temp/outputs").absolute()
        outputs_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        colors = {1024: (0, 255, 0), 640: (255, 0, 0), 320: (0, 0, 255)}

        for size in sizes:
            im_tensor, im_info, im_scale, resized_img = self._prepare_image_for_size(image, size)
            # Run model
            prediction = self.model(im_tensor, training=False)
            if not isinstance(prediction, (list, tuple)):
                net_out = [prediction]
            else:
                net_out = list(prediction)

            detections = self._postprocess(net_out, im_info, im_scale, threshold)

            # Draw on resized RGB image
            annotated = self._draw_detections(resized_img, detections, colors[size]) if len(detections) > 0 else resized_img.copy()
            out_path = outputs_dir / f"serve-annotated-{size}.jpg"
            annotated.save(str(out_path))
            print(out_path)

            # results[str(size)] = {
            #     "num_faces": len(detections),
            #     "detections": detections,
            #     "output_file": str(out_path),
            # }

        return JSONResponse({"model_file": str(self.model_path)})
    

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
