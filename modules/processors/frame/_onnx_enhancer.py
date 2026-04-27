"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Use bare provider — ONNX Runtime's defaults are fastest on
            # modern GPUs (Blackwell/sm_120).  Custom options like
            # EXHAUSTIVE cudnn_conv_algo_search hurt performance on these
            # architectures.
            config.append(p)
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    providers = build_provider_config()
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


# Per-size feathered mask cache — built once, reused every frame
_mask_cache: dict = {}


def _get_feather_mask(input_size: int) -> np.ndarray:
    """Return a cached uint8 feathered mask for the given size."""
    if input_size not in _mask_cache:
        mask = np.ones((input_size, input_size), dtype=np.float32)
        border = max(1, input_size // 16)
        ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
        ramp_dn = np.linspace(1.0, 0.0, border, dtype=np.float32)
        mask[:border, :] *= ramp_up[:, None]
        mask[-border:, :] *= ramp_dn[:, None]
        mask[:, :border] *= ramp_up[None, :]
        mask[:, -border:] *= ramp_dn[None, :]
        _mask_cache[input_size] = (mask * 255).astype(np.uint8)
    return _mask_cache[input_size]


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        input_name = session.get_inputs()[0].name
        output = run_inference(session, input_name, blob)
    enhanced = postprocess_face(output)

    h, w = frame.shape[:2]

    # Compute tight bbox to avoid full-frame warpAffine
    corners = np.array([[0,0],[input_size,0],[input_size,input_size],[0,input_size]], dtype=np.float32)
    transformed = (inv_M[:, :2] @ corners.T).T + inv_M[:, 2]
    x1 = max(0, int(np.floor(transformed[:, 0].min())))
    x2 = min(w, int(np.ceil(transformed[:, 0].max())))
    y1 = max(0, int(np.floor(transformed[:, 1].min())))
    y2 = min(h, int(np.ceil(transformed[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return frame

    # Shift inv_M to crop-local coordinates
    inv_crop = inv_M.copy()
    inv_crop[0, 2] -= x1
    inv_crop[1, 2] -= y1
    crop_w, crop_h = x2 - x1, y2 - y1

    # Warp enhanced face and mask into crop space only (much cheaper than full frame)
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    feather_mask = _get_feather_mask(input_size)
    warped_mask = cv2.warpAffine(
        feather_mask, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Fast uint8 blend via cv2 SIMD (avoids float32 round-trip)
    # Only blend pixels where mask > 0 to avoid black border artifacts
    target_crop = frame[y1:y2, x1:x2]
    alpha_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
    inv_alpha = 255 - alpha_3c
    blended = cv2.add(
        cv2.multiply(warped_enhanced, alpha_3c, scale=1.0 / 255.0),
        cv2.multiply(target_crop,    inv_alpha,  scale=1.0 / 255.0),
    )
    # Where mask is completely zero, keep original frame (avoids black border)
    mask_nonzero = warped_mask > 0
    result = frame.copy()
    crop_result = target_crop.copy()
    crop_result[mask_nonzero] = blended[mask_nonzero]
    result[y1:y2, x1:x2] = crop_result
    return result
