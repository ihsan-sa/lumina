"""CLIP-based scene classification for concert video frames.

Uses OpenAI's CLIP model (ViT-B/32) via the transformers library for
zero-shot classification of video frames into scene categories. This
filters out non-stage frames (crowd shots, LED screen closeups, camera
transitions) before lighting feature extraction.

Scene categories:
- ``stage_view``: Concert stage with visible lighting rig (useful).
- ``crowd_view``: Audience / crowd shot (discard for lighting).
- ``led_screen``: LED screen or video display closeup (discard).
- ``transition``: Camera movement, blur, or cut artifact (discard).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Zero-shot classification labels matching DOCS.md Section 3.3 Step 2
SCENE_LABELS = [
    "concert stage with theatrical lighting",
    "concert crowd audience",
    "LED screen or video display",
    "camera transition or blurry image",
]

# Mapping from CLIP label text to short category names
_LABEL_TO_CATEGORY: dict[str, str] = {
    "concert stage with theatrical lighting": "stage_view",
    "concert crowd audience": "crowd_view",
    "LED screen or video display": "led_screen",
    "camera transition or blurry image": "transition",
}

_MODEL_NAME = "openai/clip-vit-base-patch32"


@dataclass(slots=True)
class SceneClassification:
    """Result of classifying a single video frame.

    Args:
        label: Short category name (stage_view, crowd_view, led_screen, transition).
        confidence: 0.0-1.0 softmax probability for the chosen label.
        all_scores: Full label-to-score mapping for all categories.
    """

    label: str
    confidence: float
    all_scores: dict[str, float]


class SceneClassifier:
    """CLIP-based zero-shot scene classifier for concert video frames.

    Loads the CLIP ViT-B/32 model on first use (lazy initialization) and
    classifies frames as stage views, crowd shots, LED screen closeups,
    or camera transitions using zero-shot text-image similarity.

    Args:
        device: PyTorch device string (e.g. "cuda", "cpu"). Defaults to
            CUDA if available.
        batch_size: Maximum number of frames to process in a single
            forward pass for batch classification.
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._batch_size = batch_size
        self._model: object | None = None
        self._processor: object | None = None
        self._text_features: torch.Tensor | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLIP model and processor on first use."""
        if self._model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model: %s", _MODEL_NAME)
        self._model = CLIPModel.from_pretrained(_MODEL_NAME).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(_MODEL_NAME)

        # Pre-compute text features for the scene labels (done once)
        text_inputs = self._processor(
            text=SCENE_LABELS,
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            self._text_features = self._model.get_text_features(**text_inputs)
            self._text_features = self._text_features / self._text_features.norm(
                dim=-1, keepdim=True
            )

        logger.info("CLIP model loaded on %s", self._device)

    def classify_frame(self, image: Image.Image) -> SceneClassification:
        """Classify a single video frame into a scene category.

        Args:
            image: PIL Image of the video frame.

        Returns:
            SceneClassification with the predicted label and confidence.
        """
        self._ensure_loaded()

        inputs = self._processor(
            text=SCENE_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)[0]

        probs_list = probs.cpu().tolist()
        best_idx = int(probs.argmax().item())
        best_label = SCENE_LABELS[best_idx]

        all_scores = {
            _LABEL_TO_CATEGORY[label]: score
            for label, score in zip(SCENE_LABELS, probs_list)
        }

        return SceneClassification(
            label=_LABEL_TO_CATEGORY[best_label],
            confidence=probs_list[best_idx],
            all_scores=all_scores,
        )

    def classify_batch(
        self,
        images: list[Image.Image],
    ) -> list[SceneClassification]:
        """Classify a batch of video frames for efficiency.

        Processes images in sub-batches of ``batch_size`` to manage GPU
        memory. More efficient than calling ``classify_frame`` in a loop
        because it batches the image encoder forward pass.

        Args:
            images: List of PIL Images to classify.

        Returns:
            List of SceneClassification, one per input image.
        """
        if not images:
            return []

        self._ensure_loaded()
        results: list[SceneClassification] = []

        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]

            inputs = self._processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                # Compute similarity using pre-cached text features
                logits = (image_features @ self._text_features.T) * 100.0
                probs = logits.softmax(dim=-1)

            for i in range(probs.shape[0]):
                frame_probs = probs[i].cpu().tolist()
                best_idx = int(probs[i].argmax().item())
                best_label = SCENE_LABELS[best_idx]

                all_scores = {
                    _LABEL_TO_CATEGORY[label]: score
                    for label, score in zip(SCENE_LABELS, frame_probs)
                }

                results.append(
                    SceneClassification(
                        label=_LABEL_TO_CATEGORY[best_label],
                        confidence=frame_probs[best_idx],
                        all_scores=all_scores,
                    )
                )

        return results

    def classify_numpy(self, frame: np.ndarray) -> SceneClassification:
        """Classify a single frame from a numpy array (BGR, from OpenCV).

        Convenience method that converts an OpenCV BGR numpy array to PIL
        Image before classification.

        Args:
            frame: BGR image as numpy array (H, W, 3), uint8.

        Returns:
            SceneClassification with predicted label and confidence.
        """
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.classify_frame(image)
