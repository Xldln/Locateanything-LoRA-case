# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
locateanything_worker.py - A reusable worker for LocateAnything inference.
"""
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single perception query.

        Args:
            image: PIL Image (RGB).
            question: The task prompt (see supported prompts below).
            generation_mode: "fast" (MTP) | "slow" (NTP) | "hybrid".
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            verbose: If True, return timing statistics.

        Returns:
            dict with keys: "answer", "stats" (optional), "history" (optional).
        """
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points


# --------------- Usage Example ---------------
if __name__ == "__main__":
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
    img = Image.open("example.jpg").convert("RGB")

    # Object Detection
    result = worker.detect(img, ["person", "car", "bicycle"])
    print("Detection:", result["answer"])

    # Phrase Grounding (multiple)
    result = worker.ground_multi(img, "people wearing red shirts")
    print("Grounding:", result["answer"])

    # Scene Text Detection
    result = worker.detect_text(img)
    print("Text Detection:", result["answer"])

    # Pointing
    result = worker.point(img, "the traffic light")
    print("Pointing:", result["answer"])

    # GUI Grounding (point)
    result = worker.ground_gui(img, "the search button", output_type="point")
    print("GUI Point:", result["answer"])

    # Parse structured output
    w, h = img.size
    boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
    print("Parsed boxes:", boxes)
