#!/usr/bin/env python3
"""OCR pipeline: preprocess an image, extract text with Tesseract, keep words above a confidence threshold."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def make_demo_image(path: Path) -> Path:
    h, w = 500, 900
    img = np.full((h, w, 3), 235, np.uint8)
    grad = np.tile(np.linspace(0, 40, w, dtype=np.uint8), (h, 1))
    img = cv2.subtract(img, cv2.merge([grad, grad, grad]))

    lines = [
        ("INVOICE #00042", 1.6, 3),
        ("Date: 2026-03-15", 1.1, 2),
        ("Item: Server Rack Unit", 1.1, 2),
        ("Quantity: 2", 1.1, 2),
        ("Total Due: $4,990.00", 1.3, 2),
    ]
    y = 90
    for text, scale, thickness in lines:
        cv2.putText(img, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thickness, cv2.LINE_AA)
        y += 85

    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    img = cv2.warpAffine(img, rotation, (w, h), borderValue=(235, 235, 235))
    noise = np.random.default_rng(42).normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(path), img)
    return path


def estimate_skew(gray: np.ndarray) -> float:
    _, mask = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)


def preprocess(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    angle = estimate_skew(blurred)
    if 0.3 <= abs(angle) <= 15:
        h, w = blurred.shape
        rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        blurred = cv2.warpAffine(blurred, rotation, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    return gray, binary, angle


def recognise(binary: np.ndarray, psm: int, lang: str):
    data = pytesseract.image_to_data(
        binary, lang=lang, config=f"--oem 3 --psm {psm}", output_type=Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not text.strip() or conf < 0:
            continue
        words.append({
            "text": text.strip(),
            "confidence": round(conf / 100, 4),
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "line": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        })
    return words


def join_lines(words) -> str:
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    ordered = sorted(lines.values(), key=lambda ws: min(w["y"] for w in ws))
    return "\n".join(" ".join(w["text"] for w in sorted(ws, key=lambda w: w["x"])) for ws in ordered)


def annotate(binary: np.ndarray, kept, dropped) -> np.ndarray:
    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for w in dropped:
        cv2.rectangle(canvas, (w["x"], w["y"]), (w["x"] + w["w"], w["y"] + w["h"]), (0, 0, 220), 1)
    for w in kept:
        top_left, bottom_right = (w["x"], w["y"]), (w["x"] + w["w"], w["y"] + w["h"])
        cv2.rectangle(canvas, top_left, bottom_right, (0, 170, 0), 2)
        label = f'{w["text"]} {w["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(top_left[1] - 4, th + 4)
        cv2.rectangle(canvas, (top_left[0], y - th - 4), (top_left[0] + tw + 4, y + 2), (0, 170, 0), -1)
        cv2.putText(canvas, label, (top_left[0] + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def overview(original, gray, binary, annotated) -> np.ndarray:
    height = 360
    panels = []
    for title, image in [("Original", original), ("Grayscale", gray), ("Adaptive threshold", binary), ("OCR result", annotated)]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scale = height / image.shape[0]
        panel = cv2.resize(image, (int(image.shape[1] * scale), height))
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (40, 40, 40), -1)
        cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return cv2.hconcat(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="input image (default: generated demo)")
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--threshold", type=float, default=0.80, help="minimum word confidence, 0-1")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    source = args.image or make_demo_image(args.out / "sample_input.png")
    img = cv2.imread(str(source))
    if img is None:
        sys.exit(f"Could not read image: {source}")

    gray, binary, angle = preprocess(img)
    words = recognise(binary, args.psm, args.lang)
    kept = [w for w in words if w["confidence"] >= args.threshold]
    dropped = [w for w in words if w["confidence"] < args.threshold]

    text = join_lines(kept)
    mean_conf = float(np.mean([w["confidence"] for w in kept])) if kept else 0.0
    annotated = annotate(binary, kept, dropped)

    cv2.imwrite(str(args.out / "annotated.png"), annotated)
    cv2.imwrite(str(args.out / "overview.png"), overview(img, gray, binary, annotated))
    (args.out / "extracted_text.txt").write_text(text, encoding="utf-8")

    strip = lambda ws: [{k: v for k, v in w.items() if k != "line"} for w in ws]
    report = {
        "source": str(source),
        "psm": args.psm,
        "threshold": args.threshold,
        "skew_angle_deg": round(angle, 3),
        "mean_confidence": round(mean_conf, 4),
        "kept": strip(kept),
        "dropped": strip(dropped),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(text or "(no words passed the confidence threshold)")
    print(f"\nwords: {len(words)} detected, {len(kept)} kept, {len(dropped)} dropped | mean confidence: {mean_conf:.1%}")

    passed = bool(kept) and mean_conf >= args.threshold
    print(f"validation: {'PASS' if passed else 'FAIL'} (mean confidence >= {args.threshold:.0%})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
