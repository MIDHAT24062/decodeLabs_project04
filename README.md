# decodeLabs_project04
# OCR Recognition Pipeline

Extracts text from an image with OpenCV and Tesseract, keeping only words at or above a confidence threshold (default 80%).

## Pipeline

1. Grayscale conversion
2. Gaussian blur
3. Deskew
4. Adaptive thresholding
5. Tesseract OCR (`pytesseract`)
6. Confidence filter, then annotated output

## Setup

```bash
pip install opencv-python pytesseract numpy
```

Install the Tesseract engine separately: `sudo apt install tesseract-ocr` (Linux), `brew install tesseract` (macOS), or the [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki).

## Usage

```bash
python project4_ocr_pipeline.py                      # runs on a generated demo image
python project4_ocr_pipeline.py --image scan.png     # your own image
```

| Option | Default | Description |
|---|---|---|
| `--image` | demo | Input image path |
| `--out` | `output` | Output folder |
| `--psm` | `3` | Page segmentation: `3` auto, `6` text block, `7` single line, `11` sparse text |
| `--lang` | `eng` | Tesseract language |
| `--threshold` | `0.80` | Minimum word confidence (0-1) |

Exits with code 0 if the mean confidence of kept words meets the threshold, otherwise 1.

## Output

Written to the output folder:

- `extracted_text.txt`: filtered text
- `annotated.png`: green boxes for accepted words, red for rejected
- `overview.png`: original, grayscale, thresholded and result side by side
- `report.json`: per-word confidence and coordinates
