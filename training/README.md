# Training / calibration data (local only)

This folder holds **your real card images and their correct values**. Real
images and `labels.local.json` are **git-ignored on purpose** (see the repo
`.gitignore`) so personal ID data never gets pushed to GitHub.

## What goes where

```
training/
  images/                 # your card photos: card_001.png, card_002.png, ...
  labels.local.json       # the correct values for each photo (you create this)
  reports/                # generated debug/evaluation JSON (you can share these)
```

## Steps

1. Put 3-5 card photos in `training/images/`, named `card_001.png`, `card_002.png`, ...
2. Copy `labels.example.json` to `labels.local.json` and fill in the correct
   values for each card. Use `""` for fields that are blank on that card.
3. Generate a debug report per card (contains OCR text + box positions):

   ```
   python tools/debug_extract.py training/images/card_001.png --slow \
       --out training/reports/card_001_debug.json
   ```

4. (Optional) Score everything at once:

   ```
   python tools/evaluate_dataset.py --labels training/labels.local.json --slow
   ```

## What to share for ROI calibration

The `training/reports/*_debug.json` files contain the OCR text and the pixel
**box coordinates** of every detected word, plus the processed `image_size`.
That is enough to calibrate the field template **without sharing the images
themselves**. Note the debug JSON still contains the card's text (names, etc.),
so only share it if that is acceptable for your data.
