# Registration Card Scanner API

A REST API for extracting structured data from Tajikistan registration cards using OCR.

## Features

- **Automatic Field Extraction**: Extracts 13 different fields from registration cards
- **Multi-pass OCR**: Uses EasyOCR with multiple scanning passes for improved accuracy
- **Image Preprocessing**: Automatic deskewing, contrast enhancement, and color filtering
- **JSON Output**: Returns structured JSON data with confidence scores
- **Cyrillic Support**: Handles Russian and Tajik text

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MurtuzoPM/Registration_Card_Scanner.git
cd registration-card
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. The API will be available at:
```
http://localhost:5001
```

## Running in Google Colab

For heavier OCR workloads, run the project in Colab and enable GPU:

1. In Colab, set **Runtime → Change runtime type → GPU**
2. Run:

```bash
!git clone https://github.com/MurtuzoPM/Registration_Card_Scanner.git
%cd Registration_Card_Scanner/registration-card
!apt-get update -y
!apt-get install -y tesseract-ocr tesseract-ocr-rus
!pip install -r requirements.txt
```

3. Set runtime flags and start API:

```bash
%env OCR_USE_GPU=1
%env OCR_FAST_MODE=1
%env DEBUG=0
!python app.py
```

4. In another Colab cell, test API:

```bash
!curl -s http://127.0.0.1:5001/health
```

### Testing the API

Test the health endpoint:
```bash
curl http://localhost:5001/health
```

Extract data from an image:
```bash
curl -X POST -F "image=@path/to/card.jpg" http://localhost:5001/api/extract
```

## Usage

Send a POST request to `/api/extract` with an image file:

```bash
curl -X POST \
  -F "image=@registration_card.jpg" \
  http://localhost:5001/api/extract
```

The API returns JSON with extracted fields and confidence scores.

## Supported Fields

The system extracts the following 13 fields:

1. Registration Card Number
2. Passport Number
3. Citizenship
4. Name and Surname
5. Date of Registration
6. PRS MIA RT
7. Valid Until
8. Serial/Control Number
9. MIA (Ministry of Internal Affairs)
10. Place of Residence
11. Place of Residence (continued)
12. Inspector
13. Date of Registration/Extension

## Image Requirements

For best results:

- **Format**: JPEG or PNG
- **Resolution**: Minimum 1920x1080 pixels
- **Quality**: High quality, low compression
- **Lighting**: Even, diffused lighting
- **Angle**: Straight-on, perpendicular to card
- **File Size**: Under 50 MB

**Important**: Use actual photos of physical cards. Screenshots of cards displayed on screens perform poorly.

## Performance & Limitations

### Accuracy

- **High-quality photos**: ~77% accuracy (10/13 fields)
- **Screenshots**: 0-38% accuracy (not recommended)

### Known Limitations

1. **Image Quality Sensitivity**: System requires high-quality images
2. **Missing Fields**: Some fields may not be detected:
   - Registration Card Number (Field 1)
   - Name and Surname (sometimes has OCR errors)
   - Place of Residence Continuation (Field 11)
   - Date of Registration/Extension (Field 13)
3. **Cyrillic Recognition**: Some Cyrillic characters may be confused with Latin lookalikes
4. **Processing Time**: 5-15 seconds per image on CPU (typically faster on GPU/Colab)

For detailed performance analysis, see [PERFORMANCE_AND_LIMITATIONS.md](PERFORMANCE_AND_LIMITATIONS.md).

## Project Structure

```
registration-card/
├── app.py                  # Main Flask API application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── ocr_engine/
│   ├── __init__.py
│   ├── ocr_handler.py      # EasyOCR wrapper
│   ├── image_processor.py  # Image preprocessing
│   └── text_parser.py      # Text parsing logic
└── uploads/                # Temporary upload storage
```

## Configuration

Edit `config.py` to customize:

- **OCR Languages**: Default is Russian + English
- **GPU mode**: Auto-detect CUDA, or force on/off with `OCR_USE_GPU`
- **Image Preprocessing**: Enable/disable preprocessing
- **OCR speed mode**: Set `OCR_FAST_MODE` for lower CPU usage
- **File Upload**: Max file size, allowed formats
- **Server**: Host, port, debug mode
- **Field Definitions**: Customize field keywords and patterns

### Environment Variables

You can also set configuration via environment variables (using python-dotenv):

```bash
OCR_LANGUAGES=ru,en
OCR_ENGINE=easyocr
ENABLE_IMAGE_PREPROCESSING=true
OCR_FAST_MODE=1
OCR_USE_GPU=auto
PORT=5001
DEBUG=false
```

## API Endpoints

### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "ocr_engine": "easyocr",
  "languages": ["ru", "en"],
  "fast_mode": true,
  "gpu_enabled": true,
  "fields_count": 13,
  "fields": {
    "1": "Registration Card Number",
    "2": "Passport Number",
    ...
  }
}
```

### POST /api/extract

Extract data from uploaded registration card image.

**Request**:
- Content-Type: multipart/form-data
- Field: `image` (file)

**Success Response** (200 OK):
```json
{
  "success": true,
  "fields": {
    "registration_card_number": {
      "value": "",
      "confidence": 0.0,
      "label": "Registration Card Number",
      "field_number": 1
    },
    "passport_number": {
      "value": "EC5671065",
      "confidence": 0.74,
      "label": "Passport Number",
      "field_number": 2
    },
    "citizenship": {
      "value": "Хитой",
      "confidence": 0.959,
      "label": "Citizenship",
      "field_number": 3
    },
    ...
  },
  "confidence": 0.85,
  "raw_ocr": [
    {
      "text": "ЕС5671065",
      "confidence": 0.74,
      "bbox": [[1231, 1292], [2018, 1292], [2018, 1511], [1231, 1511]]
    },
    ...
  ]
}
```

**Error Responses**:
- `400 Bad Request`: No image provided, invalid file type, or file too large
- `500 Internal Server Error`: Processing failed

### Example Request (using curl)

```bash
curl -X POST \
  -F "image=@card.jpg" \
  http://localhost:5001/api/extract | jq .
```

### Example Request (using Python)

```python
import requests

url = "http://localhost:5001/api/extract"
files = {"image": open("card.jpg", "rb")}
response = requests.post(url, files=files)
data = response.json()

# Print extracted fields
for field_name, field_data in data["fields"].items():
    if field_data["value"]:
        print(f"{field_data['label']}: {field_data['value']} (confidence: {field_data['confidence']:.1%})")
```

## Technology Stack

- **Backend**: Flask (Python)
- **OCR Engine**: EasyOCR
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: NumPy

## Troubleshooting

### Poor Results

1. Ensure image is high resolution and well-lit
2. Use actual photos, not screenshots
3. Make sure card is flat and not bent
4. Try a different angle or lighting

### Common Errors

- **"No image file provided"**: Make sure to include the `image` field
- **"File too large"**: Reduce image size or compression
- **"Invalid image"**: Check file format is supported

### Low Confidence Scores

- **> 0.8**: Generally reliable
- **0.5 - 0.8**: Should be verified manually
- **< 0.5**: Likely contains errors

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is designed to assist with data extraction from registration cards. Due to OCR limitations, results should be verified manually, especially for critical applications. The system is not intended for fully automated processing without human oversight.

## Support

For issues, questions, or suggestions:

1. Check [PERFORMANCE_AND_LIMITATIONS.md](PERFORMANCE_AND_LIMITATIONS.md)
2. Review the troubleshooting section
3. Open an issue on GitHub

## Acknowledgments

- EasyOCR team for the excellent OCR library
- OpenCV community
- Font Awesome for icons

---

**Version**: 1.0  
**Last Updated**: June 3, 2026  
**Author**: MurtuzoPM