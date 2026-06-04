# Registration Card Scanner - Performance & Limitations Documentation

## Overview
This document describes the current performance characteristics and limitations of the Tajikistan Registration Card Scanner application.

## Current Performance

### Test Results Summary

#### High-Quality Photos (JPG images)
- **Accuracy**: ~77% (10/13 fields correctly extracted)
- **Example**: `13babb5cbbcb424dbd07184968394ee3.jpg`
- **Successfully extracted fields**:
  - ✓ Passport Number (EC5671065) - High confidence
  - ✓ Citizenship (Хитой) - High confidence  
  - ✓ Date of Registration (01.09.2025) - Medium confidence
  - ✓ PRS MIA RT (ХШБ ВКД ЧТ) - High confidence
  - ✓ Valid Until (07.09.2026) - High confidence
  - ✓ Serial/Control Number (9801) - High confidence
  - ✓ MIA (ВКД) - Very high confidence
  - ✓ Place of Residence (Душанбе) - Perfect confidence
  - ✓ Inspector (partially correct) - Medium confidence

#### Low-Quality Images (Screenshots)
- **Accuracy**: 0-38% (0-5/13 fields extracted)
- **Example files**: `Screenshot From 2026-06-02 11-2*.png`
- **Issues**: Poor OCR quality, garbled text extraction

## Known Limitations

### 1. Image Quality Requirements
**Issue**: The system is highly sensitive to image quality.

**Details**:
- Works best with high-resolution photos of physical cards
- Screenshots of cards displayed on screens perform poorly due to:
  - Moiré patterns from screen pixels
  - Different lighting conditions
  - Compression artifacts
  - Lower effective resolution

**Recommendation**: Use actual photos of physical registration cards, not screenshots.

### 2. Missing Fields

#### Registration Card Number (Field 1)
- **Status**: Often not detected
- **Reason**: May be in a location with poor contrast or the number pattern doesn't match expected format
- **Impact**: Low - this is typically redundant with other identifying information

#### Name and Surname (Field 4)  
- **Status**: Sometimes extracted with OCR errors
- **Example**: "Вашг Гюен" instead of correct name
- **Reason**: Cyrillic text recognition challenges, especially with unusual fonts or handwriting
- **Impact**: Medium - requires manual verification

#### Place of Residence Continuation (Field 11)
- **Status**: Often empty
- **Reason**: May not be present on all cards or located in hard-to-OCR area
- **Impact**: Low - only relevant for long addresses

#### Date of Registration/Extension (Field 13)
- **Status**: Often not detected
- **Reason**: May be missing from some card versions or poorly positioned
- **Impact**: Low - usually covered by Field 5 (Date of Registration)

### 3. OCR Accuracy Issues

#### Cyrillic Text Recognition
- **Challenge**: Some Cyrillic characters are confused with Latin lookalikes
- **Examples**: 
  - Cyrillic 'о' recognized as Latin 'o'
  - Cyrillic 'с' recognized as Latin 'c'
  - Cyrillic 'е' recognized as Latin 'e'
- **Mitigation**: System includes character normalization, but some errors persist

#### Low Confidence Values
- **Threshold**: Fields with confidence < 0.25 are automatically rejected
- **Impact**: Some valid fields may be excluded if OCR confidence is too low
- **Recommendation**: Review fields with confidence between 0.25-0.5 manually

### 4. Field-Specific Issues

#### Inspector Name (Field 12)
- **Issue**: Often partially garbled
- **Example**: "омализОДа ф" instead of "Имомализода Ф."
- **Reason**: Small font size, handwritten signatures nearby, or poor print quality
- **Impact**: Medium - may need manual correction

#### Citizenship (Field 3)
- **Issue**: Sometimes includes extra characters
- **Example**: "НИА (гуреза" instead of "Хитой"
- **Reason**: Nearby text bleeding into the field
- **Impact**: Medium - requires validation

### 5. Technical Limitations

#### Processing Time
- **Current**: 5-15 seconds per image (CPU only)
- **Bottleneck**: EasyOCR multi-pass processing
- **Note**: Would be faster with GPU acceleration

#### Memory Usage
- **Peak**: ~2-3 GB during OCR processing
- **Reason**: Loading OCR models and processing high-resolution images
- **Impact**: May be slow on low-end devices

#### Language Support
- **Supported**: Russian (ru) and English (en)
- **Limitation**: Tajik language not explicitly supported, relies on Cyrillic script overlap with Russian
- **Impact**: Some Tajik-specific characters may not be recognized optimally

## Best Practices for Optimal Results

### 1. Image Capture Guidelines
- **Resolution**: Minimum 1920x1080 pixels, preferably higher
- **Lighting**: Even, diffused lighting without shadows
- **Angle**: Straight-on, perpendicular to the card
- **Focus**: Sharp focus on all text areas
- **Background**: Plain, contrasting background

### 2. Image Format
- **Recommended**: JPEG or PNG
- **Quality**: High quality (low compression)
- **Size**: Under 10 MB for optimal processing

### 3. Card Condition
- **Clean**: Free of dirt, smudges, or damage
- **Flat**: Not bent or curled
- **Original**: Photocopies may not work as well

### 4. Validation Workflow
1. **Review** all extracted fields, especially those with confidence < 0.7
2. **Verify** critical fields: Passport Number, Name, Dates
3. **Cross-check** with physical card when possible
4. **Manual correction** for any obvious OCR errors

## Error Handling

### Low Confidence Warnings
The system automatically flags fields with confidence scores:
- **High confidence** (> 0.8): Generally reliable
- **Medium confidence** (0.5-0.8): Should be verified
- **Low confidence** (< 0.5): Likely contains errors

### Empty Fields
When fields are empty, possible causes:
1. Field not present on card version
2. OCR failed to detect text in that area
3. Text confidence below threshold
4. Field location differs from expected position

### Common OCR Errors
- **Number/letter confusion**: 0 vs O, 1 vs I, 5 vs S
- **Cyrillic/Latin confusion**: о vs o, с vs c, е vs e
- **Similar characters**: з vs 3, ч vs 4, ш vs w

## Future Improvements

### Potential Enhancements
1. **Better screenshot handling** - Additional preprocessing for screen captures
2. **Tajik language model** - Train OCR specifically on Tajik text
3. **Manual correction interface** - Allow users to fix OCR errors
4. **Confidence thresholds** - Configurable thresholds per field
5. **Batch processing** - Process multiple cards simultaneously
6. **GPU acceleration** - Faster processing with GPU support

### Known Issues to Address
1. Registration card number detection
2. Name field accuracy improvement
3. Inspector name extraction
4. Handling of different card versions/layouts

## Support and Troubleshooting

### When Results Are Poor
1. **Check image quality** - Ensure high resolution and good lighting
2. **Try a different image** - Some angles/lighting work better
3. **Verify card version** - Some older cards may have different layouts
4. **Manual entry** - For critical applications, manual verification recommended

### Reporting Issues
When reporting OCR errors, please include:
- Original image (if possible)
- Extracted results
- Expected/correct values
- Image capture conditions (lighting, angle, etc.)

## Conclusion

The Registration Card Scanner provides **good accuracy (~77%) for high-quality photos** but has significant limitations with screenshots and certain field types. Users should:

1. **Use high-quality photos** of physical cards
2. **Review and verify** all extracted data
3. **Understand the limitations** and plan accordingly
4. **Provide manual correction** when accuracy is critical

The system is suitable for **assistive data extraction** but should not be relied upon for **fully automated processing** without human verification.

---

**Last Updated**: June 3, 2026  
**Version**: 1.0  
**Tested With**: EasyOCR 1.6.2, Python 3.14