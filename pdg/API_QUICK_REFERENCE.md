# API Integration - Quick Reference Guide

## ✅ All Issues Fixed

Your API has been comprehensively fixed with robust error handling, input validation, and logging.

---

## Running the Application

### Start the App
```bash
python app.py
```

**What happens:**
1. ✅ App starts immediately (doesn't wait for model training)
2. ✅ Background thread starts training model if needed
3. ✅ Access at `http://127.0.0.1:5000` right away
4. ✅ If model not ready, CV fallback is used for image analysis

---

## API Endpoints

### 1. **GET** `/` - Home Page
Returns the main HTML interface

### 2. **POST** `/predict` - Text-Based Description
**Parameters:**
- `product_name` (required): Product name - max 500 chars
- `category` (required): Product category - max 500 chars
- `brand` (optional): Brand name - max 500 chars
- `key_features` (optional): Comma-separated features - max 500 chars
- `product_image` (optional): Image file (PNG, JPG, JPEG, GIF, WEBP) - max 10MB

**Response:**
- Returns HTML page with generated description
- On error: Shows error message on same page

**Example Error Handling:**
```
✓ Missing product_name → "Product Name and Category are required."
✓ File too large → "File size exceeds maximum of 10MB."
✓ Invalid image → "Image validation failed: Image is corrupted..."
✓ Model not ready → "Failed to generate description. Please try again..."
```

### 3. **POST** `/predict_image` - Image-Based Description
**Parameters:**
- `product_image` (required): Product image - max 10MB
- `product_name` (optional): Override detected name
- `category` (optional): Override detected category
- `brand` (optional): Override detected brand
- `key_features` (optional): Additional features

**Response:**
- Returns HTML page with:
  - Generated description
  - Detected product attributes
  - Image analysis results
  - AI confidence score (if available)

**Example Error Handling:**
```
✓ No image uploaded → "Please upload a product image."
✓ Corrupted image → "Image validation failed: Cannot identify image file"
✓ Image too large → "Image dimensions 5000x5000 exceed maximum of 4000x4000."
✓ AI analysis fails → Falls back to CV analysis automatically
```

---

## Input Constraints & Validation

| Field | Type | Max Length | Notes |
|-------|------|-----------|-------|
| Product Name | Text | 500 chars | Required |
| Category | Text | 500 chars | Required |
| Brand | Text | 500 chars | Optional |
| Features | Text | 500 chars | Optional |
| Image File | Image | 10MB | PNG, JPG, JPEG, GIF, WEBP |
| Image Size | Pixels | 4000x4000 | Maximum dimensions |

---

## Error Messages & Solutions

### File Upload Errors
```
"File size exceeds maximum of 10MB."
→ Use smaller image file

"Unsupported image type. Allowed: PNG, JPG, JPEG, GIF, WEBP."
→ Convert image to supported format

"Image validation failed: Image is corrupted or invalid"
→ Use a different image file that opens properly
```

### Generation Errors
```
"Product Name and Category are required."
→ Fill in both required fields

"Failed to generate description. Please check the model setup."
→ Model not loaded; restart app or wait for background training

"Error: Model not trained. Please run train_model.py first."
→ Run: python train_model.py
```

### Image Analysis Errors
```
"Failed to process the uploaded image. Please try another image."
→ Image may be corrupted or too large; try different image

"Error analyzing image."
→ Restart app or try different image

"Image dimensions 5000x5000 exceed maximum of 4000x4000..."
→ Resize image to smaller dimensions
```

---

## Logging & Debugging

### View Application Logs
The app logs all operations to console with levels:
- `INFO`: Normal operations (successful API calls, model loading)
- `WARNING`: Issues with fallbacks (AI API unavailable)
- `ERROR`: Failures (file errors, parsing errors)

### Example Log Output
```
INFO:root:NLP artifacts loaded successfully
INFO:root:Attempting AI-powered product analysis...
INFO:root:Calling Claude Vision API for image analysis
INFO:root:Claude analysis completed successfully
```

### If Something Goes Wrong
```
ERROR:root:Model files not found: [Errno 2] No such file
→ Run: python train_model.py

WARNING:root:Claude Vision API analysis failed: 401 Unauthorized
→ Check ANTHROPIC_API_KEY environment variable

ERROR:root:Failed to parse Claude response as JSON
→ Claude returned invalid JSON; check image quality
```

---

## Environment Variables

### For AI-Powered Image Analysis (Optional)
```bash
# Option 1: Claude Vision API
export ANTHROPIC_API_KEY=your_key_here

# Option 2: OpenAI GPT-4 Vision
export OPENAI_API_KEY=your_key_here
```

If neither is set, the app automatically falls back to traditional CV analysis.

---

## Startup Behavior

### First Run (No Model)
```
1. App starts immediately
2. Background thread begins training
3. Training takes ~30-60 seconds
4. During training, only CV-based analysis works
5. After training completes, full NLP features available
6. If training times out (>120s), CV fallback remains active
```

### Subsequent Runs (Model Exists)
```
1. App starts immediately
2. Model loads during startup
3. Full features available right away
4. Users can generate descriptions immediately
```

---

## Testing the API

### Test Text-Based Generation
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "product_name=SmartPhone" \
  -F "category=Electronics" \
  -F "brand=TechCorp" \
  -F "key_features=5G, OLED Screen, 50MP Camera"
```

### Test Image-Based Generation
```bash
curl -X POST http://127.0.0.1:5000/predict_image \
  -F "product_image=@product.jpg"
```

### Test Error Handling
```bash
# Missing required field
curl -X POST http://127.0.0.1:5000/predict \
  -F "product_name=Phone"
# Expected: "Product Name and Category are required."

# Oversized file
# Create 15MB file and try upload
# Expected: "File size exceeds maximum of 10MB."

# Invalid image format
curl -X POST http://127.0.0.1:5000/predict_image \
  -F "product_image=@document.pdf"
# Expected: "Unsupported image type..."
```

---

## Performance & Limits

| Operation | Timeout | Limit |
|-----------|---------|-------|
| Model Training | 120 seconds | 1000 products in dataset |
| Image Upload | 30 seconds | 10MB max |
| Image Analysis (AI) | 30 seconds | 1 image at a time |
| Description Generation | 10 seconds | 5000 chars output |

---

## Troubleshooting Guide

| Problem | Check | Solution |
|---------|-------|----------|
| App won't start | Console logs | Check Python syntax: `python -m py_compile app.py` |
| Slow description generation | Model status | Run `python train_model.py` |
| Image analysis failing | File format | Use PNG, JPG, JPEG, GIF, or WEBP |
| AI analysis not working | Environment vars | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| File upload rejected | File size | Compress image to <10MB |
| Model loading error | Model files | Run `python train_model.py` |

---

## Security Features Implemented

✅ **Input Sanitization** - All user inputs cleaned
✅ **File Validation** - Images verified before processing
✅ **Size Limits** - 10MB max file size, 4000x4000 max dimensions
✅ **Type Validation** - Only allowed image formats accepted
✅ **Length Limits** - 500 character max for text fields
✅ **Error Logging** - All issues logged without exposing system details
✅ **No Unhandled Exceptions** - All errors caught and handled gracefully

---

## Next Steps

1. **Start the app:** `python app.py`
2. **Open browser:** `http://127.0.0.1:5000`
3. **Try both modes:**
   - Image Input: Upload a product image
   - Text Input: Enter product details
4. **Check logs:** Monitor console for any warnings

For detailed technical information, see `API_FIXES_SUMMARY.md`
