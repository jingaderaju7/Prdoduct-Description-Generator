# Product Description Generator (PDG)

A Flask web application that generates product descriptions using ML/NLP techniques. The application supports both **image-based** and **text-based** input methods for generating compelling product descriptions.

---

## Features

- **Dual Input Methods:**
  - **Image Input:** Upload a product image and let the system analyze it using ML/CV techniques to extract product attributes
  - **Text Input:** Manually enter product details (name, category, brand, features)

- **Image Analysis (ML/CV):**
  - Dominant color extraction using K-Means clustering
  - Brightness and contrast analysis
  - Color temperature detection (warm/cool)
  - Color diversity measurement
  - Automatic category prediction based on visual features
  - Feature inference from visual characteristics

- **Description Generation (NLP):**
  - TF-IDF vectorization for text representation
  - K-Nearest Neighbors for finding similar products
  - Template-based description expansion
  - 200-300 word professional descriptions

---

## Tech Stack

- **Frontend:** HTML5, CSS3
- **Backend:** Python Flask
- **ML/NLP:**
  - scikit-learn (TF-IDF, KNN, K-Means)
  - NLTK (NLP preprocessing)
  - NumPy (numerical operations)
  - Pillow (image processing)

---

## Project Structure

```
product-description-generator/
├── app.py                    # Main Flask application
├── train_model.py           # Model training script
├── generate_description.py  # NLP description generator
├── image_analyzer.py        # Image analysis module (ML/CV)
├── dataset.csv              # Training data
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
├── models/                  # Generated model files
│   ├── tfidf_vectorizer.pkl
│   ├── nearest_neighbors_model.pkl
│   └── processed_data.pkl
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   └── style.css           # Styling
└── static/uploads/         # Uploaded images
```

---

## Installation & Setup

### 1. Create Virtual Environment

```bash
cd product-description-generator
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This will create the model files in the `models/` directory.

### 4. Run the Application

```bash
python app.py
```

### 5. Access the Application

Open your browser and navigate to: `http://127.0.0.1:5000`

---

## Usage

### Image Input Mode

1. Click on **"Image Input"** card
2. Upload a product image (drag & drop or click to browse)
3. Optionally add extra details (product name, category, brand, features)
4. Click **"Generate Description from Image"**
5. View the generated description along with image analysis results

### Text Input Mode

1. Click on **"Text Input"** card
2. Fill in the required fields:
   - **Product Name** (required)
   - **Category** (required)
   - Brand (optional)
   - Key Features (optional)
3. Optionally upload a product image
4. Click **"Generate Description"**
5. View the generated description

---

## Image Analysis Details

The image analyzer uses traditional ML/CV techniques:

### Color Analysis
- **K-Means Clustering:** Groups similar colors to find dominant ones
- **Color Name Mapping:** Converts RGB values to human-readable names
- **Percentage Calculation:** Shows how much each color appears

### Visual Feature Extraction
- **Brightness:** Analyzes overall lightness/darkness
- **Contrast:** Measures color variation
- **Color Temperature:** Determines warm vs cool tones
- **Color Diversity:** Counts unique colors present

### Category Prediction
Based on color associations:
- **Electronics:** black, white, silver, gray, blue
- **Clothing:** red, blue, green, yellow, pink, purple
- **Home:** white, brown, beige, gray, green
- **Sports:** black, red, blue, green, orange
- **Groceries:** green, red, yellow, brown, orange

---

## Customization

### Adding New Categories

Edit `image_analyzer.py` and update the `CATEGORY_COLOR_MAP`:

```python
CATEGORY_COLOR_MAP = {
    'Electronics': ['black', 'white', 'silver', 'gray', 'blue'],
    'YourCategory': ['color1', 'color2', 'color3'],
    # ...
}
```

### Retraining with New Data

1. Add new products to `dataset.csv`
2. Run `python train_model.py`
3. Restart the Flask application

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with mode selection |
| `/predict` | POST | Text-based description generation |
| `/predict_image` | POST | Image-based description generation |

---

## Troubleshooting

**Model not found error:**
- Run `python train_model.py` to generate model files

**Image upload fails:**
- Ensure the image format is PNG, JPG, JPEG, GIF, or WEBP
- Check file size (large images may take longer to process)

**Description quality issues:**
- For image input: try uploading clearer product images
- For text input: provide more specific features
- Consider retraining with updated dataset

---

## Requirements

- Python 3.8+
- Flask
- pandas
- scikit-learn
- nltk
- numpy
- Pillow

---

## License

This project is open source and available for educational and personal use.
