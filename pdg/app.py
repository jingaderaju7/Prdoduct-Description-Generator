import os
import uuid
import logging
from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from generate_description import generate_description
from image_analyzer import ImageAnalyzer
from ai_product_analyzer import analyze_product

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIMENSION = 4000  # pixels
MAX_TEXT_LENGTH = 500  # characters

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename: str) -> bool:
    """Validate file extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_path: str) -> tuple[bool, str]:
    """
    Validate image file integrity and dimensions.
    Returns: (is_valid, error_message)
    """
    try:
        img = Image.open(file_path)
        width, height = img.size
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image dimensions {width}x{height} exceed maximum of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}."
        
        # Verify image can be opened (detects corruption)
        img.verify()
        return True, ""
    except Exception as e:
        logger.warning(f"Image validation error: {e}")
        return False, f"Image file is corrupted or invalid: {str(e)}"


def sanitize_text_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> tuple[str, bool]:
    """
    Sanitize user input: strip whitespace, escape HTML, enforce length limits.
    Returns: (sanitized_text, is_valid)
    """
    if not text:
        return "", True
    
    text = str(text).strip()
    
    if len(text) > max_length:
        return text[:max_length], False
    
    return text, True


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle text-based form submission
@app.route('/predict', methods=['POST'])
def predict():
    """Text-based description generation with comprehensive error handling."""
    try:
        # Validate required fields
        product_name = request.form.get('product_name', '').strip()
        category = request.form.get('category', '').strip()
        
        if not product_name or not category:
            return render_template('index.html', 
                                 error="Product Name and Category are required.")
        
        # Sanitize inputs
        product_name, name_valid = sanitize_text_input(product_name)
        category, cat_valid = sanitize_text_input(category)
        brand, _ = sanitize_text_input(request.form.get('brand', ''))
        key_features, _ = sanitize_text_input(request.form.get('key_features', ''))
        existing_image = request.form.get('existing_image', '').strip()
        
        image_file = request.files.get('product_image')
        image_filename = ''
        image_url = None
        image_keywords = ''

        # Handle image upload or reuse
        if image_file and image_file.filename:
            if not allowed_file(image_file.filename):
                return render_template('index.html',
                                     error="Unsupported image type. Allowed: PNG, JPG, JPEG, GIF, WEBP.",
                                     product_name=product_name,
                                     category=category,
                                     brand=brand,
                                     key_features=key_features)

            try:
                original_name = secure_filename(image_file.filename)
                ext = original_name.rsplit('.', 1)[1].lower()
                image_filename = f"{uuid.uuid4().hex}.{ext}"
                save_path = os.path.join(UPLOAD_FOLDER, image_filename)
                image_file.save(save_path)
                
                # Validate image
                is_valid, error_msg = validate_image_file(save_path)
                if not is_valid:
                    os.remove(save_path)
                    return render_template('index.html',
                                         error=f"Image validation failed: {error_msg}",
                                         product_name=product_name,
                                         category=category,
                                         brand=brand,
                                         key_features=key_features)
                
                image_url = url_for('static', filename=f'uploads/{image_filename}')
                base_name = os.path.splitext(original_name)[0]
                image_keywords = " ".join(base_name.replace('_', ' ').replace('-', ' ').split())
                
            except Exception as e:
                logger.error(f"Image upload error: {e}")
                return render_template('index.html',
                                     error="Failed to process image upload. Please try again.",
                                     product_name=product_name,
                                     category=category,
                                     brand=brand,
                                     key_features=key_features)
        elif existing_image:
            image_filename = os.path.basename(existing_image)
            image_url = url_for('static', filename=f'uploads/{image_filename}')

        try:
            # Generate description
            combined_features = key_features
            if image_keywords:
                combined_features = f"{key_features} {image_keywords}".strip()

            description = generate_description(product_name, category, brand, combined_features)
            
            # Check if description generation failed
            if not description or "error" in description.lower():
                logger.error(f"Description generation failed: {description}")
                return render_template('index.html',
                                     error="Failed to generate description. Please try again or check the model setup.",
                                     product_name=product_name,
                                     category=category,
                                     brand=brand,
                                     key_features=key_features)
            
            return render_template('index.html', 
                                 description=description,
                                 product_name=product_name,
                                 category=category,
                                 brand=brand,
                                 key_features=key_features,
                                 image_url=image_url,
                                 image_filename=image_filename or existing_image)
        except Exception as e:
            logger.error(f"Description generation error: {e}")
            return render_template('index.html',
                                 error="Error generating description. Please try again.",
                                 product_name=product_name,
                                 category=category,
                                 brand=brand,
                                 key_features=key_features)

    except RequestEntityTooLarge:
        logger.error("File upload exceeded size limit")
        return render_template('index.html', 
                             error=f"File size exceeds maximum of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB.")
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}")
        return render_template('index.html', 
                             error="An unexpected error occurred. Please try again.")


# Route to handle image-based form submission
@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Image-based description generation with comprehensive error handling."""
    try:
        image_file = request.files.get('product_image')
        
        # Check if image was uploaded
        if not image_file or not image_file.filename:
            return render_template('index.html', error="Please upload a product image.")
        
        if not allowed_file(image_file.filename):
            return render_template('index.html', 
                                 error="Unsupported image type. Allowed: PNG, JPG, JPEG, GIF, WEBP.")
        
        try:
            # Save the uploaded image
            original_name = secure_filename(image_file.filename)
            ext = original_name.rsplit('.', 1)[1].lower()
            image_filename = f"{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(save_path)
            
            # Validate image
            is_valid, error_msg = validate_image_file(save_path)
            if not is_valid:
                os.remove(save_path)
                return render_template('index.html', 
                                     error=f"Image validation failed: {error_msg}")
            
            image_url = url_for('static', filename=f'uploads/{image_filename}')
            
        except Exception as e:
            logger.error(f"Image save/validation error: {e}")
            return render_template('index.html', 
                                 error="Failed to process the uploaded image.")
        
        # Get and sanitize optional user inputs
        user_product_name, _ = sanitize_text_input(request.form.get('product_name', ''))
        user_category, _ = sanitize_text_input(request.form.get('category', ''))
        user_brand, _ = sanitize_text_input(request.form.get('brand', ''))
        user_features, _ = sanitize_text_input(request.form.get('key_features', ''))
        
        try:
            # Try AI-powered analysis first (Claude or OpenAI Vision)
            logger.info("Attempting AI-powered product analysis...")
            image_attrs = analyze_product(save_path, user_product_name, user_category, 
                                        user_brand, user_features)
            
            # If AI analysis fails, fallback to traditional CV-based analyzer
            if not image_attrs:
                logger.info("AI analysis unavailable, using traditional CV-based analysis...")
                analyzer = ImageAnalyzer()
                if not analyzer.load_image(save_path):
                    os.remove(save_path)
                    return render_template('index.html', 
                                         error="Failed to process the uploaded image. Please try another image.")
                
                image_attrs = analyzer.generate_product_attributes()
            else:
                logger.info("AI analysis successful!")
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return render_template('index.html', 
                                 error="Error analyzing image. Please try again or try a different image.")
        
        try:
            # Use detected values if user didn't provide them
            product_name = user_product_name if user_product_name else image_attrs.get('product_name', 'Premium Product Item')
            category = user_category if user_category else image_attrs.get('category', 'Home')
            brand = user_brand if user_brand else image_attrs.get('brand', 'TechPro')
            
            # Combine detected features with user-provided features
            detected_features = image_attrs.get('key_features', '')
            if user_features:
                combined_features = f"{user_features}, {detected_features}"
            else:
                combined_features = detected_features
            
            # Use AI-generated description if available, otherwise generate with NLP model
            if image_attrs.get('ai_generated') and image_attrs.get('realistic_description'):
                description = image_attrs.get('realistic_description')
            else:
                description = generate_description(product_name, category, brand, combined_features)
            
            # Check if description generation failed
            if not description or "error" in description.lower():
                logger.error(f"Description generation failed: {description}")
                return render_template('index.html',
                                     error="Failed to generate description. Please try again.")
            
            # Return result with image analysis info
            return render_template('index.html',
                                 description=description,
                                 product_name=product_name,
                                 category=category,
                                 brand=brand,
                                 key_features=combined_features,
                                 image_url=image_url,
                                 image_filename=image_filename,
                                 image_analysis=image_attrs,
                                 ai_powered=image_attrs.get('ai_generated', False))
        except Exception as e:
            logger.error(f"Description processing error: {e}")
            return render_template('index.html',
                                 error="Error processing image analysis. Please try again.")

    except RequestEntityTooLarge:
        logger.error("File upload exceeded size limit")
        return render_template('index.html', 
                             error=f"File size exceeds maximum of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB.")
    except Exception as e:
        logger.error(f"Unexpected error in /predict_image: {e}")
        return render_template('index.html', 
                             error="An unexpected error occurred. Please try again.")


if __name__ == '__main__':
    # Non-blocking model training with timeout
    import subprocess
    import threading
    import time
    
    def train_model_async():
        """Train model in background without blocking app startup."""
        if not os.path.exists('models/nearest_neighbors_model.pkl'):
            logger.info("Model files not found. Starting async training...")
            try:
                # Run training with timeout (120 seconds)
                result = subprocess.run(
                    ['python', 'train_model.py'],
                    capture_output=True,
                    timeout=120,
                    text=True
                )
                if result.returncode == 0:
                    logger.info("Model training completed successfully")
                else:
                    logger.error(f"Model training failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Model training timed out. The app will use CV fallback for image analysis.")
            except Exception as e:
                logger.error(f"Error during async model training: {e}")
        else:
            logger.info("Model files already exist")
    
    # Start model training in background thread
    training_thread = threading.Thread(target=train_model_async, daemon=True)
    training_thread.start()
    
    # Start Flask app immediately
    app.run(debug=True, port=5000)
