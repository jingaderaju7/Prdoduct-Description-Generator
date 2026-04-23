"""
AI Product Analyzer - Uses Claude Vision API to accurately detect products from images
and generate realistic product descriptions.
"""

import os
import base64
import json
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path):
    """Convert image file to base64 for API submission."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None


def analyze_product_with_ai(image_path, user_product_name="", user_category="", user_brand="", user_features=""):
    """
    Use Claude Vision API to analyze product image and generate descriptions.
    
    Args:
        image_path: Path to the product image
        user_product_name: Optional user-provided product name
        user_category: Optional user-provided category
        user_brand: Optional user-provided brand
        user_features: Optional user-provided features
        
    Returns:
        dict: Product attributes including realistic detection and description
    """
    try:
        import anthropic
    except ImportError:
        logger.debug("Anthropic library not available")
        return None  # API not available, will use fallback
    
    try:
        # Initialize Anthropic client (uses ANTHROPIC_API_KEY env variable)
        client = anthropic.Anthropic()
        
        # Encode image
        image_data = encode_image_to_base64(image_path)
        if not image_data:
            return None
        
        # Determine image media type
        file_ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(file_ext, 'image/jpeg')
        
        # Create the analysis prompt
        analysis_prompt = f"""Analyze this product image and provide a detailed product analysis in JSON format.
        
IMPORTANT: Return ONLY valid JSON, no markdown code blocks, no explanations.

Use this exact JSON structure:
{{
    "product_name": "Specific product name detected from image",
    "category": "One of: Electronics, Clothing, Home, Sports, Groceries, or Other",
    "detected_brand": "Brand if visible, otherwise infer from design",
    "primary_color": "Main color of the product",
    "confidence": 0.85,
    "key_features": [
        "Feature 1 based on visual appearance",
        "Feature 2 based on visual appearance",
        "Feature 3 based on visual design",
        "Feature 4 based on material/texture",
        "Feature 5 based on overall impression"
    ],
    "realistic_description": "2-3 sentence realistic product description that describes what you see in the image, as if written for an e-commerce site",
    "visual_analysis": {{
        "material": "What material appears to be used",
        "condition": "new/used/vintage",
        "style": "Design style observed",
        "target_audience": "Who this product is for",
        "prominent_features": "Most noticeable features"
    }},
    "unique_characteristics": [
        "Distinctive characteristic 1",
        "Distinctive characteristic 2"
    ]
}}

User context (use if provided, otherwise infer from image):
- Product Name: {user_product_name or 'Infer from image'}
- Category: {user_category or 'Infer from image'}
- Brand: {user_brand or 'Infer from image'}
- Features: {user_features or 'Infer from image'}

Focus on what you actually see in the image for realistic detection."""

        # Call Claude Vision API
        logger.info("Calling Claude Vision API for image analysis")
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": analysis_prompt
                        }
                    ],
                }
            ],
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            return None
        
        logger.info("Claude analysis completed successfully")
        # Ensure all required fields are present
        return {
            'product_name': analysis_data.get('product_name', 'Product'),
            'category': analysis_data.get('category', 'Home'),
            'brand': analysis_data.get('detected_brand', 'Premium'),
            'key_features': ', '.join(analysis_data.get('key_features', ['quality', 'design'])),
            'realistic_description': analysis_data.get('realistic_description', ''),
            'confidence': analysis_data.get('confidence', 0.85),
            'primary_color': analysis_data.get('primary_color', 'varied'),
            'visual_analysis': analysis_data.get('visual_analysis', {}),
            'unique_characteristics': analysis_data.get('unique_characteristics', []),
            'color_info': [],  # For compatibility
            'ai_generated': True
        }
        
    except Exception as e:
        logger.warning(f"Claude Vision API analysis failed: {e}")
        return None


def analyze_product_with_openai(image_path, user_product_name="", user_category="", user_brand="", user_features=""):
    """
    Alternative: Use OpenAI GPT-4 Vision API for product analysis.
    Requires OPENAI_API_KEY environment variable.
    
    Args:
        image_path: Path to the product image
        user_product_name: Optional user-provided product name
        user_category: Optional user-provided category
        user_brand: Optional user-provided brand
        user_features: Optional user-provided features
        
    Returns:
        dict: Product attributes from OpenAI analysis or None
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.debug("OpenAI library not available")
        return None
    
    try:
        client = OpenAI()
        
        # Encode image
        image_data = encode_image_to_base64(image_path)
        if not image_data:
            return None
        
        # Determine image media type
        file_ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(file_ext, 'image/jpeg')
        
        analysis_prompt = f"""Analyze this product image and provide a detailed product analysis in JSON format.

IMPORTANT: Return ONLY valid JSON, no markdown code blocks, no explanations.

Use this exact JSON structure:
{{
    "product_name": "Specific product name detected from image",
    "category": "One of: Electronics, Clothing, Home, Sports, Groceries, or Other",
    "detected_brand": "Brand if visible, otherwise infer from design",
    "primary_color": "Main color of the product",
    "confidence": 0.85,
    "key_features": [
        "Feature 1 based on visual appearance",
        "Feature 2 based on visual appearance",
        "Feature 3 based on visual design",
        "Feature 4 based on material/texture",
        "Feature 5 based on overall impression"
    ],
    "realistic_description": "2-3 sentence realistic product description that describes what you see in the image, as if written for an e-commerce site",
    "visual_analysis": {{
        "material": "What material appears to be used",
        "condition": "new/used/vintage",
        "style": "Design style observed",
        "target_audience": "Who this product is for",
        "prominent_features": "Most noticeable features"
    }},
    "unique_characteristics": [
        "Distinctive characteristic 1",
        "Distinctive characteristic 2"
    ]
}}

User context (use if provided, otherwise infer from image):
- Product Name: {user_product_name or 'Infer from image'}
- Category: {user_category or 'Infer from image'}
- Brand: {user_brand or 'Infer from image'}
- Features: {user_features or 'Infer from image'}

Focus on what you actually see in the image for realistic detection."""

        logger.info("Calling OpenAI GPT-4 Vision API for image analysis")
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": analysis_prompt
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return None
        
        logger.info("OpenAI analysis completed successfully")
        return {
            'product_name': analysis_data.get('product_name', 'Product'),
            'category': analysis_data.get('category', 'Home'),
            'brand': analysis_data.get('detected_brand', 'Premium'),
            'key_features': ', '.join(analysis_data.get('key_features', ['quality', 'design'])),
            'realistic_description': analysis_data.get('realistic_description', ''),
            'confidence': analysis_data.get('confidence', 0.85),
            'primary_color': analysis_data.get('primary_color', 'varied'),
            'visual_analysis': analysis_data.get('visual_analysis', {}),
            'unique_characteristics': analysis_data.get('unique_characteristics', []),
            'color_info': [],
            'ai_generated': True
        }
        
    except Exception as e:
        logger.warning(f"OpenAI Vision API analysis failed: {e}")
        return None


def analyze_product(image_path, user_product_name="", user_category="", user_brand="", user_features=""):
    """
    Main function to analyze product image using available AI APIs.
    Tries Claude API first, then OpenAI, then falls back to None.
    
    Args:
        image_path: Path to the product image
        user_product_name: Optional user-provided product name
        user_category: Optional user-provided category
        user_brand: Optional user-provided brand
        user_features: Optional user-provided features
        
    Returns:
        dict: Product analysis from AI or None if no API available
    """
    logger.info(f"Starting product analysis for image: {image_path}")
    
    # Try Claude API first
    result = analyze_product_with_ai(image_path, user_product_name, user_category, user_brand, user_features)
    if result:
        return result
    
    logger.info("Claude API unavailable, attempting OpenAI API...")
    # Try OpenAI API
    result = analyze_product_with_openai(image_path, user_product_name, user_category, user_brand, user_features)
    if result:
        return result
    
    # No AI API available
    logger.warning("No AI API available for analysis. Will use CV fallback.")
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = analyze_product(image_path)
        
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Error: No AI API available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
    else:
        print("Usage: python ai_product_analyzer.py <image_path>")
