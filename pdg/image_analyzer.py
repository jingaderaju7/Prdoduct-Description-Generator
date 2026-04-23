"""
Image Analyzer Module - Uses ML/CV techniques to extract product information from images.
This module analyzes uploaded product images using traditional computer vision and ML algorithms
to infer product attributes like category, dominant colors, and visual features.
"""

import numpy as np
from PIL import Image
from collections import Counter
import os

# Try to import cv2 for advanced edge/shape detection
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import sklearn for clustering, fallback to simple method if not available
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ImageAnalyzer:
    """
    Analyzes product images using traditional ML/CV techniques to extract
    meaningful features that can be used for description generation.
    """
    
    # Category color associations (based on common product colors)
    CATEGORY_COLOR_MAP = {
        'Electronics': ['black', 'white', 'silver', 'gray', 'blue'],
        'Clothing': ['red', 'blue', 'green', 'yellow', 'pink', 'purple', 'orange', 'brown'],
        'Home': ['white', 'brown', 'beige', 'gray', 'green', 'blue'],
        'Sports': ['black', 'red', 'blue', 'green', 'orange', 'yellow'],
        'Groceries': ['green', 'red', 'yellow', 'brown', 'orange', 'white'],
    }
    
    # Feature associations based on visual characteristics
    FEATURE_PATTERNS = {
        'bright': ['energy efficient', 'modern design', 'vibrant colors'],
        'dark': ['sleek design', 'professional look', 'premium finish'],
        'colorful': ['stylish', 'eye-catching', 'modern aesthetic'],
        'monochrome': ['elegant', 'minimalist design', 'classic look'],
        'high_contrast': ['bold design', 'premium quality', 'striking appearance'],
        'low_contrast': ['subtle design', 'soft finish', 'gentle aesthetic'],
        'warm_colors': ['cozy feel', 'inviting design', 'warm aesthetic'],
        'cool_colors': ['modern look', 'refreshing design', 'calming presence'],
    }
    
    def __init__(self):
        """Initialize the image analyzer."""
        self.image = None
        self.image_array = None
        
    def load_image(self, image_path):
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        try:
            self.image = Image.open(image_path)
            # Convert to RGB if necessary
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')
            self.image_array = np.array(self.image)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_image_from_pil(self, pil_image):
        """
        Load an image from a PIL Image object.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        try:
            self.image = pil_image
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')
            self.image_array = np.array(self.image)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def get_dominant_colors(self, n_colors=5):
        """
        Extract the dominant colors from the image using K-Means clustering.
        
        Args:
            n_colors: Number of dominant colors to extract
            
        Returns:
            list: List of (color_name, rgb_tuple, percentage) tuples
        """
        if self.image_array is None:
            return []
        
        # Resize image for faster processing
        small_img = self.image.resize((150, 150))
        img_array = np.array(small_img)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Remove very dark and very light pixels (likely background)
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 20) & (brightness < 235)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            filtered_pixels = pixels
        
        if HAS_SKLEARN and len(filtered_pixels) >= n_colors:
            # Determine optimal number of clusters based on unique colors
            unique_pixels = np.unique(filtered_pixels, axis=0)
            actual_n_clusters = min(n_colors, len(unique_pixels))
            actual_n_clusters = max(1, actual_n_clusters)  # At least 1 cluster
            
            # Use K-Means clustering
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate percentage of each color
            label_counts = Counter(labels)
            total = len(labels)
            
            color_info = []
            for i, color in enumerate(colors):
                percentage = (label_counts[i] / total) * 100
                color_name = self._rgb_to_color_name(tuple(color.astype(int)))
                color_info.append((color_name, tuple(color.astype(int)), percentage))
            
            # Sort by percentage
            color_info.sort(key=lambda x: x[2], reverse=True)
            return color_info
        else:
            # Fallback: simple color quantization
            return self._simple_color_extraction(filtered_pixels, n_colors)
    
    def _simple_color_extraction(self, pixels, n_colors):
        """
        Simple color extraction without sklearn.
        Uses histogram-based approach.
        """
        # Quantize colors to reduce unique values
        quantized = (pixels // 32) * 32
        quantized = np.clip(quantized, 0, 255)
        
        # Count unique colors
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1][:n_colors]
        
        color_info = []
        total = len(pixels)
        for idx in sorted_indices:
            color = unique_colors[idx]
            percentage = (counts[idx] / total) * 100
            color_name = self._rgb_to_color_name(tuple(color.astype(int)))
            color_info.append((color_name, tuple(color.astype(int)), percentage))
        
        return color_info
    
    def _rgb_to_color_name(self, rgb):
        """
        Convert RGB values to a color name.
        
        Args:
            rgb: Tuple of (R, G, B) values
            
        Returns:
            str: Color name
        """
        r, g, b = rgb
        
        # Calculate brightness
        brightness = (r + g + b) / 3
        
        # Check for grayscale first
        if max(r, g, b) - min(r, g, b) < 30:
            if brightness < 30:
                return 'black'
            elif brightness > 225:
                return 'white'
            elif brightness < 100:
                return 'gray'
            elif brightness < 180:
                return 'silver'
            else:
                return 'white'
        
        # Calculate hue-like value
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Determine dominant color
        if r >= g and r >= b:
            if r - min_val < 30:
                if brightness < 100:
                    return 'brown'
                return 'pink' if b > g else 'orange'
            if g > b + 30:
                return 'yellow' if r > 180 else 'orange'
            if r > 180:
                return 'pink' if b > min_val + 40 else 'red'
            return 'brown'
        elif g >= r and g >= b:
            if g > 180:
                return 'green' if b < r else 'lime'
            return 'green'
        else:  # b is highest
            if r > g + 30 and r > 150:
                return 'purple'
            if g > r + 30:
                return 'cyan' if b > 180 else 'teal'
            return 'blue'
    
    def detect_edges(self):
        """
        Detect edges in the image to understand shape and structure.
        
        Returns:
            tuple: (edge_count, edge_density, shape_complexity)
        """
        if self.image_array is None:
            return (0, 0, 'simple')
        
        if not HAS_CV2:
            # Fallback: use PIL-based edge detection
            try:
                from PIL import ImageFilter
                edges = self.image.filter(ImageFilter.FIND_EDGES)
                edge_array = np.array(edges)
                edge_count = np.count_nonzero(edge_array)
                edge_density = edge_count / (self.image_array.shape[0] * self.image_array.shape[1])
                
                if edge_density < 0.05:
                    complexity = 'simple'
                elif edge_density < 0.15:
                    complexity = 'moderate'
                else:
                    complexity = 'complex'
                
                return (edge_count, round(edge_density, 3), complexity)
            except:
                return (0, 0, 'simple')
        
        try:
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.count_nonzero(edges)
            edge_density = edge_count / (self.image_array.shape[0] * self.image_array.shape[1])
            
            if edge_density < 0.05:
                complexity = 'simple'
            elif edge_density < 0.15:
                complexity = 'moderate'
            else:
                complexity = 'complex'
            
            return (edge_count, round(edge_density, 3), complexity)
        except:
            return (0, 0, 'simple')
    
    def detect_shapes(self):
        """
        Detect basic shapes in the image (circular, rectangular, irregular).
        
        Returns:
            str: Dominant shape type
        """
        if self.image_array is None:
            return 'unknown'
        
        if not HAS_CV2:
            # Fallback: use PIL and numpy-based shape detection
            try:
                gray_array = np.mean(self.image_array, axis=2).astype(np.uint8)
                # Simple threshold to get binary image
                binary = (gray_array > 100).astype(np.uint8) * 255
                
                # Check image aspect ratio and variance to guess shape
                height, width = binary.shape
                aspect_ratio = max(height, width) / min(height, width)
                
                if aspect_ratio < 1.2:
                    return 'circular'  # Nearly square might be circular
                elif aspect_ratio < 2:
                    return 'rectangular'
                else:
                    return 'irregular'
            except:
                return 'unknown'
        
        try:
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'undefined'
            
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return 'undefined'
            
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            
            # Approx the contour
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            sides = len(approx)
            
            if circularity > 0.8:
                return 'circular'
            elif sides >= 4 and sides <= 6:
                return 'rectangular'
            else:
                return 'irregular'
        except:
            return 'unknown'
    
    def detect_texture(self):
        """
        Detect texture smoothness vs roughness.
        
        Returns:
            dict: Texture analysis
        """
        if self.image_array is None:
            return {'texture': 'unknown'}
        
        if not HAS_CV2:
            # Fallback: use PIL-based Laplacian
            try:
                from PIL import ImageFilter
                gray_image = self.image.convert('L')
                laplacian_img = gray_image.filter(ImageFilter.FIND_EDGES)
                laplacian = np.array(laplacian_img)
                texture_value = np.var(laplacian)
                
                if texture_value < 50:
                    texture = 'smooth'
                elif texture_value < 200:
                    texture = 'moderate'
                else:
                    texture = 'textured'
                
                return {
                    'texture': texture,
                    'value': round(texture_value, 2)
                }
            except:
                return {'texture': 'unknown', 'value': 0}
        
        try:
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_value = np.var(laplacian)
            
            if texture_value < 50:
                texture = 'smooth'
            elif texture_value < 200:
                texture = 'moderate'
            else:
                texture = 'textured'
            
            return {
                'texture': texture,
                'value': round(texture_value, 2)
            }
        except:
            return {'texture': 'unknown', 'value': 0}
    
    def analyze_brightness(self):
        """
        Analyze the overall brightness of the image.
        
        Returns:
            dict: Brightness analysis results
        """
        if self.image_array is None:
            return {'brightness': 'unknown', 'value': 0}
        
        # Calculate average brightness
        gray = np.mean(self.image_array, axis=2)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 85:
            level = 'dark'
        elif avg_brightness < 170:
            level = 'medium'
        else:
            level = 'bright'
        
        return {
            'brightness': level,
            'value': round(avg_brightness, 2)
        }
    
    def analyze_contrast(self):
        """
        Analyze the contrast of the image.
        
        Returns:
            dict: Contrast analysis results
        """
        if self.image_array is None:
            return {'contrast': 'unknown', 'value': 0}
        
        # Calculate contrast using standard deviation
        gray = np.mean(self.image_array, axis=2)
        contrast_value = np.std(gray)
        
        if contrast_value < 40:
            level = 'low'
        elif contrast_value < 75:
            level = 'medium'
        else:
            level = 'high'
        
        return {
            'contrast': level,
            'value': round(contrast_value, 2)
        }
    
    def analyze_color_temperature(self):
        """
        Analyze whether the image has warm or cool colors.
        
        Returns:
            dict: Color temperature analysis
        """
        if self.image_array is None:
            return {'temperature': 'neutral'}
        
        r, g, b = self.image_array[:,:,0], self.image_array[:,:,1], self.image_array[:,:,2]
        
        # Calculate average warmth (red-yellow vs blue)
        warm_score = np.mean(r) * 0.5 + np.mean(g) * 0.3 - np.mean(b) * 0.4
        
        if warm_score > 20:
            temperature = 'warm'
        elif warm_score < -20:
            temperature = 'cool'
        else:
            temperature = 'neutral'
        
        return {
            'temperature': temperature,
            'warm_score': round(warm_score, 2)
        }
    
    def analyze_color_diversity(self):
        """
        Analyze the color diversity/variety in the image.
        
        Returns:
            dict: Color diversity analysis
        """
        if self.image_array is None:
            return {'diversity': 'unknown'}
        
        # Quantize and count unique colors
        quantized = (self.image_array // 16) * 16
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        # Normalize (max possible = 4096 for 16-level quantization)
        diversity_ratio = unique_colors / 4096
        
        if diversity_ratio < 0.1:
            level = 'monochrome'
        elif diversity_ratio < 0.3:
            level = 'low_diversity'
        elif diversity_ratio < 0.6:
            level = 'medium_diversity'
        else:
            level = 'colorful'
        
        return {
            'diversity': level,
            'unique_colors': unique_colors,
            'ratio': round(diversity_ratio, 3)
        }
    
    def predict_category(self):
        """
        Predict product category based on comprehensive visual analysis.
        
        Returns:
            tuple: (predicted_category, confidence_score)
        """
        if self.image_array is None:
            return ('Home', 0.3)
        
        # Get dominant colors
        dominant_colors = self.get_dominant_colors(3)
        
        if not dominant_colors:
            return ('Home', 0.3)
        
        # Score each category based on color match
        category_scores = {}
        
        for category, category_colors in self.CATEGORY_COLOR_MAP.items():
            score = 0
            for color_name, _, percentage in dominant_colors:
                if color_name in category_colors:
                    score += percentage
            category_scores[category] = score
        
        # Get advanced analysis
        brightness = self.analyze_brightness()
        color_temp = self.analyze_color_temperature()
        diversity = self.analyze_color_diversity()
        shape = self.detect_shapes()
        texture = self.detect_texture()
        edge_count, edge_density, complexity = self.detect_edges()
        
        # Shape-based heuristics
        if shape == 'rectangular':
            category_scores['Electronics'] *= 1.5  # Rectangles are common in electronics
            category_scores['Home'] *= 1.3
        elif shape == 'circular':
            category_scores['Sports'] *= 1.4
            category_scores['Home'] *= 1.2
        
        # Texture-based heuristics
        if texture['texture'] == 'smooth':
            category_scores['Electronics'] *= 1.3
            category_scores['Clothing'] *= 1.2
        elif texture['texture'] == 'textured':
            category_scores['Groceries'] *= 1.3
            category_scores['Clothing'] *= 1.2
        
        # Complexity-based heuristics
        if complexity == 'complex':
            category_scores['Clothing'] *= 1.3
            category_scores['Sports'] *= 1.2
        elif complexity == 'simple':
            category_scores['Electronics'] *= 1.2
        
        # Color temperature heuristics
        if brightness['brightness'] in ['dark', 'bright'] and color_temp['temperature'] == 'cool':
            category_scores['Electronics'] *= 1.3
        
        if color_temp['temperature'] == 'warm' and diversity['diversity'] in ['colorful', 'medium_diversity']:
            category_scores['Groceries'] *= 1.3
        
        if brightness['brightness'] == 'bright':
            category_scores['Sports'] *= 1.2
        
        if diversity['diversity'] in ['colorful', 'medium_diversity']:
            category_scores['Clothing'] *= 1.2
        
        # Get best category
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category] / 100, 0.9)
        
        return (best_category, round(confidence, 2))
    
    def extract_features(self):
        """
        Extract visual features that can be used in product descriptions.
        Enhanced to generate more accurate product-specific features.
        
        Returns:
            list: List of extracted feature strings
        """
        if self.image_array is None:
            return ['quality design', 'modern style']
        
        features = []
        
        # Analyze various aspects
        brightness = self.analyze_brightness()
        contrast = self.analyze_contrast()
        color_temp = self.analyze_color_temperature()
        diversity = self.analyze_color_diversity()
        texture = self.detect_texture()
        shape = self.detect_shapes()
        edge_count, edge_density, complexity = self.detect_edges()
        dominant_colors = self.get_dominant_colors(3)
        category, _ = self.predict_category()
        
        # Texture-based features
        if texture['texture'] == 'smooth':
            features.extend(['smooth finish', 'polished design', 'refined texture'])
        elif texture['texture'] == 'textured':
            features.extend(['textured surface', 'material depth', 'tactile feel'])
        else:
            features.extend(['balanced finish', 'professional grade'])
        
        # Shape-based features
        if shape == 'rectangular':
            features.extend(['geometric design', 'structured form', 'clean lines'])
        elif shape == 'circular':
            features.extend(['curved design', 'ergonomic shape', 'rounded edges'])
        
        # Complexity-based features
        if complexity == 'complex':
            features.extend(['intricate details', 'sophisticated design'])
        elif complexity == 'simple':
            features.extend(['minimalist aesthetic', 'simple elegance'])
        
        # Brightness-based features - category specific
        if brightness['brightness'] == 'dark':
            if category == 'Electronics':
                features.extend(['sleek black finish', 'premium look', 'professional aesthetic'])
            elif category == 'Clothing':
                features.extend(['classic dark tones', 'versatile style'])
            else:
                features.extend(['elegant dark finish', 'sophisticated appearance'])
        elif brightness['brightness'] == 'bright':
            if category == 'Groceries' or category == 'Clothing':
                features.extend(['vibrant colors', 'eye-catching design', 'fresh appearance'])
            else:
                features.extend(['bright finish', 'modern aesthetic'])
        
        # Contrast-based features
        if contrast['contrast'] == 'high':
            features.extend(['bold design', 'striking appearance', 'high visual impact'])
        elif contrast['contrast'] == 'low':
            features.extend(['subtle design', 'elegant finish', 'sophisticated tone'])
        
        # Color temperature features
        if color_temp['temperature'] == 'warm':
            features.extend(['warm tones', 'inviting appearance', 'comfortable feel'])
        elif color_temp['temperature'] == 'cool':
            features.extend(['cool aesthetic', 'modern design', 'contemporary style'])
        
        # Color diversity features
        if diversity['diversity'] == 'monochrome':
            features.extend(['minimalist design', 'classic elegance', 'timeless style'])
        elif diversity['diversity'] == 'colorful':
            features.extend(['vibrant colors', 'stylish appearance', 'dynamic design'])
        elif diversity['diversity'] == 'medium_diversity':
            features.extend(['balanced colors', 'versatile tone'])
        
        # Category-specific feature additions
        if category == 'Electronics':
            features.extend(['tech-forward', 'reliable performance', 'cutting-edge design'])
        elif category == 'Clothing':
            features.extend(['fashionable', 'quality construction', 'wearable comfort'])
        elif category == 'Sports':
            features.extend(['athletic design', 'performance-oriented', 'active lifestyle'])
        elif category == 'Groceries':
            features.extend(['natural ingredients', 'quality product', 'freshness'])
        elif category == 'Home':
            features.extend(['home-friendly', 'durable construction', 'comfort focused'])
        
        # Color-specific features
        color_names = [c[0] for c in dominant_colors[:2]]
        if 'black' in color_names:
            features.append('sleek black finish')
        if 'white' in color_names:
            features.append('clean white design')
        if 'silver' in color_names or 'gray' in color_names:
            features.append('modern metallic look')
        if 'red' in color_names:
            features.append('bold red accent')
        if 'blue' in color_names:
            features.append('professional blue tone')
        
        # Removal of duplicates and limit to top 5-7 features
        unique_features = list(dict.fromkeys(features))
        return unique_features[:7] if unique_features else ['quality design', 'modern style']
    
    def generate_product_attributes(self):
        """
        Generate complete product attributes from image analysis.
        This is the main method to call for image-based description generation.
        Includes comprehensive visual analysis.
        
        Returns:
            dict: Dictionary containing inferred product attributes
        """
        if self.image_array is None:
            return {
                'category': 'Home',
                'key_features': 'quality design, modern style',
                'color_info': [],
                'confidence': 0.0
            }
        
        # Get dominant colors
        dominant_colors = self.get_dominant_colors(5)
        primary_color = dominant_colors[0][0] if dominant_colors else 'unknown'
        
        # Predict category
        category, confidence = self.predict_category()
        
        # Extract features
        features = self.extract_features()
        
        # Get all analysis data
        shape = self.detect_shapes()
        texture = self.detect_texture()
        brightness_data = self.analyze_brightness()
        
        # Generate brand and product name based on comprehensive analysis
        brand_prefixes = {
            'black': ['Dark', 'Shadow', 'Midnight', 'Onyx', 'Black'],
            'white': ['Pure', 'Crystal', 'Snow', 'Ivory', 'White'],
            'silver': ['Silver', 'Platinum', 'Steel', 'Chrome', 'Metallic'],
            'gray': ['Graphite', 'Ash', 'Slate', 'Stone', 'Neutral'],
            'blue': ['Azure', 'Ocean', 'Sky', 'Cobalt', 'Sapphire'],
            'red': ['Crimson', 'Ruby', 'Scarlet', 'Fire', 'Blaze'],
            'green': ['Forest', 'Emerald', 'Jade', 'Nature', 'Verdant'],
            'brown': ['Earth', 'Wood', 'Bronze', 'Copper', 'Timber'],
            'yellow': ['Gold', 'Solar', 'Amber', 'Bright', 'Radiant'],
        }
        
        # Category-specific suffixes
        category_suffixes = {
            'Electronics': ['Tech', 'Pro', 'Digital', 'Smart', 'Elite', 'Nexus'],
            'Clothing': ['Style', 'Fashion', 'Wear', 'Collection', 'Thread', 'Design'],
            'Sports': ['Sport', 'Athletic', 'Pro', 'Active', 'Peak', 'Force'],
            'Home': ['Home', 'Living', 'Comfort', 'Style', 'Craft', 'Design'],
            'Groceries': ['Fresh', 'Pure', 'Natural', 'Select', 'Prime', 'Quality'],
        }
        
        prefix_list = brand_prefixes.get(primary_color, ['Premium', 'Elite', 'Prime', 'Pro'])
        suffix_list = category_suffixes.get(category, ['Design', 'Tech', 'Style', 'Pro', 'Works'])
        
        import random
        random.seed(hash(primary_color + category) % 1000)
        brand = random.choice(prefix_list) + random.choice(suffix_list)
        
        # Generate more descriptive product name
        product_descriptors = {
            'rectangular': ['Premium', 'Standard', 'Compact'],
            'circular': ['Round', 'Spherical', 'Curved'],
            'irregular': ['Ergonomic', 'Design', 'Professional']
        }
        
        descriptor = product_descriptors.get(shape, ['Classic'])[0]
        
        if category == 'Electronics':
            product_type = 'Device'
        elif category == 'Clothing':
            product_type = 'Collection' if brightness_data['brightness'] == 'colorful' else 'Item'
        elif category == 'Sports':
            product_type = 'Equipment'
        elif category == 'Groceries':
            product_type = 'Product'
        else:
            product_type = 'Item'
        
        product_name = f"{descriptor} {category} {product_type}"
        
        return {
            'category': category,
            'key_features': ', '.join(features[:4]),
            'color_info': dominant_colors,
            'primary_color': primary_color,
            'brand': brand,
            'confidence': confidence,
            'brightness': brightness_data,
            'contrast': self.analyze_contrast(),
            'color_temperature': self.analyze_color_temperature(),
            'color_diversity': self.analyze_color_diversity(),
            'texture': texture,
            'shape': shape,
            'product_name': product_name
        }


def analyze_image_file(image_path):
    """
    Convenience function to analyze an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Product attributes extracted from the image
    """
    analyzer = ImageAnalyzer()
    if analyzer.load_image(image_path):
        return analyzer.generate_product_attributes()
    return None


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = analyze_image_file(image_path)
        
        if result:
            print("\n=== Image Analysis Results ===")
            print(f"Predicted Category: {result['category']} (confidence: {result['confidence']})")
            print(f"Inferred Brand: {result['brand']}")
            print(f"Key Features: {result['key_features']}")
            print(f"Primary Color: {result['primary_color']}")
            print(f"\nDominant Colors:")
            for color_name, rgb, pct in result['color_info']:
                print(f"  - {color_name}: RGB{rgb} ({pct:.1f}%)")
            print(f"\nBrightness: {result['brightness']['brightness']} ({result['brightness']['value']})")
            print(f"Contrast: {result['contrast']['contrast']} ({result['contrast']['value']})")
            print(f"Color Temperature: {result['color_temperature']['temperature']}")
            print(f"Color Diversity: {result['color_diversity']['diversity']}")
        else:
            print("Failed to analyze image.")
    else:
        print("Usage: python image_analyzer.py <image_path>")
