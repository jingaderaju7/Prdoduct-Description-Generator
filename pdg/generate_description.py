import pickle
import pandas as pd
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load artifacts globally to avoid loading them on every request
def load_artifacts():
    artifacts = {}
    try:
        # Get the directory of the current script to construct correct paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        model_path = os.path.join(models_dir, 'nearest_neighbors_model.pkl')
        data_path = os.path.join(models_dir, 'processed_data.pkl')
        
        with open(vectorizer_path, 'rb') as f:
            artifacts['vectorizer'] = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            artifacts['model'] = pickle.load(f)
            
        artifacts['df'] = pd.read_pickle(data_path)
        logger.info("NLP artifacts loaded successfully")
        return artifacts
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}. Please run train_model.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None

# Load once when module is imported
artifacts = load_artifacts()

def clean_text_input(text):
    """Helper to clean user input matching training data style."""
    if not text:
        return ""
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"Text cleaning error: {e}")
        return ""

def generate_description(product_name, category, brand, key_features):
    """
    Generates a description by finding the most similar product
    in the dataset and adapting its description.
    
    Args:
        product_name: Name of the product
        category: Product category
        brand: Brand name
        key_features: Comma-separated features
        
    Returns:
        str: Generated product description
    """
    try:
        if artifacts is None:
            logger.error("Model artifacts not loaded. Cannot generate description.")
            return "Error: Model not trained. Please run train_model.py first."
        
        # Validate inputs
        if not product_name or not category:
            logger.warning("Missing required fields: product_name or category")
            return "Error: Product name and category are required."
        
        # 1. Prepare the input string (same way we trained)
        clean_name = clean_text_input(product_name)
        clean_cat = clean_text_input(category)
        clean_brand = clean_text_input(brand)
        clean_feat = clean_text_input(key_features)
        
        # Combine inputs into a single string for prediction
        input_soup = (clean_name + " " + clean_cat + " " + 
                      clean_brand + " " + clean_feat + " " + clean_cat)
        
        if not input_soup.strip():
            logger.warning("Input soup is empty after cleaning")
            return "Error: Could not process inputs."
        
        # 2. Vectorize the input
        input_vector = artifacts['vectorizer'].transform([input_soup])
        
        # 3. Find the nearest neighbor
        distances, indices = artifacts['model'].kneighbors(input_vector)
        matched_index = indices[0][0]
        
        # 4. Retrieve the base description safely
        try:
            base_description = artifacts['df'].iloc[matched_index]['description']
            matched_product_name = artifacts['df'].iloc[matched_index]['product_name']
        except (KeyError, IndexError) as e:
            logger.error(f"Error retrieving matched product: {e}")
            return "Error: Could not match a suitable product."
        
        # 5. Adapt the matched description
        adapted = str(base_description) if base_description else product_name
        
        # Strategy 1: Exact match of product name in description
        if matched_product_name and matched_product_name.lower() in adapted.lower():
            adapted = re.sub(re.escape(matched_product_name), product_name, adapted, flags=re.IGNORECASE)
        
        # Strategy 2: Try partial matches from the matched product name
        if matched_product_name:
            matched_words = matched_product_name.split()
            replaced = False
            for length in range(len(matched_words), 0, -1):
                for start in range(len(matched_words) - length + 1):
                    phrase = ' '.join(matched_words[start:start+length])
                    if phrase.lower() in adapted.lower():
                        adapted = re.sub(re.escape(phrase), product_name, adapted, flags=re.IGNORECASE, count=1)
                        replaced = True
                        break
                if replaced:
                    break
        
        # Strategy 3: If product name still not in description, prepend it
        if product_name.lower() not in adapted.lower():
            adapted = f"This is {product_name}. {adapted}"
        
        # Replace any remaining matched brand references with user's brand if provided
        if brand:
            try:
                matched_brand = artifacts['df'].iloc[matched_index].get('brand', '')
                if matched_brand and matched_brand.lower() in adapted.lower():
                    if brand.lower() not in adapted.lower():
                        adapted = re.sub(re.escape(matched_brand), brand, adapted, flags=re.IGNORECASE)
                    else:
                        adapted = re.sub(re.escape(matched_brand) + r'\s*', '', adapted, flags=re.IGNORECASE)
                        adapted = re.sub(r'\s+', ' ', adapted)
            except Exception as e:
                logger.warning(f"Brand replacement error: {e}")

        # Build description with additional context
        def _normalize(text):
            return str(text).strip() if text else ""

        clean_brand = _normalize(brand)
        clean_category = _normalize(category)
        clean_features = [f.strip() for f in key_features.split(',') if f.strip()] if key_features else []

        # Remove redundant intro if base description already begins with it
        adapted_text = adapted
        if adapted_text.lower().startswith("introducing"):
            parts = adapted_text.split('. ', 1)
            if len(parts) > 1:
                adapted_text = parts[1].strip()

        # Break the adapted text into multiple sentences/paragraphs
        adapted_sentences = []
        for s in re.split(r'(?<=[.!?])\s+', adapted_text):
            s = s.strip()
            if not s:
                continue
            if s[-1] not in '.!?':
                s += '.'
            adapted_sentences.append(s)

        # Build an intro sentence
        intro = f"Introducing {product_name}"
        if clean_brand:
            intro += f" from {clean_brand}"

        if clean_category:
            lower_cat = clean_category.lower()
            article = "an" if lower_cat and lower_cat[0] in "aeiou" else "a"
            intro += f" — {article} {lower_cat} solution"

        intro_sentence = intro.strip()
        if not intro_sentence.endswith('.'):
            intro_sentence += '.'

        # Add a feature-focused sentence if any key features were listed
        feature_sentence = ""
        if clean_features:
            if len(clean_features) == 1:
                feature_sentence = f"It stands out thanks to {clean_features[0]}."
            elif len(clean_features) == 2:
                feature_sentence = f"It stands out thanks to {clean_features[0]} and {clean_features[1]}."
            else:
                feature_sentence = (
                    f"It stands out thanks to {', '.join(clean_features[:-1])}, and {clean_features[-1]}.")

        # Add a practical use-case sentence to make it feel actionable
        usage_sentence = ""
        if clean_features or clean_category:
            main_feature = clean_features[0] if clean_features else None
            category_phrase = f" in {clean_category.lower()}" if clean_category else ""

            if main_feature:
                usage_sentence = f"Great for users who want {main_feature}{category_phrase} without the hassle."
            else:
                usage_sentence = f"Perfect for anyone looking to elevate their{category_phrase} experience."  

        # Add a closing sentence for stronger realism
        closing_sentence = ""
        if clean_category:
            lower_cat = clean_category.lower()
            article = "an" if lower_cat and lower_cat[0] in "aeiou" else "a"
            closing_sentence = f"Perfect for anyone looking for {article} {lower_cat} solution that delivers real value."
        elif clean_brand:
            closing_sentence = f"Built for people who trust {clean_brand} for consistent performance."

        # Combine into a single-paragraph description
        description_sentences = [intro_sentence]
        description_sentences.extend(adapted_sentences)
        if feature_sentence:
            description_sentences.append(feature_sentence)
        if usage_sentence:
            description_sentences.append(usage_sentence)
        if closing_sentence:
            description_sentences.append(closing_sentence)

        def word_count(text: str) -> int:
            return len([w for w in re.findall(r"\w+", text)])

        def make_sentence(template: str) -> str:
            return template.format(
                name=product_name,
                brand=clean_brand or "",
                category=clean_category or "",
                features=", ".join(clean_features) if clean_features else "",
                first_feature=clean_features[0] if clean_features else "",
                second_feature=clean_features[1] if len(clean_features) > 1 else "",
            )

        extra_templates = [
            "{name} is built to deliver seamless performance in everyday use, making it a reliable choice when you need consistent results.",
            "With {features}, this product keeps you ahead of the curve and ensures you get the most out of your investment.",
            "Every detail of {name} has been refined, from the ergonomic design to the intuitive controls, so you can focus on what matters.",
            "Whether you are using it at home or on the go, {name} provides the flexibility and power you expect from {brand}.",
            "Its advanced design and thoughtful engineering make {name} feel premium and dependable every time you use it.",
            "When paired with modern workflows, {name} adapts easily and helps you maintain productivity without trade-offs.",
            "Customers often praise how {name} delivers exceptional value, thanks to its combination of quality, performance, and ease of use.",
            "If you want a product that feels tailored to your needs, {name} offers a polished experience right out of the box.",
            "From the moment you start using {name}, you'll notice the attention to detail and how well it fits into your routine.",
            "For anyone looking to upgrade, {name} offers a compelling balance of features and cost that stands out in its category.",
        ]

        # Build base description and extend until we hit between 200 and 300 words.
        paragraph_index = 0
        max_loops = 25

        while word_count(" ".join(description_sentences)) < 200 and paragraph_index < max_loops:
            template = extra_templates[paragraph_index % len(extra_templates)]
            description_sentences.append(make_sentence(template))
            paragraph_index += 1

        final_description = " ".join([s.strip() for s in description_sentences if s])

        # Trim to max 300 words if needed
        words = re.findall(r"\w+", final_description)
        if len(words) > 300:
            final_description = " ".join(words[:300])
            if not final_description.endswith('.'):
                final_description = final_description.rstrip(',') + '.'

        return final_description
    
    except Exception as e:
        logger.error(f"Error during description generation: {e}")
        return f"Error generating description: {str(e)}"

# Test the function locally if needed
if __name__ == "__main__":
    try:
        desc = generate_description("Super Phone", "Electronics", "TechCorp", "fast, 5g, smart")
        print("Generated Description:", desc)
    except Exception as e:
        logger.error(f"Test failed: {e}")