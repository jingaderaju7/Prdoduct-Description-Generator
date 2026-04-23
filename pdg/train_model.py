import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
import re
import os

# Download NLTK data (stopwords) if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# 1. Load the Dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# 2. Text Cleaning Function
def clean_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation,
    and stripping whitespace.
    """
    if not text:
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = text.strip()
    return text

# Apply cleaning to relevant columns
# We combine features into a single string for vectorization
print("Cleaning text data...")
df['clean_name'] = df['product_name'].apply(clean_text)
df['clean_category'] = df['category'].apply(clean_text)
df['clean_brand'] = df['brand'].apply(clean_text)
df['clean_features'] = df['key_features'].apply(clean_text)

# Create a 'soup' of text features (combined text)
# We give category slightly more weight by repeating it (a simple heuristic)
df['text_soup'] = (df['clean_name'] + " " + 
                   df['clean_category'] + " " + 
                   df['clean_brand'] + " " + 
                   df['clean_features'] + " " + 
                   df['clean_category'])

# 3. TF-IDF Vectorization
# Convert text to numerical feature vectors
print("Vectorizing text (TF-IDF)...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text_soup'])

# 4. Train Model (Nearest Neighbors)
# We use this model to find the most similar product in our dataset
print("Training Nearest Neighbors model...")
model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(tfidf_matrix)

# 5. Save the artifacts
# Create a directory for models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model, vectorizer, and the processed dataframe
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('models/nearest_neighbors_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the dataframe for looking up descriptions later
df.to_pickle('models/processed_data.pkl')

print("Training complete. Models saved to /models folder.")