#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import ast
import logging
from pathlib import Path
from nltk.stem.porter import PorterStemmer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert(text):
    """Convert JSON string to list of names"""
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def convert_cast(text):
    """Extract top 3 cast members"""
    try:
        L = []
        for i, val in enumerate(ast.literal_eval(text)):
            if i < 3:
                L.append(val['name'])
            else:
                break
        return L
    except:
        return []

def fetch_director(text):
    """Extract director from crew"""
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

def remove_space(L):
    """Remove spaces from list items"""
    return [i.replace(" ", "") for i in L if isinstance(i, str)]

def stems(text):
    """Apply stemming to text"""
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies-data", type=str, help="Path to movies CSV file")
    parser.add_argument("--credits-data", type=str, help="Path to credits CSV file")
    
    args = parser.parse_args()
    
    logger.info("Starting movie data preprocessing")
    
    # Load input data
    logger.info(f"Loading movies data from: {args.movies_data}")
    logger.info(f"Loading credits data from: {args.credits_data}")
    
    movies = pd.read_csv(args.movies_data)
    credits = pd.read_csv(args.credits_data)
    
    logger.info(f"Movies dataset shape: {movies.shape}")
    logger.info(f"Credits dataset shape: {credits.shape}")
    
    # Merge datasets
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Handle missing values
    movies.dropna(inplace=True)
    logger.info(f"Dataset shape after removing NaN: {movies.shape}")
    
    # Process features
    logger.info("Processing movie features...")
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    # Remove spaces
    movies['cast'] = movies['cast'].apply(remove_space)
    movies['crew'] = movies['crew'].apply(remove_space)
    movies['genres'] = movies['genres'].apply(remove_space)
    movies['keywords'] = movies['keywords'].apply(remove_space)
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create final dataframe
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    # Apply stemming
    logger.info("Applying stemming to text data...")
    new_df['tags'] = new_df['tags'].apply(stems)
    
    # Split data (for this use case, we'll use all data for training)
    # But we'll create train/validation/test splits as per MLOps best practices
    
    # Shuffle the data
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split ratios
    n = len(new_df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = new_df[:train_end]
    val_df = new_df[train_end:val_end]
    test_df = new_df[val_end:]
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Save processed data
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/validation", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    
    train_df.to_csv("/opt/ml/processing/train/train.csv", index=False)
    val_df.to_csv("/opt/ml/processing/validation/validation.csv", index=False)
    test_df.to_csv("/opt/ml/processing/test/test.csv", index=False)
    
    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Files saved to /opt/ml/processing/[train|validation|test]/")
