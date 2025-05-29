#!/usr/bin/env python3

import json
import os
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_recommendations(similarity_matrix, test_df, top_k=5):
    """
    Evaluate the recommendation system using various metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['total_movies'] = len(test_df)
    metrics['similarity_matrix_shape'] = similarity_matrix.shape
    metrics['average_similarity'] = float(np.mean(similarity_matrix))
    metrics['similarity_std'] = float(np.std(similarity_matrix))
    
    # Coverage metrics
    non_zero_similarities = np.count_nonzero(similarity_matrix)
    total_possible_pairs = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    metrics['coverage'] = float(non_zero_similarities / total_possible_pairs)
    
    # Diversity metrics (average pairwise distance of top recommendations)
    diversity_scores = []
    for i in range(min(100, len(test_df))):  # Sample 100 movies for efficiency
        # Get top k similar movies
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[-top_k-1:-1]  # Exclude self
        
        if len(top_indices) > 1:
            # Calculate average pairwise distance in top recommendations
            top_similarities = similarities[top_indices]
            avg_diversity = 1 - np.mean(top_similarities)
            diversity_scores.append(avg_diversity)
    
    if diversity_scores:
        metrics['average_diversity'] = float(np.mean(diversity_scores))
        metrics['diversity_std'] = float(np.std(diversity_scores))
    else:
        metrics['average_diversity'] = 0.0
        metrics['diversity_std'] = 0.0
    
    # Recommendation quality score (composite metric)
    # Higher similarity variance suggests better discrimination
    similarity_variance = np.var(similarity_matrix, axis=1)
    metrics['average_discrimination'] = float(np.mean(similarity_variance))
    
    # Calculate a composite quality score
    # Good recommendation system should have:
    # - High coverage
    # - Good diversity
    # - Good discrimination ability
    quality_score = (
        metrics['coverage'] * 0.3 + 
        metrics['average_diversity'] * 0.4 + 
        metrics['average_discrimination'] * 0.3
    )
    metrics['quality_score'] = float(quality_score)
    
    return metrics

if __name__ == "__main__":
    logger.info("Starting model evaluation")
    
    # Load model artifacts
    model_path = "/opt/ml/processing/model"
    with open(os.path.join(model_path, "similarity.pkl"), "rb") as f:
        similarity_matrix = pickle.load(f)
    
    with open(os.path.join(model_path, "movie_list.pkl"), "rb") as f:
        movie_list = pickle.load(f)
    
    logger.info(f"Loaded similarity matrix of shape: {similarity_matrix.shape}")
    logger.info(f"Loaded movie list with {len(movie_list)} movies")
    
    # Load test data
    test_data_path = "/opt/ml/processing/test/test.csv"
    test_df = pd.read_csv(test_data_path)
    logger.info(f"Loaded test data with {len(test_df)} movies")
    
    # Evaluate the model
    evaluation_metrics = evaluate_recommendations(similarity_matrix, test_df)
    
    # Log key metrics
    logger.info("Evaluation Results:")
    logger.info(f"Quality Score: {evaluation_metrics['quality_score']:.4f}")
    logger.info(f"Coverage: {evaluation_metrics['coverage']:.4f}")
    logger.info(f"Average Diversity: {evaluation_metrics['average_diversity']:.4f}")
    logger.info(f"Average Discrimination: {evaluation_metrics['average_discrimination']:.4f}")
    
    # Prepare evaluation report
    report_dict = {
        "recommendation_metrics": evaluation_metrics,
        "model_quality": {
            "quality_score": evaluation_metrics['quality_score'],
            "status": "PASS" if evaluation_metrics['quality_score'] > 0.3 else "FAIL"
        }
    }
    
    # Save evaluation report
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Evaluation report saved to {evaluation_path}")
    logger.info("Model evaluation completed successfully!")
