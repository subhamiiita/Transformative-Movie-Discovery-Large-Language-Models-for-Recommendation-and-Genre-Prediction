import openai
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
import os
from collections import defaultdict

class MovieLensRankingPredictor:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the MovieLensRankingPredictor with OpenAI API key
        
        Args:
            api_key (str): OpenAI API key
            model (str): LLM model to use
        """
        openai.api_key = api_key
        self.model = model
    
    def load_movielens_ratings(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens-100K ratings data
        
        Args:
            data_path (str): Path to MovieLens-100K data directory
            
        Returns:
            pd.DataFrame: Ratings dataframe
        """
        ratings_df = pd.read_csv(
            os.path.join(data_path, 'u.data'),
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        return ratings_df
    
    def create_user_profiles(self, ratings_df: pd.DataFrame, movies_with_summaries: Dict,
                           min_ratings: int = 20) -> List[Dict]:
        """
        Create user profiles for ranking prediction
        
        Args:
            ratings_df (pd.DataFrame): User ratings data
            movies_with_summaries (Dict): Movies with generated summaries
            min_ratings (int): Minimum number of ratings required per user
            
        Returns:
            List[Dict]: User profiles with movie history
        """
        users_profiles = []
        
        # Group by user and filter users with sufficient ratings
        user_groups = ratings_df.groupby('user_id')
        
        for user_id, user_ratings in user_groups:
            if len(user_ratings) >= min_ratings:
                # Sort by rating (descending) and timestamp (descending for tie-breaking)
                user_ratings_sorted = user_ratings.sort_values(['rating', 'timestamp'], 
                                                              ascending=[False, False])
                
                # Get top 15 movies for user history
                top_movies = user_ratings_sorted.head(15)
                user_history = []
                
                for _, rating_row in top_movies.iterrows():
                    movie_id = str(rating_row['movie_id'])
                    if movie_id in movies_with_summaries:
                        movie_info = movies_with_summaries[movie_id]
                        user_history.append({
                            'movie_id': rating_row['movie_id'],
                            'title': movie_info['title'],
                            'summary': movie_info.get('generated_summary', ''),
                            'rating': rating_row['rating'],
                            'genres': movie_info.get('genres', [])
                        })
                
                # Get candidate movies (next 5 movies with rating >= 3)
                candidate_movies = user_ratings_sorted.iloc[15:].query('rating >= 3').head(5)
                candidates = []
                
                for _, rating_row in candidate_movies.iterrows():
                    movie_id = str(rating_row['movie_id'])
                    if movie_id in movies_with_summaries:
                        movie_info = movies_with_summaries[movie_id]
                        candidates.append({
                            'movie_id': rating_row['movie_id'],
                            'title': movie_info['title'],
                            'summary': movie_info.get('generated_summary', ''),
                            'actual_rating': rating_row['rating'],
                            'genres': movie_info.get('genres', [])
                        })
                
                if len(user_history) >= 10 and len(candidates) >= 3:
                    # Create actual ranking based on ratings
                    actual_ranking = [movie['title'] for movie in 
                                    sorted(candidates, key=lambda x: x['actual_rating'], reverse=True)]
                    
                    users_profiles.append({
                        'user_id': user_id,
                        'user_history': user_history,
                        'candidate_movies': candidates,
                        'actual_ranking': actual_ranking
                    })
        
        return users_profiles
    
    def create_user_profile_text(self, user_history: List[Dict]) -> str:
        """
        Create a formatted user profile string from user history
        
        Args:
            user_history (List[Dict]): User's movie history with ratings
            
        Returns:
            str: Formatted user profile
        """
        profile_text = ""
        for i, movie in enumerate(user_history, 1):
            profile_text += f"{i}. {movie['title']} (Rating: {movie['rating']}/5)\n"
            if movie.get('summary'):
                profile_text += f"   Summary: {movie['summary']}\n"
            if movie.get('genres'):
                profile_text += f"   Genres: {', '.join(movie['genres'])}\n"
            profile_text += "\n"
        
        return profile_text
    
    def predict_ranking(self, user_profile: Dict) -> List[str]:
        """
        Predict ranking of candidate movies based on user history
        
        Args:
            user_profile (Dict): User profile with history and candidates
            
        Returns:
            List[str]: Ranked movie titles from most to least liked
        """
        # Create user profile text
        user_profile_text = self.create_user_profile_text(user_profile['user_history'])
        
        # Create candidate movies list
        candidate_list = []
        for movie in user_profile['candidate_movies']:
            candidate_list.append(f"- {movie['title']}")
            if movie.get('summary'):
                candidate_list.append(f"  Summary: {movie['summary']}")
            if movie.get('genres'):
                candidate_list.append(f"  Genres: {', '.join(movie['genres'])}")
        
        candidate_text = "\n".join(candidate_list)
        
        prompt = f"""The movies watched by a user are given below, ordered from most liked to least liked based on their ratings.

User's Movie History (most liked to least liked):
{user_profile_text}

Based on this user's preferences, rank the following {len(user_profile['candidate_movies'])} movies from most likely to be liked to least likely to be liked by this user. Provide only the movie titles in a Python list format:

{candidate_text}

<output: predicted ranking>"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract ranking from response
            try:
                if '[' in response_text and ']' in response_text:
                    start = response_text.find('[')
                    end = response_text.find(']') + 1
                    list_str = response_text[start:end]
                    ranking = eval(list_str)
                    
                    # Validate that all movies are in the ranking
                    candidate_titles = [movie['title'] for movie in user_profile['candidate_movies']]
                    valid_ranking = [title for title in ranking if title in candidate_titles]
                    
                    # Add missing movies at the end
                    for title in candidate_titles:
                        if title not in valid_ranking:
                            valid_ranking.append(title)
                    
                    return valid_ranking[:len(user_profile['candidate_movies'])]
                else:
                    # Fallback: return original order
                    return [movie['title'] for movie in user_profile['candidate_movies']]
                    
            except Exception as parse_error:
                print(f"Parsing error for user {user_profile['user_id']}: {parse_error}")
                return [movie['title'] for movie in user_profile['candidate_movies']]
                
        except Exception as e:
            print(f"Error predicting ranking for user {user_profile['user_id']}: {e}")
            return [movie['title'] for movie in user_profile['candidate_movies']]
    
    def process_user_recommendations(self, user_profiles: List[Dict], 
                                   max_users: int = 50) -> List[Dict]:
        """
        Process recommendations for multiple users
        
        Args:
            user_profiles (List[Dict]): User profiles
            max_users (int): Maximum number of users to process
            
        Returns:
            List[Dict]: Recommendation results for each user
        """
        results = []
        
        for i, user_profile in enumerate(user_profiles[:max_users]):
            print(f"Processing user {i+1}/{min(max_users, len(user_profiles))}: {user_profile['user_id']}")
            
            # Predict ranking
            predicted_ranking = self.predict_ranking(user_profile)
            
            user_result = {
                'user_id': user_profile['user_id'],
                'user_history_titles': [movie['title'] for movie in user_profile['user_history']],
                'candidate_movies': [movie['title'] for movie in user_profile['candidate_movies']],
                'predicted_ranking': predicted_ranking,
                'actual_ranking': user_profile['actual_ranking']
            }
            
            results.append(user_result)
            
            # Add delay to respect API rate limits
            time.sleep(2)
        
        return results
    
    def calculate_ranking_metrics(self, results: List[Dict], 
                                k_values: List[int] = [3, 5]) -> Dict:
        """
        Calculate ranking metrics (MAP, MRR, NDCG)
        
        Args:
            results (List[Dict]): Recommendation results
            k_values (List[int]): K values for evaluation
            
        Returns:
            Dict: Ranking metrics
        """
        metrics = {}
        
        for k in k_values:
            map_scores = []
            mrr_scores = []
            ndcg_scores = []
            
            for result in results:
                predicted = result['predicted_ranking'][:k]
                actual = result['actual_ranking']
                
                # Calculate MAP@k
                ap_score = self._calculate_ap(predicted, actual, k)
                map_scores.append(ap_score)
                
                # Calculate MRR@k
                rr_score = self._calculate_rr(predicted, actual, k)
                mrr_scores.append(rr_score)
                
                # Calculate NDCG@k
                ndcg_score = self._calculate_ndcg(predicted, actual, k)
                ndcg_scores.append(ndcg_score)
            
            metrics[f'MAP@{k}'] = np.mean(map_scores)
            metrics[f'MRR@{k}'] = np.mean(mrr_scores)
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores)
        
        return metrics
    
    def _calculate_ap(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Calculate Average Precision at k"""
        if not actual:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_items = set(actual)
        
        if not relevant_items:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(predicted_k):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / min(len(relevant_items), k)
    
    def _calculate_rr(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Calculate Reciprocal Rank at k"""
        if not actual:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_items = set(actual)
        
        for i, item in enumerate(predicted_k):
            if item in relevant_items:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg(self, predicted: List[str], actual: List[str], k: int) -> float:
        """Calculate NDCG at k"""
        if not actual:
            return 0.0
        
        predicted_k = predicted[:k]
        
        # Create relevance scores based on position in actual ranking
        relevance_scores = []
        for item in predicted_k:
            if item in actual:
                # Higher relevance for higher position in actual ranking
                pos = actual.index(item)
                relevance_scores.append(len(actual) - pos)
            else:
                relevance_scores.append(0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)
        
        # Calculate IDCG
        ideal_relevance = sorted([len(actual) - i for i in range(len(actual))], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance[:k]):
            if rel > 0:
                idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save_results(self, results: List[Dict], filename: str):
        """
        Save results to JSON file
        
        Args:
            results (List[Dict]): Results to save
            filename (str): Output filename
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = MovieLensRankingPredictor("your-openai-api-key")
    
    # Load ratings data
    ratings_df = predictor.load_movielens_ratings("path/to/movielens-100k/")
    
    # Load movies with summaries
    with open("movielens_genre_predictions.json", 'r') as f:
        movies_data = json.load(f)
    
    # Convert to dictionary for easy lookup
    movies_with_summaries = {str(movie['movie_id']): movie for movie in movies_data}
    
    # Create user profiles
    user_profiles = predictor.create_user_profiles(ratings_df, movies_with_summaries)
    print(f"Created {len(user_profiles)} user profiles")
    
    # Process recommendations
    results = predictor.process_user_recommendations(user_profiles, max_users=20)
    predictor.save_results(results, "movielens_ranking_predictions.json")
    
    # Calculate metrics
    metrics = predictor.calculate_ranking_metrics(results)
    print("\nRanking Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
