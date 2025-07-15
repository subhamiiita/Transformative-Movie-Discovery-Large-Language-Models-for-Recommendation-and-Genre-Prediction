import openai
import json
import pandas as pd
from typing import List, Dict
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

class MovieLensGenrePredictor:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the MovieLensGenrePredictor with OpenAI API key
        
        Args:
            api_key (str): OpenAI API key
            model (str): LLM model to use
        """
        openai.api_key = api_key
        self.model = model
        # MovieLens-100K genres (excluding 'unknown')
        self.genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", 
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
            "Thriller", "War", "Western"
        ]
    
    def predict_genre(self, movie_summary: str, movie_title: str) -> List[str]:
        """
        Predict movie genres based on generated summary
        
        Args:
            movie_summary (str): Generated movie summary
            movie_title (str): Movie title for context
            
        Returns:
            List[str]: Predicted genres
        """
        prompt = f"""Given the set of genres in square brackets:
[Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western].

A movie summary for "{movie_title}" is provided as:
{movie_summary}

Please provide the genres of this movie from the set provided, based on the summary given, in Python list format. Only include the genres in the Python list, without any additional text.

<output: predicted genres>"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response to extract genres
            try:
                if '[' in response_text and ']' in response_text:
                    start = response_text.find('[')
                    end = response_text.find(']') + 1
                    list_str = response_text[start:end]
                    # Clean up the string
                    list_str = list_str.replace("'", '"')
                    predicted_genres = eval(list_str)
                    
                    # Validate genres
                    valid_genres = []
                    for genre in predicted_genres:
                        if genre in self.genres:
                            valid_genres.append(genre)
                        elif genre == "Children" and "Children's" in self.genres:
                            valid_genres.append("Children's")
                    
                    return valid_genres
                else:
                    # Fallback: extract genres mentioned in text
                    mentioned_genres = []
                    for genre in self.genres:
                        if genre.lower() in response_text.lower():
                            mentioned_genres.append(genre)
                    return mentioned_genres
                    
            except Exception as parse_error:
                print(f"Parsing error for {movie_title}: {parse_error}")
                # Fallback parsing
                mentioned_genres = []
                for genre in self.genres:
                    if genre.lower() in response_text.lower():
                        mentioned_genres.append(genre)
                return mentioned_genres
                
        except Exception as e:
            print(f"Error predicting genres for {movie_title}: {e}")
            return []
    
    def process_movielens_summaries(self, summaries_data: List[Dict]) -> List[Dict]:
        """
        Process MovieLens movies with generated summaries to predict genres
        
        Args:
            summaries_data (List[Dict]): Movies with generated summaries
            
        Returns:
            List[Dict]: Movies with predicted genres
        """
        results = []
        
        for movie in summaries_data:
            print(f"Predicting genres for: {movie['title']}")
            
            if 'generated_summary' in movie and movie['generated_summary']:
                predicted_genres = self.predict_genre(
                    movie['generated_summary'], 
                    movie['title']
                )
            else:
                predicted_genres = []
            
            movie_result = movie.copy()
            movie_result['predicted_genres'] = predicted_genres
            results.append(movie_result)
            
            # Add delay to respect API rate limits
            time.sleep(1)
        
        return results
    
    def evaluate_predictions(self, results: List[Dict]) -> Dict:
        """
        Evaluate genre predictions against MovieLens ground truth
        
        Args:
            results (List[Dict]): Results with predicted and actual genres
            
        Returns:
            Dict: Evaluation metrics
        """
        # Prepare data for evaluation
        y_true = []
        y_pred = []
        
        for movie in results:
            actual_genres = movie.get('genres', [])
            predicted_genres = movie.get('predicted_genres', [])
            
            # Convert to binary vectors
            true_vector = [1 if genre in actual_genres else 0 for genre in self.genres]
            pred_vector = [1 if genre in predicted_genres else 0 for genre in self.genres]
            
            y_true.append(true_vector)
            y_pred.append(pred_vector)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics per genre
        metrics = {}
        
        for i, genre in enumerate(self.genres):
            # Calculate precision, recall, F1 for each genre
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            
            metrics[genre] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        
        # Calculate overall metrics
        overall_precision = np.mean([metrics[genre]['precision'] for genre in self.genres])
        overall_recall = np.mean([metrics[genre]['recall'] for genre in self.genres])
        overall_f1 = np.mean([metrics[genre]['f1'] for genre in self.genres])
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }
        
        return metrics
    
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
    predictor = MovieLensGenrePredictor("your-openai-api-key")
    
    # Load movies with summaries
    with open("movielens_zero_shot_summaries.json", 'r') as f:
        summaries_data = json.load(f)
    
    # Predict genres
    results = predictor.process_movielens_summaries(summaries_data)
    predictor.save_results(results, "movielens_genre_predictions.json")
    
    # Evaluate predictions
    metrics = predictor.evaluate_predictions(results)
    print("\nGenre Prediction Metrics:")
    for genre, metric in metrics.items():
        if genre != 'overall':
            print(f"{genre}: P={metric['precision']:.3f}, R={metric['recall']:.3f}, F1={metric['f1']:.3f}")
    
    print(f"\nOverall: P={metrics['overall']['precision']:.3f}, R={metrics['overall']['recall']:.3f}, F1={metrics['overall']['f1']:.3f}")
