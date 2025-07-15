import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import json
from typing import List, Dict, Optional
import time
import os
from rouge import Rouge

class MistralMovieSummaryGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", use_quantization: bool = True):
        """
        Initialize the MistralMovieSummaryGenerator with Hugging Face model
        
        Args:
            model_name (str): Mistral model name from Hugging Face
            use_quantization (bool): Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure quantization for memory efficiency
        if use_quantization:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Initialize ROUGE scorer
        self.rouge = Rouge()
        
    def load_movielens_data(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens-100K dataset
        
        Args:
            data_path (str): Path to MovieLens-100K data directory
            
        Returns:
            pd.DataFrame: Combined movie data with ratings
        """
        # Load movie information
        movies_df = pd.read_csv(
            os.path.join(data_path, 'u.item'),
            sep='|',
            encoding='latin-1',
            header=None,
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                  [f'genre_{i}' for i in range(19)]
        )
        
        # Load ratings
        ratings_df = pd.read_csv(
            os.path.join(data_path, 'u.data'),
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Merge movies with average ratings
        avg_ratings = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
        movies_with_ratings = movies_df.merge(avg_ratings, on='movie_id', how='left')
        
        return movies_with_ratings

    def simulate_trailer_transcript(self, movie_title: str, genres: List[str], 
                                  avg_rating: float, release_year: str) -> str:
        """
        Simulate a movie trailer transcript based on MovieLens data
        
        Args:
            movie_title (str): Movie title
            genres (List[str]): Movie genres
            avg_rating (float): Average rating
            release_year (str): Release year
            
        Returns:
            str: Simulated trailer transcript
        """
        genre_str = ", ".join(genres)
        
        transcript = f"""
        From the {release_year} {genre_str} film {movie_title}.
        
        [Background music plays]
        
        Experience the story that captivated audiences worldwide.
        
        {movie_title} - A {genre_str} adventure that will keep you on the edge of your seat.
        
        [Dramatic music intensifies]
        
        Don't miss this critically acclaimed film.
        
        {movie_title} - Coming to theaters.
        """.strip()
        
        return transcript

    def zero_shot_summary(self, audio_transcript: str, movie_title: str) -> str:
        """
        Generate movie summary using zero-shot prompting (exact prompt from paper)
        
        Args:
            audio_transcript (str): Audio transcript from movie trailer
            movie_title (str): Movie title for context
            
        Returns:
            str: Generated movie summary
        """
        # Exact prompt from the paper
        prompt = f"""System: you are an expert at providing a brief summary of the movie based on its audio transcript.
User: Audio transcript of movie is: {audio_transcript}
Assistant: Based on the audio transcript of movie, brief summary of the movie in paragraph format is

<<output: generated summary>>"""

        try:
            # Apply chat template for proper formatting
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    encoded,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][encoded.shape[1]:]
            summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary for {movie_title}: {e}")
            return f"A {movie_title} story."

    def few_shot_summary(self, audio_transcript: str, movie_title: str, 
                        example_transcript: str, example_summary: str) -> str:
        """
        Generate movie summary using few-shot prompting (exact prompt from paper)
        
        Args:
            audio_transcript (str): Audio transcript from movie trailer
            movie_title (str): Movie title for context
            example_transcript (str): Example transcript for few-shot learning
            example_summary (str): Example summary for few-shot learning
            
        Returns:
            str: Generated movie summary
        """
        # Exact prompt from the paper
        prompt = f"""System: you are an expert at providing a meaningful summary of the movie based on its audio transcript.one example of a movie trailer's audio transcript and it's IMDB summary is given below.Generate meaningful summary of audio transcript given after the example :
User : 
example((
Audio transcript:
{example_transcript}
IMDB Summary: {example_summary}))
Audio transcript of the movie is: {audio_transcript}
Generate meaningful summary of Audio transcript given after the above example of Audio transcript and its IMDB summary
Assistant: Based on the Audio transcript of the movie, meaningful summary of the movie in paragraph format is:

<<output: generated summary>>"""

        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    encoded,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][encoded.shape[1]:]
            summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary for {movie_title}: {e}")
            return f"A {movie_title} story."

    def calculate_rouge_scores(self, generated_summaries: List[str], 
                              reference_summaries: List[str]) -> Dict:
        """
        Calculate ROUGE scores for generated summaries
        
        Args:
            generated_summaries (List[str]): Generated summaries
            reference_summaries (List[str]): Reference summaries
            
        Returns:
            Dict: ROUGE scores
        """
        try:
            scores = self.rouge.get_scores(generated_summaries, reference_summaries, avg=True)
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {}

    def process_movielens_dataset(self, movies_df: pd.DataFrame, use_few_shot: bool = False,
                                example_transcript: str = "", example_summary: str = "",
                                max_movies: int = 100) -> List[Dict]:
        """
        Process MovieLens dataset to generate summaries
        
        Args:
            movies_df (pd.DataFrame): MovieLens movies dataframe
            use_few_shot (bool): Whether to use few-shot prompting
            example_transcript (str): Example transcript for few-shot learning
            example_summary (str): Example summary for few-shot learning
            max_movies (int): Maximum number of movies to process
            
        Returns:
            List[Dict]: Movie data with generated summaries
        """
        results = []
        
        # Get genre column names
        genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
        genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                      'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Process first max_movies movies
        for idx, row in movies_df.head(max_movies).iterrows():
            print(f"Processing movie {idx + 1}/{max_movies}: {row['title']}")
            
            # Extract genres
            movie_genres = []
            for i, genre_col in enumerate(genre_cols):
                if row[genre_col] == 1:
                    movie_genres.append(genre_names[i])
            
            # Extract release year
            release_year = row['release_date'].split('-')[-1] if pd.notna(row['release_date']) else "Unknown"
            
            # Generate simulated transcript
            transcript = self.simulate_trailer_transcript(
                row['title'], movie_genres, row['mean'], release_year
            )
            
            # Generate summary
            if use_few_shot:
                summary = self.few_shot_summary(
                    transcript, row['title'], example_transcript, example_summary
                )
            else:
                summary = self.zero_shot_summary(transcript, row['title'])
            
            movie_result = {
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'release_date': row['release_date'],
                'imdb_url': row['imdb_url'],
                'genres': movie_genres,
                'avg_rating': float(row['mean']) if pd.notna(row['mean']) else 0.0,
                'rating_count': int(row['count']) if pd.notna(row['count']) else 0,
                'simulated_transcript': transcript,
                'generated_summary': summary
            }
            
            results.append(movie_result)
            
            # Add delay for memory management
            time.sleep(0.5)
        
        return results

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
    # Initialize generator
    generator = MistralMovieSummaryGenerator(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        use_quantization=True
    )
    
    # Load MovieLens-100K data
    movies_df = generator.load_movielens_data("path/to/movielens-100k/")
    
    # Generate summaries using zero-shot
    zero_shot_results = generator.process_movielens_dataset(
        movies_df, use_few_shot=False, max_movies=50
    )
    generator.save_results(zero_shot_results, "mistral_zero_shot_summaries.json")
    
    # Generate summaries using few-shot
    example_transcript = """
    From the 1995 Animation, Children's, Comedy film Toy Story.
    [Background music plays]
    Experience the story that captivated audiences worldwide.
    Toy Story - A Animation, Children's, Comedy adventure that will keep you on the edge of your seat.
    [Dramatic music intensifies]
    Don't miss this critically acclaimed film.
    Toy Story - Coming to theaters.
    """
    
    example_summary = """
    A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.
    """
    
    few_shot_results = generator.process_movielens_dataset(
        movies_df, use_few_shot=True, example_transcript=example_transcript,
        example_summary=example_summary, max_movies=50
    )
    generator.save_results(few_shot_results, "mistral_few_shot_summaries.json")
