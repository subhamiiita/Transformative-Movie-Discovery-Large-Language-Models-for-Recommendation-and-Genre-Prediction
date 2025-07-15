# Transformative-Movie-Discovery-Large-Language-Models-for-Recommendation-and-Genre-Prediction
This repository implements a comprehensive movie discovery system that leverages 4 different Large Language Models (including GPT-3.5-turbo) for personalized movie recommendations. The system transforms movie trailer audio transcripts into movie summaries, predicts genres, and generates personalized movie rankings based on user preferences.

# Clone and install
git clone https://github.com/yourusername/movielens-movie-discovery.git
cd movielens-movie-discovery

# Download MovieLens-100K dataset
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/

# Add to each .py file or set as environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

