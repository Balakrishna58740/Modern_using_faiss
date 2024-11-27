import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from model import MovieRecommender

class MovieRecommendationSystem:
    client = chromadb.PersistentClient(path="./databasedb")
    collection = None
    # Initialize model once as a class variable
    model = None
    metadatas = []  # To store metadata for querying

    @staticmethod
    def initialize_model():
        if MovieRecommendationSystem.model is None:
            MovieRecommendationSystem.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        return MovieRecommendationSystem.model

    @staticmethod
    def get_collection():
        if MovieRecommendationSystem.collection is None:
            try:
                # Try to get existing collection
                MovieRecommendationSystem.collection = MovieRecommendationSystem.client.get_collection(name="movie_recommendations")
            except:
                # Create new collection if it doesn't exist
                MovieRecommendationSystem.collection = MovieRecommendationSystem.client.create_collection(name="movie_recommendations")
        return MovieRecommendationSystem.collection
    @staticmethod
    async def add_movies():
        print("Job started")
        movies = await MovieRecommender.fetch_movie_data()
        documents = []
        ids = []
        embeddings = []
        MovieRecommendationSystem.metadatas.clear()
        
        collection = MovieRecommendationSystem.get_collection()
        model = MovieRecommendationSystem.initialize_model()
        
        for movie in movies:
            genres = movie.get('genres', '').strip('; ').split(';')  # Split and remove trailing semicolon/spaces
            genres = [g.strip() for g in genres if g.strip()]  # Clean each genre
            description = f"A {', '.join(genres)} movie rated {movie.get('rating', 'N/A')}."
            vector = model.encode(description).astype(float).tolist()  # Convert embedding to list
            
            documents.append(description)
            embeddings.append(vector)
            MovieRecommendationSystem.metadatas.append({
                "name": movie.get("name"), 
                "genres": movie.get("genres", []), 
                "rating": movie.get("rating")
            })
            ids.append(movie["_id"])  # Use MongoDB ID as the document ID
            
        # Add documents with embeddings to the Chroma collection
        collection.add(
            documents=documents,
            embeddings=embeddings, 
            metadatas=MovieRecommendationSystem.metadatas,
            ids=ids
        )
        return {"message": "Movies added successfully with embeddings."}
    @staticmethod
    async def find_similar_movies(movie_title: str, top_n: int = 5):
        collection = MovieRecommendationSystem.get_collection()
        model = MovieRecommendationSystem.initialize_model()
        
        # Get embedding for query movie title
        query_embedding = model.encode(movie_title).astype(float).tolist()
        
        # Query the collection with embedding, requesting extra results since we'll filter out the query movie
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n + 1  # Request one extra to account for potential self-match
        )
        
        try:
            if results and results['metadatas']:
                # Filter out the query movie if it appears in results
                filtered_results = {
                    'ids': [],
                    'distances': [],
                    'metadatas': [],
                    'documents': []
                }
                
                for i in range(len(results['ids'][0])):
                    if results['metadatas'][0][i]['name'].lower() != movie_title.lower():
                        filtered_results['ids'].append(results['ids'][0][i])
                        filtered_results['distances'].append(results['distances'][0][i])
                        filtered_results['metadatas'].append(results['metadatas'][0][i])
                        filtered_results['documents'].append(results['documents'][0][i])
                        
                        if len(filtered_results['ids']) == top_n:
                            break
                
                return filtered_results
            return results
        except Exception as e:
            print(f"Error querying similar movies: {e}")
            return []
