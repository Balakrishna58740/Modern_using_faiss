import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from model import MovieRecommender

class MovieRecommendationSystem:
    client = chromadb.Client(Settings(persist_directory="/Users/pukarchalise/Desktop/Projects/Machine_learning_Project/Modern_using_faiss/databasedb2"))
    collection = client.create_collection(name="movie_recommendations")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    metadatas = []  # To store metadata for querying
    async def add_movies():
        movies = await MovieRecommender.fetch_movie_data()
        documents = []
        ids = []
        MovieRecommendationSystem.metadatas.clear()
        for movie in movies:
            description = f"A {', '.join(movie.get('genres', []))} movie rated {movie.get('rating', 'N/A')}."
            vector = MovieRecommendationSystem.model.encode(description).astype(float).tolist()  # Convert embedding to list
            
            documents.append(description)
            MovieRecommendationSystem.metadatas.append({"name": movie.get("name"), "genres": movie.get("genres", []), "rating": movie.get("rating")})
            ids.append(movie["_id"])  # Use MongoDB ID as the document ID

        # Add documents to the Chroma collection
        MovieRecommendationSystem.collection.add(documents=documents, metadatas=MovieRecommendationSystem.metadatas, ids=ids)
        return {"message": "Movies added successfully."}

    @staticmethod
    async def find_similar_movies(movie_title: str, top_n: int = 5):
        # Fetch the target movie metadata
        target_movie = next((movie for movie in MovieRecommendationSystem.metadatas if movie["name"] == movie_title), None)
        if not target_movie:
            raise ValueError(f"Movie '{movie_title}' not found.")
        
        # Generate the embedding for the target movie's description
        description = f"A {', '.join(target_movie['genres'])} movie rated {target_movie['rating']}."
        target_vector = MovieRecommendationSystem.model.encode(description).astype(float).tolist()

        # Querying similar movies
        results = MovieRecommendationSystem.collection.query(query_embeddings=[target_vector], n_results=top_n)
        try:
            if results:
                return results
        except Exception as e:
            print(f"Error querying similar movies: {e}")
            return []

    
        
