import asyncio
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from db_config import mongodb  # Assumes a configured MongoDB instance



class MovieRecommender:
    VECTOR_SIZE = 384  # Embedding vector size from SentenceTransformer
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Pre-trained model
    index = None  # FAISS index for L2 distance
    movie_map = {}  # Map to store movie ID and metadata for lookup
    INDEX_FILE = "faiss_index1.bin"
    MAP_FILE = "movie_map1.pkl"
    # @staticmethod
    # def initialize_index():
    #     """
    #     Initializes the FAISS index.
    #     """
    #     if MovieRecommender.index is None:
    #         MovieRecommender.index = faiss.IndexFlatL2(MovieRecommender.VECTOR_SIZE)
    #         print(f"Initialized a new FAISS index with vector size {MovieRecommender.VECTOR_SIZE}.")
    #     else:
    #         print("FAISS index already initialized.")


    @staticmethod
    async def fetch_movie_data():
        """
        Fetch movie data from MongoDB.
        If `train_only=True`, fetches only movies with `Train=False`.
        """
        try:
            movies = await mongodb.collections["Movies"].find().to_list(length=None)
            for movie in movies:
                movie["_id"] = str(movie["_id"])  # Convert MongoDB ObjectId to string
            return movies
        except Exception as e:
            raise Exception(f"Error fetching movie data: {e}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text for embedding generation.
        """
        return text.lower().strip() if text else ""

    @staticmethod
    async def store_movie_vectors():
        """
        Generates embeddings for movie descriptions and stores them in FAISS.
        Updates the FAISS index and movie map if `train_only=True`.
        """
        try:
            MovieRecommender.initialize_index()
            movie_data = await MovieRecommender.fetch_movie_data()
            vectors = []
            new_ids = []

            for movie in movie_data:
                # Construct a synthetic description using genres and rating
                genres = " and ".join(movie.get("genres", []))
                rating = movie.get("rating", "N/A")
                synthetic_description = f"A {genres} movie rated {rating} out of 10."

                # Preprocess the synthetic description
                description = MovieRecommender.preprocess_text(
                    movie.get("description", synthetic_description)
                )

                # Generate the embedding
                vector = MovieRecommender.MODEL.encode(description).astype(np.float32)
                vectors.append(vector)

                # Add movie details to the lookup map
                MovieRecommender.movie_map[movie["_id"]] = {
                    "name": movie.get("name"),
                    "genres": movie.get("genres", []),
                    "movie_rated": movie.get("rating", ""),
                }
                new_ids.append(movie["_id"])
            # Add vectors to the FAISS index
            if vectors:
                MovieRecommender.index.add(np.array(vectors))
                # Update the database to mark movies as trained
                await mongodb.collections["Movies"].update_many(
                    {"_id": {"$in": new_ids}}, {"$set": {"Train": True}}
                )
                await mongodb.collections["TrainLogs"].insert_many([{"timestamp": datetime.now(), "records_count": len(vectors), "Status": "Success"}])
            return f"Stored {len(vectors)} movie vectors in FAISS."
        except Exception as e:
            raise Exception(f"Error storing movie vectors: {e}")

    # @staticmethod
    # def save_model():
    #     """
    #     Save the FAISS index and movie map to disk.
    #     """
    #     try:
    #         faiss.write_index(MovieRecommender.index, MovieRecommender.INDEX_FILE)
    #         with open(MovieRecommender.MAP_FILE, "wb") as f:
    #             pickle.dump(MovieRecommender.movie_map, f)
    #         print("Model saved successfully.")
    #     except Exception as e:
    #         raise Exception(f"Error saving model: {e}")

    # @staticmethod
    # def load_model():
    #     """
    #     Load the FAISS index and movie map from disk.
    #     """
    #     try:
    #         MovieRecommender.index = faiss.read_index(MovieRecommender.INDEX_FILE)
    #         with open(MovieRecommender.MAP_FILE, "rb") as f:
    #             MovieRecommender.movie_map = pickle.load(f)
    #         print("Model loaded successfully.")
    #     except Exception as e:
    #         raise Exception(f"Error loading model: {e}")
    # @staticmethod
    # def find_similar_movies(movie_title: str, top_n: int = 5):
    #     """
    #     Finds similar movies based on a given movie title.
    #     """
    #     try:
    #         if MovieRecommender.index is None or MovieRecommender.index.ntotal == 0:
    #             raise Exception("FAISS index is empty. Please load or train the model first.")

    #         # Find the target movie by title
    #         target_movie = next(
    #             (movie for movie_id, movie in MovieRecommender.movie_map.items() if movie["name"] == movie_title),
    #             None,
    #         )
    #         if not target_movie:
    #             raise ValueError(f"Movie '{movie_title}' not found.")

    #         # Generate the embedding for the target movie's description
    #         genres = " and ".join(target_movie.get("genres", []))
    #         rating = target_movie.get("movie_rated", "N/A")
    #         synthetic_description = f"A {genres} movie rated {rating}."
    #         description = MovieRecommender.preprocess_text(target_movie.get("description", synthetic_description))
    #         target_vector = MovieRecommender.MODEL.encode(description).astype(np.float32).reshape(1, -1)

    #         # Search for similar vectors
    #         distances, indices = MovieRecommender.index.search(target_vector, top_n + 1)
    #         # Validate the indices
    #         similar_movies = []
    #         for idx in indices[0]:
    #             if idx != -1 and idx < len(MovieRecommender.movie_map):  # Check bounds
    #                 movie_id = list(MovieRecommender.movie_map.keys())[idx]
    #                 movie = MovieRecommender.movie_map.get(movie_id)
    #                 if movie and movie["name"] != movie_title:  # Exclude the input movie itself
    #                     similar_movies.append(movie["name"])
    #         return similar_movies[:top_n]
    #     except ValueError as ve:
    #         raise ve  # Propagate specific errors
    #     except Exception as e:
    #         raise Exception(f"Error finding similar movies: {e}")
        




        