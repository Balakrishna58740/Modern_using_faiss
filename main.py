import asyncio
from http.client import HTTPException
import os
from sched import scheduler
from newdb import MovieRecommendationSystem
from model import MovieRecommender
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import numpy as np
from sentence_transformers import SentenceTransformer
from db_config import mongodb  # Assumes a configured MongoDB instance
from fastapi import FastAPI

scheduler = AsyncIOScheduler()
app = FastAPI()
@app.on_event("startup")
async def startup():
    try:
        # Connect to MongoDB
        await mongodb.connect(
            uri="mongodb+srv://pukar:pukarpass@cluster0.2xqtwbl.mongodb.net/DataScience",
            db_name="DataScience",
        )
        print("MongoDB connected successfully.")
        # scheduler.start()
        # scheduler.add_job(MovieRecommendationSystem.add_movies,trigger=CronTrigger(hour="*", minute="*"),id='add_movies_job')
        # scheduler.start()
        # scheduler.add_job(fetch_and_train, "cron", hour=0, minute=0)
        # MovieRecommender.load_model()
        # print("Scheduled training job added.")
    except Exception as e:
        print(f"Error during startup: {str(e)}")


@app.on_event("shutdown")
async def shutdown():
    try:
        # Shutdown MongoDB connection
        await mongodb.close()
        print("MongoDB connection closed.")
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")

# @app.get("/recommendationsFAISS/")
# async def get_movie_recommendations(movie_title: str, top_n: int = 5):
#     try:
#         recommendations = MovieRecommender.find_similar_movies(movie_title, top_n)
#         return {"recommended_movies": recommendations}
#     except Exception as e:
#         raise HTTPException(500, str(e))  # Correct way to raise HTTPException

# async def fetch_and_train():
#     try:
#         await MovieRecommender.store_movie_vectors()
#         MovieRecommender.save_model()
#         print("Model updated and saved.")
#     except Exception as e:
#         print(f"Error during scheduled training: {e}")


@app.get("/recommendations/{movie_title}/{top_n}")
async def get_recommendations(movie_title: str, top_n: int = 5):
    try:
        await MovieRecommendationSystem.add_movies()
        recommendations = await MovieRecommendationSystem.find_similar_movies(movie_title, top_n)
        return recommendations
    except ValueError as e:
        raise HTTPException(404, str(e))

    # try:
    #     recommendations = await MovieRecommendationSystem.find_similar_movies(movie_title, top_n)
    #     return {"recommended_movies": recommendations}
    # except ValueError as e:
    #     raise HTTPException(status_code=404, detail=str(e))
    