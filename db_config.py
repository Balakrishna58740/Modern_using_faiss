import motor.motor_asyncio
from contextlib import asynccontextmanager

class MongoDB:
    def __init__(self):
        self.client = None
        self.database = None
        self.collections = {}

    async def connect(self, uri: str, db_name: str):
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
            self.database = self.client[db_name]
            collection_names = ['Movies', 'TrainLogs']  # Specify collections you're working with
            self.collections = {name: self.database[name] for name in collection_names}
            print("MongoDB connected")
            print("Collections loaded:", self.collections.keys())  # Debug print to check loaded collections
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            
    async def close(self):
        self.client.close()
        print("MongoDB connection closed")

mongodb = MongoDB()
