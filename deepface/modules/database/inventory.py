# built-in dependencies
from typing import TypedDict, Type, Dict

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.database.postgres import PostgresClient
from deepface.modules.database.pgvector import PGVectorClient
from deepface.modules.database.mongo import MongoDbClient as MongoClient
from deepface.modules.database.weaviate import WeaviateClient
from deepface.modules.database.neo4j import Neo4jClient
from deepface.modules.database.pinecone import PineconeClient


class DatabaseSpec(TypedDict):
    is_vector_db: bool
    connection_string: str
    client: Type["Database"]


database_inventory: Dict[str, DatabaseSpec] = {
    "postgres": {
        "is_vector_db": False,
        "connection_string": "DEEPFACE_POSTGRES_URI",
        "client": PostgresClient,
    },
    "mongo": {
        "is_vector_db": False,
        "connection_string": "DEEPFACE_MONGO_URI",
        "client": MongoClient,
    },
    "weaviate": {
        "is_vector_db": True,
        "connection_string": "DEEPFACE_WEAVIATE_URI",
        "client": WeaviateClient,
    },
    "neo4j": {
        "is_vector_db": True,
        "connection_string": "DEEPFACE_NEO4J_URI",
        "client": Neo4jClient,
    },
    "pgvector": {
        "is_vector_db": True,
        "connection_string": "DEEPFACE_POSTGRES_URI",
        "client": PGVectorClient,
    },
    "pinecone": {
        "is_vector_db": True,
        "connection_string": "DEEPFACE_PINECONE_API_KEY",
        "client": PineconeClient,
    },
}
