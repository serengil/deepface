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
from deepface.modules.database.milvus import MilvusClient
from deepface.modules.database.qdrant import QdrantClient
from deepface.modules.database.sqlite import SqliteClient
from deepface.modules.database.oracle import OracleClient
from deepface.modules.database.mssql import MsSqlClient
from deepface.modules.database.db2 import Db2Client
from deepface.modules.database.redis import RedisClient
from deepface.modules.database.cassandra import CassandraClient
from deepface.modules.database.mysql import MySqlClient


class DatabaseSpec(TypedDict):
    is_vector_db: bool
    is_graph_db: bool
    connection_string: str
    client: Type["Database"]


database_inventory: Dict[str, DatabaseSpec] = {
    "postgres": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_POSTGRES_URI",
        "client": PostgresClient,
    },
    "mongo": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_MONGO_URI",
        "client": MongoClient,
    },
    "weaviate": {
        "is_vector_db": True,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_WEAVIATE_URI",
        "client": WeaviateClient,
    },
    "neo4j": {
        "is_vector_db": True,
        "is_graph_db": True,
        "connection_string": "DEEPFACE_NEO4J_URI",
        "client": Neo4jClient,
    },
    "pgvector": {
        "is_vector_db": True,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_POSTGRES_URI",
        "client": PGVectorClient,
    },
    "pinecone": {
        "is_vector_db": True,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_PINECONE_API_KEY",
        "client": PineconeClient,
    },
    "milvus": {
        "is_vector_db": True,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_MILVUS_URI",
        "client": MilvusClient,
    },
    "qdrant": {
        "is_vector_db": True,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_QDRANT_URI",
        "client": QdrantClient,
    },
    "sqlite": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_SQLITE_PATH",
        "client": SqliteClient,
    },
    "oracle": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_ORACLE_URI",
        "client": OracleClient,
    },
    "mssql": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_MSSQL_URI",
        "client": MsSqlClient,
    },
    "db2": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_DB2_URI",
        "client": Db2Client,
    },
    "redis": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_REDIS_URI",
        "client": RedisClient,
    },
    "cassandra": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_CASSANDRA_URI",
        "client": CassandraClient,
    },
    "mysql": {
        "is_vector_db": False,
        "is_graph_db": False,
        "connection_string": "DEEPFACE_MYSQL_URI",
        "client": MySqlClient,
    },
}
