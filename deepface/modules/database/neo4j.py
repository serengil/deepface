# built-in dependencies
import os
import json
import hashlib
import struct
from typing import Any, Dict, Optional, List, Union
from urllib.parse import urlparse


# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.modeling import build_model
from deepface.modules.verification import find_cosine_distance, find_euclidean_distance
from deepface.commons.logger import Logger

logger = Logger()

_SCHEMA_CHECKED: Dict[str, bool] = {}


# pylint: disable=too-many-positional-arguments
class Neo4jClient(Database):
    def __init__(
        self,
        connection_details: Optional[Union[Dict[str, Any], str]] = None,
        connection: Any = None,
    ) -> None:
        # Import here to avoid mandatory dependency
        try:
            from neo4j import GraphDatabase
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "neo4j is an optional dependency, ensure the library is installed."
                "Please install using 'pip install neo4j' "
            ) from e

        self.GraphDatabase = GraphDatabase
        if connection is not None:
            self.conn = connection
        else:
            self.conn_details = connection_details or os.environ.get("DEEPFACE_NEO4J_URI")
            if not self.conn_details:
                raise ValueError(
                    "Neo4j connection information not found. "
                    "Please provide connection_details or set the DEEPFACE_NEO4J_URI"
                    " environment variable."
                )

            if isinstance(self.conn_details, str):
                parsed = urlparse(self.conn_details)
                uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                self.conn = self.GraphDatabase.driver(uri, auth=(parsed.username, parsed.password))
            else:
                raise ValueError("connection_details must be a string.")

        if not self.__is_gds_installed():
            raise ValueError(
                "Neo4j Graph Data Science (GDS) plugin is not installed. "
                "Please install the GDS plugin to use Neo4j as a database backend."
            )

    def close(self) -> None:
        """
        Close the Neo4j database connection.
        """
        if self.conn:
            self.conn.close()
            logger.debug("Neo4j connection closed.")

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure Neo4j database has the necessary constraints and indexes for storing embeddings.
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        node_label = self.__generate_node_label(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        model = build_model(task="facial_recognition", model_name=model_name)
        dimensions = model.output_shape
        similarity_function = "cosine" if l2_normalized else "euclidean"

        if _SCHEMA_CHECKED.get(node_label):
            logger.debug(f"Neo4j index {node_label} already exists, skipping creation.")
            return

        index_query = f"""
            CREATE VECTOR INDEX {node_label}_embedding_idx IF NOT EXISTS
            FOR (d:{node_label})
            ON (d.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }};
        """

        uniq_query = f"""
            CREATE CONSTRAINT {node_label}_unique IF NOT EXISTS
            FOR (n:{node_label})
            REQUIRE (n.face_hash, n.embedding_hash) IS UNIQUE;
        """

        with self.conn.session() as session:
            session.execute_write(lambda tx: tx.run(index_query))
            session.execute_write(lambda tx: tx.run(uniq_query))

        _SCHEMA_CHECKED[node_label] = True
        logger.debug(f"Neo4j index {node_label} ensured.")

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert embeddings into Neo4j database in batches.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        node_label = self.__generate_node_label(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )

        query = f"""
        UNWIND $rows AS r
        MERGE (n:{node_label} {{face_hash: r.face_hash, embedding_hash: r.embedding_hash}})
        ON CREATE SET
          n.img_name = r.img_name,
          n.embedding = r.embedding,
          n.face = r.face,
          n.model_name = r.model_name,
          n.detector_backend = r.detector_backend,
          n.aligned = r.aligned,
          n.l2_normalized = r.l2_normalized
        RETURN count(*) AS processed
        """

        total = 0
        with self.conn.session() as session:
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i : i + batch_size]
                rows = []
                for e in batch:
                    face_json = json.dumps(e["face"].tolist())
                    face_hash = hashlib.sha256(face_json.encode()).hexdigest()
                    embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])
                    embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

                    rows.append(
                        {
                            "face_hash": face_hash,
                            "embedding_hash": embedding_hash,
                            "img_name": e["img_name"],
                            "embedding": e["embedding"],
                            # "face": e["face"].tolist(),
                            # "face_shape": list(e["face"].shape),
                            "model_name": e.get("model_name"),
                            "detector_backend": e.get("detector_backend"),
                            "aligned": bool(e.get("aligned", True)),
                            "l2_normalized": bool(e.get("l2_normalized", False)),
                        }
                    )

                processed = session.execute_write(
                    lambda tx, q=query, r=rows: int(tx.run(q, rows=r).single()["processed"])
                )
                total += processed

        return total

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings from Neo4j database in batches.
        """
        node_label = self.__generate_node_label(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        query = f"""
        MATCH (n:{node_label})
        WHERE n.embedding IS NOT NULL
        AND ($last_eid IS NULL OR elementId(n) > $last_eid)
        RETURN
        elementId(n) AS cursor,
        coalesce(n.id, elementId(n)) AS id,
        n.img_name AS img_name,
        n.embedding AS embedding
        ORDER BY cursor ASC
        LIMIT $limit
        """

        out: List[Dict[str, Any]] = []
        last_eid: Optional[str] = None
        with self.conn.session() as session:
            while True:
                result = session.run(query, last_eid=last_eid, limit=batch_size)
                rows = list(result)
                if not rows:
                    break

                for r in rows:
                    out.append(
                        {
                            "id": r["id"],
                            "img_name": r["img_name"],
                            "embedding": r["embedding"],
                            "model_name": model_name,
                            "detector_backend": detector_backend,
                            "aligned": aligned,
                            "l2_normalized": l2_normalized,
                        }
                    )

                # advance cursor using elementId
                last_eid = rows[-1]["cursor"]

        return out

    def search_by_vector(
        self,
        vector: List[float],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        ANN search using the main vector (embedding).
        """
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        node_label = self.__generate_node_label(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        index_name = f"{node_label}_embedding_idx"

        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $vector)
        YIELD node, score
        RETURN
          elementId(node) AS id,
          node.img_name AS img_name,
          node.face_hash AS face_hash,
          node.embedding AS embedding,
          node.embedding_hash AS embedding_hash,
          score AS score
        ORDER BY score DESC
        """

        with self.conn.session() as session:
            result = session.run(
                query,
                index_name=index_name,
                limit=limit,
                vector=vector,
            )
            out: List[Dict[str, Any]] = []
            for r in result:

                if l2_normalized:
                    distance = find_cosine_distance(vector, r.get("embedding"))
                    # distance = 2 * (1 - r.get("score"))
                else:
                    distance = find_euclidean_distance(vector, r.get("embedding"))
                    # distance = math.sqrt(1.0 / r.get("score"))

                out.append(
                    {
                        "id": r.get("id"),
                        "img_name": r.get("img_name"),
                        "face_hash": r.get("face_hash"),
                        "embedding_hash": r.get("embedding_hash"),
                        "model_name": model_name,
                        "detector_backend": detector_backend,
                        "aligned": aligned,
                        "l2_normalized": l2_normalized,
                        "distance": distance,
                    }
                )

        return out

    def __is_gds_installed(self) -> bool:
        """
        Check if the Graph Data Science (GDS) plugin is installed in the Neo4j database.
        """
        query = "RETURN gds.version() AS version"
        try:
            with self.conn.session() as session:
                result = session.run(query).single()
                logger.debug(f"GDS version: {result['version']}")
                return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"GDS plugin not installed or error occurred: {e}")
            return False

    @staticmethod
    def __generate_node_label(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate a Neo4j node label based on model and preprocessing parameters.
        """
        label_parts = [
            model_name.replace("-", "_").capitalize(),
            detector_backend.capitalize(),
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "".join(label_parts)
