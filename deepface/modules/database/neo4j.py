# built-in dependencies
import os
import json
import hashlib
import struct
from typing import Any, Dict, Optional, List, Tuple, Union
from itertools import combinations
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
          n.l2_normalized = r.l2_normalized,
          n.age = r.age,
          n.gender = r.gender,
          n.emotion = r.emotion,
          n.race = r.race
        RETURN count(*) AS processed
        """

        rows = []
        for e in embeddings:
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
                    "age": e.get("age"),
                    "gender": e.get("gender"),
                    "emotion": e.get("emotion"),
                    "race": e.get("race"),
                }
            )

        total = 0
        with self.conn.session() as session:
            for i in range(0, len(rows), batch_size):
                processed = session.execute_write(
                    lambda tx, q=query, r=rows[i : i + batch_size]: int(
                        tx.run(q, rows=r).single()["processed"]
                    )
                )
                total += processed

        self.__link_co_occurring_faces(
            node_label=node_label,
            rows=rows,
            img_indexes=[e.get("img_index") for e in embeddings],
            batch_size=batch_size,
        )

        return total

    def __link_co_occurring_faces(
        self,
        node_label: str,
        rows: List[Dict[str, Any]],
        img_indexes: List[Optional[int]],
        batch_size: int = 100,
    ) -> int:
        """
        Connect faces detected in the same source image with APPEARS_WITH relationships.
            Relationship is stored once per pair, from the node with the smaller key to the
            other one, so that re-registering the same image does not duplicate it.
        Args:
            node_label (str): Node label storing the faces.
            rows (List[Dict[str, Any]]): Inserted rows having face_hash and embedding_hash.
            img_indexes (List[Optional[int]]): Source image index of each row. Rows without
                an index are not linked.
            batch_size (int): Number of relationships to merge per transaction.
        Returns:
            int: Number of face pairs processed.
        """
        groups: Dict[int, List[Tuple[str, str]]] = {}
        for row, img_index in zip(rows, img_indexes):
            if img_index is None:
                continue
            groups.setdefault(img_index, []).append((row["face_hash"], row["embedding_hash"]))

        pairs = set()
        for keys in groups.values():
            for src, dst in combinations(sorted(set(keys)), 2):
                pairs.add((src, dst))

        if not pairs:
            return 0

        query = f"""
        UNWIND $pairs AS p
        MATCH (a:{node_label} {{face_hash: p.src_face_hash, embedding_hash: p.src_embedding_hash}})
        MATCH (b:{node_label} {{face_hash: p.dst_face_hash, embedding_hash: p.dst_embedding_hash}})
        MERGE (a)-[:APPEARS_WITH]->(b)
        RETURN count(*) AS processed
        """

        payload = [
            {
                "src_face_hash": src[0],
                "src_embedding_hash": src[1],
                "dst_face_hash": dst[0],
                "dst_embedding_hash": dst[1],
            }
            for src, dst in sorted(pairs)
        ]

        return self.__run_in_batches(query=query, payload=payload, batch_size=batch_size)

    def link_verified_identities(
        self,
        clusters: List[List[str]],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
        batch_size: int = 100,
    ) -> int:
        """
        Connect nodes verified as the same person with VERIFIED relationships. Each cluster
            is the list of node ids matched to one face in a search, and every pair within a
            cluster is connected. Relationship is stored once per pair, from the node with the
            smaller id to the other one, so that repeated searches do not duplicate it.
        Args:
            clusters (List[List[str]]): Lists of node ids verified as the same person.
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
            batch_size (int): Number of relationships to merge per transaction.
        Returns:
            int: Number of node pairs processed.
        """
        node_label = self.__generate_node_label(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        pairs = set()
        for cluster in clusters:
            for src, dst in combinations(sorted(set(cluster)), 2):
                pairs.add((src, dst))

        if not pairs:
            return 0

        query = f"""
        UNWIND $pairs AS p
        MATCH (a:{node_label}) WHERE elementId(a) = p.src
        MATCH (b:{node_label}) WHERE elementId(b) = p.dst
        MERGE (a)-[:VERIFIED]->(b)
        RETURN count(*) AS processed
        """

        payload = [{"src": src, "dst": dst} for src, dst in sorted(pairs)]

        return self.__run_in_batches(query=query, payload=payload, batch_size=batch_size)

    def __run_in_batches(
        self, query: str, payload: List[Dict[str, Any]], batch_size: int = 100
    ) -> int:
        """
        Run an UNWIND $pairs query over the payload in batches, one transaction per batch.
        """
        total = 0
        with self.conn.session() as session:
            for i in range(0, len(payload), batch_size):
                total += session.execute_write(
                    lambda tx, q=query, p=payload[i : i + batch_size]: int(
                        tx.run(q, pairs=p).single()["processed"]
                    )
                )
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

    def fetch_embedding(
        self,
        identity_id: Union[str, int],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single embedding record with its vector from Neo4j. Criteria arguments are
            required to find the node label storing the record.
        Args:
            identity_id (str or int): ID of the node to fetch.
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            Optional[Dict[str, Any]]: Embedding record, or None if no record found for given id.
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
        AND coalesce(n.id, elementId(n)) = $identity_id
        RETURN
          coalesce(n.id, elementId(n)) AS id,
          n.img_name AS img_name,
          n.model_name AS model_name,
          n.detector_backend AS detector_backend,
          n.aligned AS aligned,
          n.l2_normalized AS l2_normalized,
          n.embedding AS embedding
        LIMIT 1
        """

        with self.conn.session() as session:
            record = session.run(query, identity_id=identity_id).single()

        if record is None:
            return None

        return {
            "id": record["id"],
            "img_name": record["img_name"],
            "model_name": record["model_name"],
            "detector_backend": record["detector_backend"],
            "aligned": record["aligned"],
            "l2_normalized": record["l2_normalized"],
            "embedding": record["embedding"],
        }

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
