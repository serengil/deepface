# built-in dependencies
import os
import json
import hashlib
import struct
import base64
import uuid
import math
import functools
import re
from typing import Any, Dict, Optional, List, Set, Tuple, Union
from urllib.parse import urlparse

# project dependencies
from deepface import __version__
from deepface.modules.database.types import Database
from deepface.commons.logger import Logger

logger = Logger()

# Weaviate uses this header to learn which integrations talk to it
INTEGRATION_HEADER = "X-Weaviate-Client-Integration"
INTEGRATION_NAME = f"deepface/{__version__}"

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50051
CLOUD_HOST_SUFFIXES = (".weaviate.cloud", ".weaviate.network")
DEPLOYMENTS = ("custom", "cloud", "local")

# keys accepted in connection_details. Names follow the weaviate client's connect_to_* helpers.
CONNECTION_KEYS = {
    "url",
    "deployment",
    "api_key",
    "auth",
    "headers",
    "http_host",
    "http_port",
    "http_secure",
    "grpc_host",
    "grpc_port",
    "grpc_secure",
    "skip_init_checks",
    "timeout",
    "proxies",
    "trust_env",
    "connection_config",
    "grpc_config",
    "additional_config",
}

# options build_additional_config turns into an AdditionalConfig
ADDITIONAL_CONFIG_KEYS = {"timeout", "proxies", "trust_env", "connection_config", "grpc_config"}

# keys of the grpc_config option, the fields of weaviate.config.GrpcConfig
GRPC_CONFIG_KEYS = {"channel_options", "credentials"}

# keys that configure the collections deepface creates, not the connection
COLLECTION_KEYS = {"tenant", "quantization"}

# quantization option -> (minimum weaviate version, vectorIndexConfig entries). Servers older
# than the minimum silently drop unknown quantizers, so the version is checked here.
QUANTIZATIONS: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {
    # also opts out of a server side DEFAULT_QUANTIZATION, e.g. on Weaviate Cloud
    "none": ((1, 0), {"skipDefaultQuantization": True}),
    "rq": ((1, 32), {"rq": {"enabled": True, "bits": 8}}),
    "rq-8": ((1, 32), {"rq": {"enabled": True, "bits": 8}}),
    "rq-1": ((1, 33), {"rq": {"enabled": True, "bits": 1}}),
    "bq": ((1, 24), {"bq": {"enabled": True}}),
    "sq": ((1, 26), {"sq": {"enabled": True}}),
    "pq": ((1, 23), {"pq": {"enabled": True}}),
}

# auto tenant creation and activation need weaviate 1.25
MIN_TENANCY_VERSION = (1, 25)
TENANT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# pylint: disable=too-many-positional-arguments
class WeaviateClient(Database):
    """
    Weaviate client for storing and retrieving face embeddings and indices.
    Requires weaviate-client v4 (pip install "weaviate-client>=4.16.0"), which talks to
    Weaviate over REST and gRPC, and Weaviate 1.27 or newer.

    connection_details is a url, or a dict with any of the options below. The same dict can
    also be given as a JSON string, e.g. in DEEPFACE_CONNECTION_DETAILS for the API.

      # self-hosted, gRPC on the default port 50051
      connection_details = "http://localhost:8080"

      # Weaviate Cloud
      connection_details = {"url": "https://my-cluster.weaviate.cloud", "api_key": "..."}

      # custom deployment
      connection_details = {
          "url": "https://weaviate.example.com",
          "grpc_host": "grpc.weaviate.example.com",
          "grpc_port": 443,
          "auth": {"client_secret": "...", "scope": "openid"},
          "timeout": {"init": 5, "query": 60, "insert": 120},
          "skip_init_checks": False,
      }

    Options:
      url                 http(s) url of the REST endpoint. Without a port, http:// uses 80,
                          https:// 443 and a host without a scheme 8080
      deployment          custom, cloud or local. Inferred from the url when not given:
                          *.weaviate.cloud and *.weaviate.network hosts are cloud
      http_host, http_port, http_secure
                          override what the url gives
      grpc_host, grpc_port, grpc_secure
                          default to the http host, 50051 and the http scheme
      api_key             shortcut for {"auth": {"api_key": ...}}
      auth                {"api_key"}, {"access_token", "expires_in", "refresh_token"},
                          {"client_secret", "scope"} (OIDC client credentials),
                          {"username", "password", "scope"} (OIDC password),
                          or an object from weaviate.classes.init.Auth
      timeout             seconds for query and insert, [init, query, insert], or a dict
                          with init, query, insert and stream keys
      headers             extra request headers
      skip_init_checks, proxies, trust_env
                          passed to the weaviate client as is
      connection_config   kwargs of weaviate.config.ConnectionConfig
      grpc_config         {"channel_options", "credentials"}: extra gRPC channel options,
                          e.g. [("grpc.keepalive_time_ms", 10000)], and grpc.ChannelCredentials
                          for TLS, e.g. a private CA. Works with any supported client version
      additional_config   a ready weaviate.classes.init.AdditionalConfig, used instead of
                          timeout, proxies, trust_env, connection_config and grpc_config

    Unknown options raise an error. When an option is not given, these environment variables
    are used: DEEPFACE_WEAVIATE_URI (the url or the JSON object), WEAVIATE_API_KEY,
    DEEPFACE_WEAVIATE_GRPC_PORT, DEEPFACE_WEAVIATE_TIMEOUT and
    DEEPFACE_WEAVIATE_SKIP_INIT_CHECKS. A weaviate.WeaviateClient can also be passed as
    connection, and is then used as is.

    Two more options configure the collections deepface creates. They can be combined with
    an existing connection too, e.g. connection_details={"tenant": "acme"}.

      tenant              keeps each gallery of faces isolated with weaviate multi-tenancy,
                          e.g. one tenant per customer, site or event sharing a cluster.
                          A search only sees faces registered with the same tenant, tenants
                          are created on their first register, and deleting a tenant erases
                          its faces. Collections are created as multi-tenant only when a
                          tenant is given, and an existing collection keeps its mode. Needs
                          Weaviate 1.25 or newer, and with RBAC the read and create tenant
                          permissions. Falls back to DEEPFACE_WEAVIATE_TENANT.
      quantization        compresses the vectors of new collections: rq (8-bit, Weaviate
                          1.32+), rq-1 (1-bit, 1.33+), bq, sq, pq, or none to opt out of a
                          server default quantization. Weaviate rescores results with the
                          original vectors, so distances and thresholds are unchanged. When
                          not given, the server default applies. Falls back to
                          DEEPFACE_WEAVIATE_QUANTIZATION.
    """

    def __init__(
        self,
        connection_details: Optional[Union[str, Dict[str, Any]]] = None,
        connection: Any = None,
    ):
        try:
            import weaviate
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "weaviate-client is an optional dependency. "
                "Install with 'pip install \"weaviate-client>=4.16.0\"'"
            ) from e

        self.weaviate = weaviate

        # resolved before connecting, so invalid options don't leave a connection open
        options = resolve_collection_options(connection_details)
        self.tenant: Optional[str] = options["tenant"]
        self.quantization: Optional[str] = options["quantization"]
        self.__server_version: Optional[Tuple[int, int]] = None

        if not hasattr(weaviate, "WeaviateClient"):
            raise ValueError(
                "weaviate-client v4 is required, but an older version is installed. "
                "Upgrade with 'pip install -U \"weaviate-client>=4.16.0\"'"
            )

        if connection is not None:
            if not isinstance(connection, weaviate.WeaviateClient):
                raise ValueError(
                    "connection must be a weaviate.WeaviateClient created with weaviate-client"
                    " v4 (e.g. weaviate.connect_to_local()). weaviate.Client from v3 is no"
                    " longer supported."
                )
            self.client = connection
        else:
            self.conn_details = resolve_connection_details(connection_details)
            self.client = connect(self.conn_details)

    def initialize_database(self, **kwargs: Any) -> None:
        """
        Ensure the Weaviate collection for the given criteria exists. Collections storing
            l2 normalized embeddings use cosine distance, others use l2-squared.
        """
        model_name = kwargs.get("model_name", "VGG-Face")
        detector_backend = kwargs.get("detector_backend", "opencv")
        aligned = kwargs.get("aligned", True)
        l2_normalized = kwargs.get("l2_normalized", False)

        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        # not cached, so a collection dropped outside deepface is created again
        if self.client.collections.exists(class_name):
            logger.debug(f"Weaviate collection {class_name} already exists.")
            self.__check_collection_config(class_name)
            return

        vector_index_config: Dict[str, Any] = {
            "distance": "cosine" if l2_normalized else "l2-squared",
        }
        if os.getenv("WEAVIATE_HNSW_M"):
            vector_index_config["maxConnections"] = int(os.environ["WEAVIATE_HNSW_M"])

        if self.quantization is not None:
            min_version, quantizer_config = QUANTIZATIONS[self.quantization]
            if self.quantization != "none":
                self.__require_server_version(min_version, f"quantization '{self.quantization}'")
            vector_index_config.update(quantizer_config)

        schema: Dict[str, Any] = {
            "class": class_name,
            "vectorIndexType": "hnsw",
            "vectorizer": "none",
            "vectorIndexConfig": vector_index_config,
        }

        if self.tenant is not None:
            self.__require_server_version(MIN_TENANCY_VERSION, "multi-tenancy")
            schema["multiTenancyConfig"] = {
                "enabled": True,
                "autoTenantCreation": True,
                "autoTenantActivation": True,
            }

        self.client.collections.create_from_dict(
            {
                **schema,
                "properties": [
                    {"name": "img_name", "dataType": ["text"]},
                    {"name": "face", "dataType": ["blob"]},
                    {"name": "face_shape", "dataType": ["int[]"]},
                    {"name": "model_name", "dataType": ["text"]},
                    {"name": "detector_backend", "dataType": ["text"]},
                    {"name": "aligned", "dataType": ["boolean"]},
                    {"name": "l2_normalized", "dataType": ["boolean"]},
                    {"name": "face_hash", "dataType": ["text"]},
                    {"name": "embedding_hash", "dataType": ["text"]},
                    # embedding property is optional since we pass it as vector
                    {"name": "embedding", "dataType": ["number[]"]},
                ],
            }
        )

        logger.debug(f"Weaviate collection {class_name} created successfully.")

    def insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Insert multiple embeddings into Weaviate using the batch API. Embeddings already
            stored in the collection are skipped.
        Returns:
            inserted (int): number of embeddings actually inserted.
        """
        if not embeddings:
            raise ValueError("No embeddings to insert.")

        self.initialize_database(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )
        class_name = self.__generate_class_name(
            model_name=embeddings[0]["model_name"],
            detector_backend=embeddings[0]["detector_backend"],
            aligned=embeddings[0]["aligned"],
            l2_normalized=embeddings[0]["l2_normalized"],
        )
        collection = self.__collection(class_name)

        records: Dict[str, Dict[str, Any]] = {}
        for e in embeddings:
            face_json = json.dumps(e["face"].tolist())
            face_hash = hashlib.sha256(face_json.encode()).hexdigest()
            embedding_bytes = struct.pack(f'{len(e["embedding"])}d', *e["embedding"])
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()

            if embedding_hash in records:
                logger.warn(f"Embedding with hash {embedding_hash} is duplicated in the input.")
                continue

            # ids are scoped to the tenant, so the same face gets unrelated ids in each tenant
            id_name = f"{face_hash}:{embedding_hash}"
            if self.tenant is not None:
                id_name = f"{self.tenant}:{id_name}"

            # the face is encoded only when added to the batch, after duplicates are dropped
            records[embedding_hash] = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_OID, id_name)),
                "vector": e["embedding"],
                "face": e["face"],
                "properties": {
                    "img_name": e["img_name"],
                    "face_shape": list(e["face"].shape),
                    "model_name": e["model_name"],
                    "detector_backend": e["detector_backend"],
                    "aligned": e["aligned"],
                    "l2_normalized": e["l2_normalized"],
                    "embedding": e["embedding"],  # optional
                    "face_hash": face_hash,
                    "embedding_hash": embedding_hash,
                },
            }

        # a missing tenant is created by the batch insert, and nothing is stored in it yet
        if self.__tenant_exists(collection):
            for embedding_hash in self.__find_existing_hashes(collection, list(records.keys())):
                logger.warn(
                    f"Embedding with hash {embedding_hash} already exists in {class_name}."
                )
                del records[embedding_hash]

        if not records:
            return 0

        with collection.batch.fixed_size(batch_size=batch_size) as batcher:
            for record in records.values():
                face = base64.b64encode(record["face"].tobytes()).decode("utf-8")
                batcher.add_object(
                    properties={**record["properties"], "face": face},
                    vector=record["vector"],
                    uuid=record["uuid"],
                )

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            raise ValueError(
                f"Failed to insert {len(failed_objects)} of {len(records)} embeddings into"
                f" {class_name}. First error: {failed_objects[0].message}"
            )

        return len(records)

    def fetch_all_embeddings(
        self,
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings with filters.
        """
        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        collection = self.__collection(class_name)
        if not self.__tenant_exists(collection):
            return []

        embeddings = []
        for obj in collection.iterator(
            return_properties=["img_name", "embedding"],
            cache_size=batch_size,
        ):
            embeddings.append(
                {
                    "id": str(obj.uuid),
                    "img_name": obj.properties["img_name"],
                    "embedding": obj.properties["embedding"],
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": aligned,
                    "l2_normalized": l2_normalized,
                }
            )
        return embeddings

    def fetch_embedding(
        self,
        identity_id: Union[str, int],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        aligned: bool = True,
        l2_normalized: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single embedding record with its vector from Weaviate. Criteria arguments are
            required to find the class storing the record.
        Args:
            identity_id (str or int): uuid of the object to fetch.
            model_name (str): Name of the model.
            detector_backend (str): Name of the detector backend.
            aligned (bool): Whether the faces are aligned.
            l2_normalized (bool): Whether the embeddings are L2 normalized.
        Returns:
            Optional[Dict[str, Any]]: Embedding record, or None if no record found for given id.
        """
        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        # weaviate object ids are uuids, so any other id cannot be registered here
        try:
            uuid.UUID(str(identity_id))
        except ValueError:
            return None

        if not self.client.collections.exists(class_name):
            return None
        self.__check_collection_config(class_name)

        collection = self.__collection(class_name)
        if not self.__tenant_exists(collection):
            return None

        obj = collection.query.fetch_object_by_id(
            str(identity_id),
            include_vector=True,
        )

        if obj is None:
            return None

        properties = obj.properties

        return {
            "id": str(obj.uuid),
            "img_name": properties.get("img_name"),
            "model_name": properties.get("model_name"),
            "detector_backend": properties.get("detector_backend"),
            "aligned": properties.get("aligned"),
            "l2_normalized": properties.get("l2_normalized"),
            # embedding is stored both as a property and as the vector of the object
            "embedding": properties.get("embedding") or obj.vector.get("default"),
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
        from weaviate.classes.query import MetadataQuery

        class_name = self.__generate_class_name(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )
        self.initialize_database(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=aligned,
            l2_normalized=l2_normalized,
        )

        collection = self.__collection(class_name)
        if not self.__tenant_exists(collection):
            return []

        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_properties=["img_name", "embedding"],
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance or 0.0
            results.append(
                {
                    "id": str(obj.uuid),
                    "img_name": obj.properties["img_name"],
                    "embedding": obj.properties["embedding"],
                    # l2-squared distance is converted to euclidean distance
                    "distance": distance if l2_normalized else math.sqrt(max(distance, 0.0)),
                }
            )
        return results

    def close(self) -> None:
        """
        Close the Weaviate client connection.
        """
        self.client.close()

    def __collection(self, class_name: str) -> Any:
        """
        Get a collection handle, scoped to the tenant when multi-tenancy is used.
        """
        collection = self.client.collections.use(class_name)
        if self.tenant is not None:
            collection = collection.with_tenant(self.tenant)
        return collection

    def __tenant_exists(self, collection: Any) -> bool:
        """
        Check that the tenant exists. Tenants are created on the first insert only.
        """
        if self.tenant is None:
            return True
        return bool(collection.tenants.exists(self.tenant))

    def __check_collection_config(self, class_name: str) -> None:
        """
        Check that an existing collection matches the tenant and quantization options.
        """
        config = self.client.collections.use(class_name).config.get()

        multi_tenancy = bool(config.multi_tenancy_config.enabled)
        if self.tenant is not None and not multi_tenancy:
            raise ValueError(
                f"Weaviate collection {class_name} was created without multi-tenancy, so it"
                " cannot be used with a tenant. Remove the tenant option, or drop the"
                " collection to have it created again with multi-tenancy."
            )
        if self.tenant is None and multi_tenancy:
            raise ValueError(
                f"Weaviate collection {class_name} has multi-tenancy enabled, so a tenant is"
                " required. Set the tenant option or DEEPFACE_WEAVIATE_TENANT."
            )

        if self.quantization is not None:
            current = quantizer_name(config.vector_index_config)
            if current != self.quantization.replace("rq-8", "rq"):
                logger.warn(
                    f"Weaviate collection {class_name} already exists with quantization"
                    f" '{current}', so quantization '{self.quantization}' is not applied."
                    " Quantization is only set when a collection is created."
                )

    def __require_server_version(self, min_version: Tuple[int, int], feature: str) -> None:
        """
        Raise if the weaviate server is older than the given version.
        """
        if self.__server_version is None:
            self.__server_version = parse_version(self.client.get_meta().get("version", ""))
        if self.__server_version < min_version:
            raise ValueError(
                f"{feature} requires Weaviate {min_version[0]}.{min_version[1]} or newer,"
                f" but the server is {self.__server_version[0]}.{self.__server_version[1]}."
            )

    @staticmethod
    def __find_existing_hashes(collection: Any, embedding_hashes: List[str]) -> Set[str]:
        """
        Find which embedding hashes are already stored in the collection.
        """
        from weaviate.classes.query import Filter

        existing: Set[str] = set()
        chunk_size = 100
        for i in range(0, len(embedding_hashes), chunk_size):
            remaining = set(embedding_hashes[i : i + chunk_size])
            # a hash may be stored more than once, so query again until no new hash is found
            while remaining:
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("embedding_hash").contains_any(list(remaining)),
                    return_properties=["embedding_hash"],
                    limit=len(remaining),
                )
                found = {obj.properties["embedding_hash"] for obj in response.objects}
                found &= remaining
                if not found:
                    break
                existing |= found
                remaining -= found
        return existing

    @staticmethod
    def __generate_class_name(
        model_name: str,
        detector_backend: str,
        aligned: bool,
        l2_normalized: bool,
    ) -> str:
        """
        Generate Weaviate class name based on parameters.
        """
        class_name_attributes = [
            model_name.replace("-", ""),
            detector_backend,
            "Aligned" if aligned else "Unaligned",
            "Norm" if l2_normalized else "Raw",
        ]
        return "Embeddings_" + "_".join(class_name_attributes).lower()


def resolve_connection_details(
    connection_details: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Normalize connection details into a dict, applying environment variable fallbacks.
    Args:
        connection_details (str or dict): a Weaviate URL, a JSON object string, or a dict
            whose keys are listed in CONNECTION_KEYS. Falls back to DEEPFACE_WEAVIATE_URI.
    Returns:
        details (dict): connection details.
    """
    if connection_details is None:
        connection_details = os.getenv("DEEPFACE_WEAVIATE_URI") or os.getenv(
            "DEEPFACE_WEAVIATE_URL"
        )

    details: Dict[str, Any]
    if isinstance(connection_details, dict):
        details = dict(connection_details)
    elif isinstance(connection_details, str) and connection_details.strip().startswith("{"):
        parsed = json.loads(connection_details)
        if not isinstance(parsed, dict):
            raise ValueError("Weaviate connection details JSON must be an object.")
        details = parsed
    elif isinstance(connection_details, str) and connection_details.strip():
        details = {"url": connection_details.strip()}
    elif connection_details is None:
        raise ValueError(
            "Weaviate connection details not provided. Pass connection_details or set"
            " DEEPFACE_WEAVIATE_URI."
        )
    else:
        raise ValueError("connection_details must be a string or dict with 'url'.")

    validate_option_keys(details)

    # a ready AdditionalConfig replaces the options deepface would build it from
    if details.get("additional_config") is not None:
        overridden = sorted(k for k in ADDITIONAL_CONFIG_KEYS if details.get(k) is not None)
        if overridden:
            raise ValueError(
                f"additional_config cannot be combined with {overridden}."
                " Set them in the AdditionalConfig instead."
            )

    env_fallbacks = {
        "grpc_port": os.getenv("DEEPFACE_WEAVIATE_GRPC_PORT"),
        "skip_init_checks": os.getenv("DEEPFACE_WEAVIATE_SKIP_INIT_CHECKS"),
        "timeout": os.getenv("DEEPFACE_WEAVIATE_TIMEOUT"),
    }
    if details.get("additional_config") is not None:
        del env_fallbacks["timeout"]
    for key, value in env_fallbacks.items():
        if value and key not in details:
            details[key] = value

    if "api_key" not in details and "auth" not in details and os.getenv("WEAVIATE_API_KEY"):
        details["api_key"] = os.getenv("WEAVIATE_API_KEY")

    deployment = details.get("deployment") or infer_deployment(details)
    if deployment == "embedded":
        raise ValueError("Embedded Weaviate is not supported by deepface.")
    if deployment not in DEPLOYMENTS:
        raise ValueError(f"deployment must be one of {DEPLOYMENTS}, got {deployment!r}.")
    details["deployment"] = deployment

    if deployment == "cloud" and not details.get("url"):
        raise ValueError("Weaviate Cloud requires the cluster url in connection_details.")
    if deployment == "custom" and not details.get("url") and not details.get("http_host"):
        raise ValueError("Weaviate URL not provided in connection_details.")

    return details


def validate_option_keys(details: Dict[str, Any]) -> None:
    """
    Raise on unknown connection details keys, so typos are not silently ignored.
    """
    valid_keys = CONNECTION_KEYS | COLLECTION_KEYS
    unknown = set(details.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown Weaviate connection option(s): {sorted(unknown)}."
            f" Valid options are: {sorted(valid_keys)}"
        )


def resolve_collection_options(
    connection_details: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    """
    Resolve the tenant and quantization options from connection details, falling back to
        DEEPFACE_WEAVIATE_TENANT and DEEPFACE_WEAVIATE_QUANTIZATION. They are read even when
        an existing connection is passed, so connection_details may hold only these keys.
    Returns:
        options (dict): tenant and quantization, None when not set.
    """
    if connection_details is None:
        # the same fallback as the connection, which may be a JSON object with these options
        connection_details = os.getenv("DEEPFACE_WEAVIATE_URI") or os.getenv(
            "DEEPFACE_WEAVIATE_URL"
        )

    details: Dict[str, Any] = {}
    if isinstance(connection_details, dict):
        details = connection_details
    elif isinstance(connection_details, str) and connection_details.strip().startswith("{"):
        details = json.loads(connection_details)
        if not isinstance(details, dict):
            raise ValueError("Weaviate connection details JSON must be an object.")
    validate_option_keys(details)

    tenant = details.get("tenant")
    if tenant is not None and not str(tenant).strip():
        # an empty tenant would silently store faces outside any tenant
        raise ValueError("Weaviate tenant is empty. Remove the option or give a tenant name.")
    tenant = tenant or os.getenv("DEEPFACE_WEAVIATE_TENANT") or None
    if tenant is not None:
        tenant = str(tenant)
        if not TENANT_PATTERN.match(tenant):
            raise ValueError(
                f"Invalid Weaviate tenant name {tenant!r}. Tenant names are 1 to 64"
                " characters of letters, digits, '_' and '-'."
            )

    quantization = details.get("quantization") or os.getenv("DEEPFACE_WEAVIATE_QUANTIZATION")
    if quantization is not None:
        quantization = str(quantization).strip().lower()
        if quantization not in QUANTIZATIONS:
            raise ValueError(
                f"Unsupported Weaviate quantization {quantization!r}."
                f" Supported values are: {sorted(QUANTIZATIONS)}"
            )

    return {"tenant": tenant, "quantization": quantization}


def quantizer_name(vector_index_config: Any) -> str:
    """
    Name the quantizer of a collection's vector index config as a quantization option.
    """
    quantizer = getattr(vector_index_config, "quantizer", None)
    if quantizer is None:
        return "none"
    name = type(quantizer).__name__.lower().strip("_")
    for option in ("rq", "bq", "sq", "pq"):
        if name.startswith(option):
            if option == "rq" and getattr(quantizer, "bits", 8) == 1:
                return "rq-1"
            return option
    return name


def parse_version(version: str) -> Tuple[int, int]:
    """
    Parse major and minor version numbers from a version string such as 1.33.2.
    """
    match = re.match(r"^v?(\d+)\.(\d+)", version.strip())
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def infer_deployment(details: Dict[str, Any]) -> str:
    """
    Infer the deployment type from the connection details.
    """
    url = details.get("url")
    if url:
        host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
        if host.endswith(CLOUD_HOST_SUFFIXES):
            return "cloud"
    return "custom"


def connect(details: Dict[str, Any]) -> Any:
    """
    Connect to Weaviate with one of the weaviate client's connect_to_* helpers.
    Args:
        details (dict): connection details returned by resolve_connection_details.
    Returns:
        client (weaviate.WeaviateClient): connected client.
    """
    import weaviate

    common: Dict[str, Any] = {
        "headers": build_headers(details.get("headers")),
        "additional_config": build_additional_config(details),
        "skip_init_checks": parse_bool(details.get("skip_init_checks", False)),
        "auth_credentials": build_auth(details),
    }
    deployment = details["deployment"]
    # weaviate-client before 4.20 has no GrpcConfig, so deepface applies grpc_config itself
    legacy_grpc_config = (
        details.get("grpc_config") is not None and load_grpc_config_class() is None
    )

    if deployment == "cloud":
        if common["auth_credentials"] is None:
            raise ValueError("Weaviate Cloud requires an API key or auth credentials.")
        if legacy_grpc_config:
            http_host, grpc_host = parse_cloud_hosts(details["url"])
            return connect_with_legacy_grpc_config(
                details["grpc_config"], (http_host, 443, True), (grpc_host, 443, True), common
            )
        return weaviate.connect_to_weaviate_cloud(cluster_url=details["url"], **common)

    http_host, http_port, http_secure = parse_http_endpoint(details)
    grpc_port = int(details.get("grpc_port", DEFAULT_GRPC_PORT))
    if deployment == "local":
        # connect_to_local always uses plain http and gRPC on the http host
        http_secure, grpc_host, grpc_secure = False, http_host, False
    else:
        grpc_host = details.get("grpc_host") or http_host
        grpc_secure = parse_bool(details.get("grpc_secure", http_secure))

    if common["auth_credentials"] is not None:
        warn_insecure_transports(http_host, http_secure, grpc_host, grpc_secure)

    if legacy_grpc_config:
        return connect_with_legacy_grpc_config(
            details["grpc_config"],
            (http_host, http_port, http_secure),
            (grpc_host, grpc_port, grpc_secure),
            common,
        )

    if deployment == "local":
        return weaviate.connect_to_local(
            host=http_host,
            port=http_port,
            grpc_port=grpc_port,
            **common,
        )

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
        **common,
    )


def warn_insecure_transports(
    http_host: str, http_secure: bool, grpc_host: str, grpc_secure: bool
) -> None:
    """
    Warn when credentials would be sent to a remote host without TLS, over REST or gRPC.
        The warning names the hosts, never the credentials.
    """
    transports = (("REST", http_host, http_secure), ("gRPC", grpc_host, grpc_secure))
    insecure = [
        f"{name} to {host}"
        for name, host, secure in transports
        if not secure and not is_local(host)
    ]
    if insecure:
        logger.warn(
            f"Weaviate credentials are sent without TLS over {' and '.join(insecure)}."
            " Use an https url, or set http_secure and grpc_secure, to protect them."
        )


def parse_http_endpoint(details: Dict[str, Any]) -> Tuple[str, int, bool]:
    """
    Resolve http host, port and secure flag from the url and the explicit overrides.
    """
    host, port, secure = "localhost", DEFAULT_HTTP_PORT, False
    url = details.get("url")
    if url:
        has_scheme = "://" in url
        parsed = urlparse(url if has_scheme else f"http://{url}")
        secure = parsed.scheme == "https"
        host = parsed.hostname or host
        # a url with a scheme uses its standard port, a bare host weaviate's default
        default_port = (443 if secure else 80) if has_scheme else DEFAULT_HTTP_PORT
        port = parsed.port or default_port

    secure = parse_bool(details.get("http_secure", secure))
    host = details.get("http_host") or host
    port = int(details.get("http_port", port))
    return host, port, secure


def is_local(host: str) -> bool:
    """
    Check if a host is the local machine, where plain http does not expose credentials.
    """
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def build_headers(extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build request headers, always including the integration header.
    """
    headers = dict(extra_headers or {})
    headers.setdefault(INTEGRATION_HEADER, INTEGRATION_NAME)
    return headers


def build_auth(details: Dict[str, Any]) -> Any:
    """
    Build weaviate auth credentials from connection details.
    Supported forms of the auth option:
        - {"api_key": ...}
        - {"access_token": ..., "expires_in": ..., "refresh_token": ...} (bearer token)
        - {"client_secret": ..., "scope": ...} (OIDC client credentials)
        - {"username": ..., "password": ..., "scope": ...} (OIDC resource owner password)
        - an object created with weaviate.classes.init.Auth
    """
    from weaviate.classes.init import Auth

    auth = details.get("auth")
    if auth is not None and details.get("api_key"):
        raise ValueError("Pass either api_key or auth in connection_details, not both.")
    if auth is None:
        api_key = details.get("api_key")
        return Auth.api_key(api_key) if api_key else None

    if not isinstance(auth, dict):
        return auth

    if "api_key" in auth:
        return Auth.api_key(auth["api_key"])
    if "access_token" in auth:
        return Auth.bearer_token(
            access_token=auth["access_token"],
            expires_in=int(auth.get("expires_in", 60)),
            refresh_token=auth.get("refresh_token"),
        )
    if "client_secret" in auth:
        return Auth.client_credentials(
            client_secret=auth["client_secret"],
            scope=auth.get("scope"),
        )
    if "username" in auth and "password" in auth:
        return Auth.client_password(
            username=auth["username"],
            password=auth["password"],
            scope=auth.get("scope"),
        )
    raise ValueError(
        "auth must contain one of 'api_key', 'access_token', 'client_secret'"
        " or 'username' and 'password'."
    )


def build_additional_config(details: Dict[str, Any]) -> Any:
    """
    Build weaviate AdditionalConfig from timeout, proxies, trust_env, connection_config
        and grpc_config options. A ready AdditionalConfig passed as additional_config
        is used as is.
    """
    from weaviate.classes.init import AdditionalConfig
    from weaviate.config import ConnectionConfig

    if details.get("additional_config") is not None:
        return details["additional_config"]

    kwargs: Dict[str, Any] = {}
    if details.get("timeout") is not None:
        kwargs["timeout"] = build_timeout(details["timeout"])
    if details.get("proxies") is not None:
        kwargs["proxies"] = details["proxies"]
    if details.get("trust_env") is not None:
        kwargs["trust_env"] = parse_bool(details["trust_env"])
    if details.get("connection_config") is not None:
        kwargs["connection"] = ConnectionConfig(**details["connection_config"])
    grpc_config_class = load_grpc_config_class()
    if details.get("grpc_config") is not None and grpc_config_class is not None:
        kwargs["grpc_config"] = grpc_config_class(**parse_grpc_config(details["grpc_config"]))

    return AdditionalConfig(**kwargs) if kwargs else None


def parse_grpc_config(grpc_config: Any) -> Dict[str, Any]:
    """
    Validate the grpc_config option, a dict with channel_options and credentials.
        Channel options given as lists, e.g. from JSON, become tuples.
    """
    if not isinstance(grpc_config, dict):
        raise ValueError("grpc_config must be a dict with channel_options and credentials.")
    unknown = set(grpc_config.keys()) - GRPC_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown grpc_config option(s): {sorted(unknown)}."
            f" Valid options are: {sorted(GRPC_CONFIG_KEYS)}"
        )
    parsed = dict(grpc_config)
    if parsed.get("channel_options") is not None:
        parsed["channel_options"] = [tuple(option) for option in parsed["channel_options"]]
    return parsed


def load_grpc_config_class() -> Any:
    """
    Return weaviate.config.GrpcConfig, or None on weaviate-client versions before 4.20.
    """
    try:
        from weaviate.config import GrpcConfig
    except ImportError:
        return None
    return GrpcConfig


def parse_cloud_hosts(cluster_url: str) -> Tuple[str, str]:
    """
    Resolve the http and gRPC hosts of a Weaviate Cloud cluster, as connect_to_weaviate_cloud does.
    """
    host = urlparse(cluster_url).netloc if cluster_url.startswith("http") else cluster_url
    if host.endswith(".weaviate.network"):
        ident, domain = host.split(".", 1)
        return host, f"{ident}.grpc.{domain}"
    return host, f"grpc-{host}"


def connect_with_legacy_grpc_config(
    grpc_config: Dict[str, Any],
    http: Tuple[str, int, bool],
    grpc_endpoint: Tuple[str, int, bool],
    common: Dict[str, Any],
) -> Any:
    """
    Connect with grpc_config on weaviate-client versions before 4.20, which have no GrpcConfig.
        Does what the connect_to_* helpers do, with connection params that add the channel
        options and credentials to the gRPC channel as GrpcConfig does in newer versions.
        Remove once weaviate-client 4.20 is the minimum.
    Args:
        grpc_config (dict): the grpc_config option.
        http (tuple): host, port and secure flag of the REST endpoint.
        grpc_endpoint (tuple): host, port and secure flag of the gRPC endpoint.
        common (dict): headers, additional_config, skip_init_checks and auth_credentials.
    Returns:
        client (weaviate.WeaviateClient): connected client.
    """
    import weaviate

    params = legacy_grpc_connection_params_class().from_grpc_config(
        http, grpc_endpoint, parse_grpc_config(grpc_config)
    )

    client = weaviate.WeaviateClient(
        connection_params=params,
        auth_client_secret=common["auth_credentials"],
        additional_headers=common["headers"],
        additional_config=common["additional_config"],
        skip_init_checks=common["skip_init_checks"],
    )
    try:
        client.connect()
    except Exception:
        client.close()
        raise
    return client


@functools.lru_cache(maxsize=None)
def legacy_grpc_connection_params_class() -> Any:
    """
    Build a ConnectionParams subclass whose gRPC channel applies grpc_config options.
        Created lazily because weaviate-client is an optional dependency.
    """
    # pylint: disable=import-outside-toplevel
    import grpc
    from pydantic import PrivateAttr
    from weaviate.connect.base import ConnectionParams, ProtocolParams, MAX_GRPC_MESSAGE_LENGTH

    class GrpcConfigConnectionParams(ConnectionParams):  # type: ignore[misc]
        """
        ConnectionParams with the channel options and credentials of grpc_config.
        """

        _channel_options: List[Tuple[str, Any]] = PrivateAttr(default_factory=list)
        _credentials: Any = PrivateAttr(default=None)

        @classmethod
        def from_grpc_config(
            cls,
            http: Tuple[str, int, bool],
            grpc_endpoint: Tuple[str, int, bool],
            grpc_config: Dict[str, Any],
        ) -> "GrpcConfigConnectionParams":
            """
            Build connection params from host, port and secure flag of both endpoints,
                and a grpc_config validated by parse_grpc_config.
            """
            params = cls(
                http=ProtocolParams(host=http[0], port=http[1], secure=http[2]),
                grpc=ProtocolParams(
                    host=grpc_endpoint[0], port=grpc_endpoint[1], secure=grpc_endpoint[2]
                ),
            )
            params._channel_options = list(grpc_config.get("channel_options") or [])
            params._credentials = grpc_config.get("credentials")
            return params

        def channel_options(
            self, proxies: Dict[str, str], grpc_msg_size: Optional[int]
        ) -> List[Tuple[str, Any]]:
            """
            gRPC channel options: the client defaults, then the grpc_config options.
            """
            if grpc_msg_size is None:
                grpc_msg_size = MAX_GRPC_MESSAGE_LENGTH
            options: List[Tuple[str, Any]] = [
                ("grpc.max_send_message_length", grpc_msg_size),
                ("grpc.max_receive_message_length", grpc_msg_size),
                ("grpc.default_authority", self.grpc.host),
            ]
            if proxies.get("grpc") is not None:
                options.append(("grpc.http_proxy", proxies["grpc"]))
            return options + self._channel_options

        def _grpc_channel(  # pylint: disable=unused-argument
            self,
            proxies: Dict[str, str],
            grpc_msg_size: Optional[int],
            is_async: bool,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # same channel as ConnectionParams._grpc_channel in weaviate-client 4.20
            options = self.channel_options(proxies, grpc_msg_size)
            mod = grpc.aio if is_async else grpc
            if self.grpc.secure:
                return mod.secure_channel(
                    target=self._grpc_target,
                    credentials=self._credentials or grpc.ssl_channel_credentials(),
                    options=options,
                )
            return mod.insecure_channel(target=self._grpc_target, options=options)

    return GrpcConfigConnectionParams


def build_timeout(timeout: Any) -> Any:
    """
    Build weaviate Timeout from a number (query and insert timeout), a list or comma
        separated string of init, query and insert timeouts, or a dict with any of
        init, query, insert and stream keys.
    """
    from weaviate.classes.init import Timeout

    if isinstance(timeout, Timeout):
        return timeout
    if isinstance(timeout, dict):
        return Timeout(**timeout)
    if isinstance(timeout, str):
        timeout = [float(value) for value in timeout.split(",")]
        if len(timeout) == 1:
            timeout = timeout[0]
    if isinstance(timeout, (list, tuple)):
        if len(timeout) != 3:
            raise ValueError("timeout list must contain init, query and insert timeouts.")
        return Timeout(init=timeout[0], query=timeout[1], insert=timeout[2])
    if isinstance(timeout, (int, float)):
        return Timeout(query=timeout, insert=timeout)
    raise ValueError(f"Unsupported timeout value: {timeout!r}")


def parse_bool(value: Any) -> bool:
    """
    Parse a boolean from a bool or a string such as 'true', '1', 'yes'.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
