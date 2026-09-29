# built-in dependencies
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface import __version__
from deepface.modules.database import weaviate as weaviate_module
from deepface.modules.database.weaviate import WeaviateClient, resolve_connection_details

weaviate = pytest.importorskip("weaviate")

# pylint: disable=redefined-outer-name, unused-argument, protected-access

INTEGRATION = {"X-Weaviate-Client-Integration": f"deepface/{__version__}"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in [
        "DEEPFACE_WEAVIATE_URI",
        "DEEPFACE_WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "DEEPFACE_WEAVIATE_GRPC_PORT",
        "DEEPFACE_WEAVIATE_SKIP_INIT_CHECKS",
        "DEEPFACE_WEAVIATE_TIMEOUT",
        "WEAVIATE_HNSW_M",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def connectors(monkeypatch):
    mocks = {
        "custom": MagicMock(name="connect_to_custom"),
        "cloud": MagicMock(name="connect_to_weaviate_cloud"),
        "local": MagicMock(name="connect_to_local"),
    }
    monkeypatch.setattr(weaviate, "connect_to_custom", mocks["custom"])
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", mocks["cloud"])
    monkeypatch.setattr(weaviate, "connect_to_local", mocks["local"])
    return mocks


def test_url_string_connects_to_custom(connectors):
    WeaviateClient("http://weaviate.internal:8081")
    kwargs = connectors["custom"].call_args.kwargs
    assert kwargs["http_host"] == "weaviate.internal"
    assert kwargs["http_port"] == 8081
    assert kwargs["http_secure"] is False
    assert kwargs["grpc_host"] == "weaviate.internal"
    assert kwargs["grpc_port"] == 50051
    assert kwargs["grpc_secure"] is False
    assert kwargs["headers"] == INTEGRATION
    assert kwargs["auth_credentials"] is None
    assert kwargs["additional_config"] is None
    assert kwargs["skip_init_checks"] is False


def test_https_url_defaults_to_port_443_and_secure_grpc(connectors):
    WeaviateClient("https://weaviate.example.com")
    kwargs = connectors["custom"].call_args.kwargs
    assert kwargs["http_port"] == 443
    assert kwargs["http_secure"] is True
    assert kwargs["grpc_secure"] is True


@pytest.mark.parametrize(
    "url, port",
    [
        ("http://weaviate.internal", 80),
        ("https://weaviate.internal", 443),
        ("weaviate.internal", 8080),
        ("http://weaviate.internal:8081", 8081),
        ("https://weaviate.internal:8443", 8443),
    ],
)
def test_url_default_ports(connectors, url, port):
    WeaviateClient(url)
    assert connectors["custom"].call_args.kwargs["http_port"] == port


def test_url_without_scheme(connectors):
    WeaviateClient("localhost:8080")
    kwargs = connectors["custom"].call_args.kwargs
    assert (kwargs["http_host"], kwargs["http_port"], kwargs["http_secure"]) == (
        "localhost",
        8080,
        False,
    )


def test_custom_overrides(connectors):
    WeaviateClient(
        {
            "url": "https://api.example.com",
            "grpc_host": "grpc.example.com",
            "grpc_port": 443,
            "grpc_secure": True,
            "http_port": 8443,
            "skip_init_checks": True,
            "headers": {"X-Proxy-Token": "abc"},
        }
    )
    kwargs = connectors["custom"].call_args.kwargs
    assert kwargs["http_host"] == "api.example.com"
    assert kwargs["http_port"] == 8443
    assert kwargs["grpc_host"] == "grpc.example.com"
    assert kwargs["grpc_port"] == 443
    assert kwargs["grpc_secure"] is True
    assert kwargs["skip_init_checks"] is True
    assert kwargs["headers"] == {"X-Proxy-Token": "abc", **INTEGRATION}


def test_http_host_without_url(connectors):
    WeaviateClient({"http_host": "10.0.0.5", "http_port": 9000, "grpc_port": 9001})
    kwargs = connectors["custom"].call_args.kwargs
    assert (kwargs["http_host"], kwargs["http_port"]) == ("10.0.0.5", 9000)
    assert (kwargs["grpc_host"], kwargs["grpc_port"]) == ("10.0.0.5", 9001)


def test_user_can_override_integration_header(connectors):
    WeaviateClient({"url": "http://localhost:8080", "headers": {**INTEGRATION, "a": "b"}})
    assert connectors["custom"].call_args.kwargs["headers"] == {**INTEGRATION, "a": "b"}
    WeaviateClient(
        {"url": "http://localhost:8080", "headers": {"X-Weaviate-Client-Integration": "my-app/1"}}
    )
    headers = connectors["custom"].call_args.kwargs["headers"]
    assert headers == {"X-Weaviate-Client-Integration": "my-app/1"}


@pytest.mark.parametrize(
    "url",
    ["https://abc123.c0.europe-west3.gcp.weaviate.cloud", "abc123.weaviate.network"],
)
def test_cloud_url_connects_to_cloud(connectors, url):
    WeaviateClient({"url": url, "api_key": "secret"})
    connectors["custom"].assert_not_called()
    kwargs = connectors["cloud"].call_args.kwargs
    assert kwargs["cluster_url"] == url
    assert kwargs["auth_credentials"] == weaviate.classes.init.Auth.api_key("secret")
    assert kwargs["headers"] == INTEGRATION


def test_cloud_requires_credentials(connectors):
    with pytest.raises(ValueError, match="requires an API key"):
        WeaviateClient("https://abc123.weaviate.cloud")


def test_explicit_cloud_deployment_on_custom_domain(connectors):
    WeaviateClient({"url": "https://vectors.mycompany.com", "deployment": "cloud", "api_key": "k"})
    assert connectors["cloud"].called


def test_local_deployment(connectors):
    WeaviateClient({"deployment": "local", "http_port": 8099, "grpc_port": 50099})
    kwargs = connectors["local"].call_args.kwargs
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 8099
    assert kwargs["grpc_port"] == 50099
    assert kwargs["headers"] == INTEGRATION


def test_embedded_is_not_supported(connectors):
    with pytest.raises(ValueError, match="Embedded Weaviate is not supported"):
        WeaviateClient({"deployment": "embedded"})


def test_unknown_deployment(connectors):
    with pytest.raises(ValueError, match="deployment must be one of"):
        WeaviateClient({"url": "http://localhost:8080", "deployment": "serverless"})


def test_unknown_option_is_rejected(connectors):
    with pytest.raises(ValueError, match="Unknown Weaviate connection option"):
        WeaviateClient({"url": "http://localhost:8080", "grpcport": 50051})


def test_missing_url(connectors):
    with pytest.raises(ValueError, match="URL not provided"):
        WeaviateClient({"api_key": "k"})
    with pytest.raises(ValueError, match="Cloud requires the cluster url"):
        WeaviateClient({"deployment": "cloud", "http_host": "abc.weaviate.cloud", "api_key": "k"})
    connectors["cloud"].assert_not_called()
    with pytest.raises(ValueError, match="connection details not provided"):
        WeaviateClient()


def test_json_string(connectors):
    WeaviateClient(json.dumps({"url": "http://localhost:18080", "grpc_port": 15051}))
    kwargs = connectors["custom"].call_args.kwargs
    assert (kwargs["http_port"], kwargs["grpc_port"]) == (18080, 15051)


def test_json_string_must_be_object():
    with pytest.raises(ValueError):
        resolve_connection_details("{not json")


def test_documented_env_var_takes_precedence(connectors, monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_URI", "http://documented:8080")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_URL", "http://legacy:8080")
    WeaviateClient()
    assert connectors["custom"].call_args.kwargs["http_host"] == "documented"


def test_legacy_env_var_still_works(connectors, monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_URL", "http://legacy:8080")
    WeaviateClient()
    assert connectors["custom"].call_args.kwargs["http_host"] == "legacy"


def test_scalar_env_fallbacks(connectors, monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_URI", "http://localhost:8080")
    monkeypatch.setenv("WEAVIATE_API_KEY", "env-key")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_GRPC_PORT", "15051")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_SKIP_INIT_CHECKS", "true")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_TIMEOUT", "5,60,120")
    WeaviateClient()
    kwargs = connectors["custom"].call_args.kwargs
    assert kwargs["grpc_port"] == 15051
    assert kwargs["skip_init_checks"] is True
    assert kwargs["auth_credentials"] == weaviate.classes.init.Auth.api_key("env-key")
    timeout = kwargs["additional_config"].timeout
    assert (timeout.init, timeout.query, timeout.insert) == (5, 60, 120)


def test_explicit_options_win_over_env(connectors, monkeypatch):
    monkeypatch.setenv("WEAVIATE_API_KEY", "env-key")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_GRPC_PORT", "15051")
    WeaviateClient({"url": "http://localhost:8080", "api_key": "explicit", "grpc_port": 1})
    kwargs = connectors["custom"].call_args.kwargs
    assert kwargs["grpc_port"] == 1
    assert kwargs["auth_credentials"] == weaviate.classes.init.Auth.api_key("explicit")


@pytest.mark.parametrize(
    "auth, expected",
    [
        ({"api_key": "k"}, lambda Auth: Auth.api_key("k")),
        (
            {"access_token": "at", "expires_in": 120, "refresh_token": "rt"},
            lambda Auth: Auth.bearer_token(access_token="at", expires_in=120, refresh_token="rt"),
        ),
        (
            {"client_secret": "cs", "scope": ["openid"]},
            lambda Auth: Auth.client_credentials(client_secret="cs", scope=["openid"]),
        ),
        (
            {"username": "u", "password": "p", "scope": "offline_access"},
            lambda Auth: Auth.client_password(username="u", password="p", scope="offline_access"),
        ),
    ],
)
def test_auth_variants(connectors, auth, expected):
    WeaviateClient({"url": "http://localhost:8080", "auth": auth})
    Auth = weaviate.classes.init.Auth
    assert connectors["custom"].call_args.kwargs["auth_credentials"] == expected(Auth)


def test_auth_object_is_passed_through(connectors):
    credentials = weaviate.classes.init.Auth.client_password(username="u", password="p")
    WeaviateClient({"url": "http://localhost:8080", "auth": credentials})
    assert connectors["custom"].call_args.kwargs["auth_credentials"] is credentials


def test_invalid_auth(connectors):
    with pytest.raises(ValueError, match="auth must contain"):
        WeaviateClient({"url": "http://localhost:8080", "auth": {"token": "x"}})


@pytest.mark.parametrize(
    "timeout, expected",
    [
        (45, (2, 45, 45)),
        ([3, 40, 100], (3, 40, 100)),
        ({"init": 4, "query": 50}, (4, 50, 90)),
        ("12", (2, 12, 12)),
        ("1,2,3", (1, 2, 3)),
    ],
)
def test_timeout_forms(connectors, timeout, expected):
    WeaviateClient({"url": "http://localhost:8080", "timeout": timeout})
    t = connectors["custom"].call_args.kwargs["additional_config"].timeout
    assert (t.init, t.query, t.insert) == expected


def test_invalid_timeout(connectors):
    with pytest.raises(ValueError, match="init, query and insert"):
        WeaviateClient({"url": "http://localhost:8080", "timeout": [1, 2]})


def test_additional_config_options(connectors):
    WeaviateClient(
        {
            "url": "http://localhost:8080",
            "proxies": "http://proxy:3128",
            "trust_env": "true",
            "connection_config": {"session_pool_connections": 7},
        }
    )
    config = connectors["custom"].call_args.kwargs["additional_config"]
    assert config.proxies == "http://proxy:3128"
    assert config.trust_env is True
    assert config.connection.session_pool_connections == 7


def test_ready_additional_config_is_passed_through(connectors):
    config = weaviate.classes.init.AdditionalConfig(trust_env=True)
    WeaviateClient({"url": "http://localhost:8080", "additional_config": config})
    assert connectors["custom"].call_args.kwargs["additional_config"] is config


@pytest.mark.parametrize(
    "option",
    [
        {"timeout": 30},
        {"proxies": "http://proxy:3128"},
        {"trust_env": False},
        {"connection_config": {"session_pool_connections": 7}},
        {"grpc_config": {"channel_options": [("grpc.keepalive_time_ms", 10000)]}},
    ],
)
def test_ready_additional_config_rejects_options_it_replaces(connectors, option):
    config = weaviate.classes.init.AdditionalConfig(trust_env=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        WeaviateClient({"url": "http://localhost:8080", "additional_config": config, **option})
    connectors["custom"].assert_not_called()


def test_ready_additional_config_ignores_env_timeout(connectors, monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_TIMEOUT", "30")
    config = weaviate.classes.init.AdditionalConfig(trust_env=True)
    WeaviateClient({"url": "http://localhost:8080", "additional_config": config})
    assert connectors["custom"].call_args.kwargs["additional_config"] is config


def test_v4_connection_is_accepted(connectors):
    connection = MagicMock(spec=weaviate.WeaviateClient)
    client = WeaviateClient(connection=connection)
    assert client.client is connection
    connectors["custom"].assert_not_called()


def test_v3_connection_is_rejected(connectors):
    v3_client = SimpleNamespace(schema=object(), query=object())
    with pytest.raises(ValueError, match="weaviate.Client from v3 is no longer supported"):
        WeaviateClient(connection=v3_client)


def build_client(
    existing_hashes=(),
    failed_objects=(),
    connection_details=None,
    multi_tenancy=False,
    quantizer=None,
    server_version="1.39.0",
    tenant_exists=True,
):
    connection = MagicMock(spec=weaviate.WeaviateClient)
    connection.collections = MagicMock()
    connection.collections.exists.return_value = True
    connection.get_meta.return_value = {"version": server_version}
    base_collection = connection.collections.use.return_value
    base_collection.config.get.return_value = SimpleNamespace(
        multi_tenancy_config=SimpleNamespace(enabled=multi_tenancy),
        vector_index_config=SimpleNamespace(quantizer=quantizer),
    )
    # with_tenant returns the same mock, so assertions work with or without a tenant
    base_collection.with_tenant.return_value = base_collection
    base_collection.tenants.exists.return_value = tenant_exists
    collection = base_collection

    def fetch_objects(filters, return_properties, limit):
        objects = [
            SimpleNamespace(properties={"embedding_hash": h}) for h in existing_hashes
        ][:limit]
        return SimpleNamespace(objects=objects)

    collection.query.fetch_objects.side_effect = fetch_objects
    collection.batch.failed_objects = list(failed_objects)
    batcher = collection.batch.fixed_size.return_value.__enter__.return_value
    client = WeaviateClient(connection_details=connection_details, connection=connection)
    return client, collection, batcher


def embedding_record(value):
    return {
        "img_name": f"img{value}.jpg",
        "face": np.full((2, 2, 3), value, dtype=np.float64),
        "model_name": "Facenet",
        "detector_backend": "opencv",
        "aligned": True,
        "l2_normalized": False,
        "embedding": [float(value), 1.0],
    }


def embedding_hash(value):
    import hashlib
    import struct

    return hashlib.sha256(struct.pack("2d", float(value), 1.0)).hexdigest()


def test_insert_skips_existing_and_duplicate_embeddings():
    client, collection, batcher = build_client(existing_hashes=[embedding_hash(1)])
    records = [embedding_record(1), embedding_record(2), embedding_record(2), embedding_record(3)]

    inserted = client.insert_embeddings(records, batch_size=50)

    assert inserted == 2
    collection.batch.fixed_size.assert_called_once_with(batch_size=50)
    added = [call.kwargs for call in batcher.add_object.call_args_list]
    assert [a["properties"]["img_name"] for a in added] == ["img2.jpg", "img3.jpg"]
    assert added[0]["vector"] == [2.0, 1.0]
    assert added[0]["properties"]["embedding_hash"] == embedding_hash(2)
    face = np.frombuffer(base64.b64decode(added[0]["properties"]["face"]), dtype=np.float64)
    assert np.array_equal(face.reshape(added[0]["properties"]["face_shape"]), np.full((2, 2, 3), 2))
    # deterministic ids make retries idempotent
    client_again, _, batcher_again = build_client()
    client_again.insert_embeddings([embedding_record(2)])
    assert batcher_again.add_object.call_args.kwargs["uuid"] == added[0]["uuid"]


def test_insert_returns_zero_when_everything_exists():
    client, _, batcher = build_client(existing_hashes=[embedding_hash(1)])
    assert client.insert_embeddings([embedding_record(1)]) == 0
    batcher.add_object.assert_not_called()


def test_insert_raises_on_failed_objects():
    failed = [SimpleNamespace(message="vector lengths don't match")]
    client, _, _ = build_client(failed_objects=failed)
    with pytest.raises(ValueError, match="vector lengths don't match"):
        client.insert_embeddings([embedding_record(1)])


def test_hnsw_m_env_maps_to_max_connections(monkeypatch):
    monkeypatch.setenv("WEAVIATE_HNSW_M", "48")
    client, _, _ = build_client()
    client.client.collections.exists.return_value = False
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    schema = client.client.collections.create_from_dict.call_args.args[0]
    assert schema["class"] == "Embeddings_facenet_opencv_aligned_raw"
    assert schema["vectorIndexConfig"] == {"distance": "l2-squared", "maxConnections": 48}


@pytest.mark.parametrize("identity_id", [17, "17", "not-a-uuid"])
def test_fetch_embedding_with_non_uuid_id_returns_none(identity_id):
    client, collection, _ = build_client()
    assert client.fetch_embedding(identity_id, model_name="Facenet") is None
    collection.query.fetch_object_by_id.assert_not_called()


def test_api_key_and_auth_are_exclusive(connectors):
    with pytest.raises(ValueError, match="either api_key or auth"):
        WeaviateClient({"url": "http://localhost:8080", "api_key": "k", "auth": {"api_key": "k"}})


def test_env_api_key_does_not_conflict_with_auth(connectors, monkeypatch):
    monkeypatch.setenv("WEAVIATE_API_KEY", "env-key")
    WeaviateClient({"url": "http://localhost:8080", "auth": {"access_token": "at"}})
    credentials = connectors["custom"].call_args.kwargs["auth_credentials"]
    assert credentials == weaviate.classes.init.Auth.bearer_token(access_token="at")


@pytest.mark.parametrize(
    "url, warned",
    [
        ("http://weaviate.internal:8080", True),
        ("https://weaviate.internal", False),
        ("http://localhost:8080", False),
        ("http://127.0.0.1:8080", False),
        ({"url": "https://weaviate.internal", "grpc_secure": False}, True),
        ({"url": "https://w.internal", "grpc_host": "grpc.internal", "grpc_secure": 0}, True),
        ({"url": "https://w.internal", "grpc_host": "localhost", "grpc_secure": False}, False),
        ({"url": "http://localhost:8080", "grpc_host": "grpc.internal"}, True),
        ({"deployment": "local", "http_host": "weaviate.internal"}, True),
    ],
)
def test_credentials_over_plain_http_warn(connectors, monkeypatch, url, warned):
    warnings = []
    monkeypatch.setattr(weaviate_module.logger, "warn", warnings.append)
    details = url if isinstance(url, dict) else {"url": url}
    WeaviateClient({**details, "api_key": "s3cr3t-value"})
    assert bool(warnings) is warned
    # the warning names the host, never the credentials
    assert all("s3cr3t-value" not in w for w in warnings)


def test_no_warning_without_credentials(connectors, monkeypatch):
    warnings = []
    monkeypatch.setattr(weaviate_module.logger, "warn", warnings.append)
    WeaviateClient("http://weaviate.internal:8080")
    assert not warnings


def test_insecure_grpc_warning_names_the_grpc_host(connectors, monkeypatch):
    warnings = []
    monkeypatch.setattr(weaviate_module.logger, "warn", warnings.append)
    WeaviateClient(
        {
            "url": "https://weaviate.internal",
            "grpc_host": "grpc.internal",
            "grpc_secure": False,
            "api_key": "k",
        }
    )
    assert len(warnings) == 1
    assert "gRPC to grpc.internal" in warnings[0] and "REST" not in warnings[0]


def test_grpc_config_uses_grpc_config_class(connectors, monkeypatch):
    grpc_config_class = MagicMock(name="GrpcConfig")
    monkeypatch.setattr(weaviate_module, "load_grpc_config_class", lambda: grpc_config_class)
    monkeypatch.setattr(weaviate.classes.init, "AdditionalConfig", MagicMock())
    WeaviateClient(
        {
            "url": "http://localhost:8080",
            "grpc_config": {"channel_options": [["grpc.keepalive_time_ms", 10000]]},
        }
    )
    grpc_config_class.assert_called_once_with(
        channel_options=[("grpc.keepalive_time_ms", 10000)]
    )
    connectors["custom"].assert_called_once()


def test_unknown_grpc_config_key_is_rejected(connectors):
    with pytest.raises(ValueError, match="Unknown grpc_config option"):
        WeaviateClient({"url": "http://localhost:8080", "grpc_config": {"keepalive": 1}})
    with pytest.raises(ValueError, match="grpc_config must be a dict"):
        WeaviateClient({"url": "http://localhost:8080", "grpc_config": [("a", 1)]})


@pytest.fixture
def legacy_client(monkeypatch, connectors):
    """
    Simulate weaviate-client before 4.20: no GrpcConfig, so deepface builds the client.
    """
    monkeypatch.setattr(weaviate_module, "load_grpc_config_class", lambda: None)
    client_class = MagicMock(name="WeaviateClient")
    monkeypatch.setattr(weaviate, "WeaviateClient", client_class)
    return client_class


def endpoints(params):
    return (
        (params.http.host, params.http.port, params.http.secure),
        (params.grpc.host, params.grpc.port, params.grpc.secure),
    )


@pytest.mark.parametrize(
    "details, http, grpc_endpoint",
    [
        (
            {"url": "http://weaviate.internal:8081", "grpc_port": 50052},
            ("weaviate.internal", 8081, False),
            ("weaviate.internal", 50052, False),
        ),
        (
            {"url": "https://api.example.com", "grpc_host": "grpc.example.com", "grpc_port": 443},
            ("api.example.com", 443, True),
            ("grpc.example.com", 443, True),
        ),
        (
            {"deployment": "local", "http_port": 8099, "grpc_port": 50099},
            ("localhost", 8099, False),
            ("localhost", 50099, False),
        ),
        (
            {"url": "https://abc.c0.europe-west3.gcp.weaviate.cloud", "api_key": "k"},
            ("abc.c0.europe-west3.gcp.weaviate.cloud", 443, True),
            ("grpc-abc.c0.europe-west3.gcp.weaviate.cloud", 443, True),
        ),
        (
            {"url": "abc.weaviate.network", "api_key": "k"},
            ("abc.weaviate.network", 443, True),
            ("abc.grpc.weaviate.network", 443, True),
        ),
    ],
)
def test_legacy_grpc_config_builds_connection_params(
    legacy_client, connectors, details, http, grpc_endpoint
):
    WeaviateClient({**details, "grpc_config": {"channel_options": []}})
    kwargs = legacy_client.call_args.kwargs
    assert endpoints(kwargs["connection_params"]) == (http, grpc_endpoint)
    assert kwargs["additional_headers"] == INTEGRATION
    assert kwargs["skip_init_checks"] is False
    legacy_client.return_value.connect.assert_called_once()
    for connector in connectors.values():
        connector.assert_not_called()


def test_legacy_grpc_config_passes_common_options(legacy_client):
    WeaviateClient(
        {
            "url": "https://weaviate.internal",
            "api_key": "k",
            "skip_init_checks": True,
            "timeout": 30,
            "grpc_config": {},
        }
    )
    kwargs = legacy_client.call_args.kwargs
    assert kwargs["auth_client_secret"] == weaviate.classes.init.Auth.api_key("k")
    assert kwargs["skip_init_checks"] is True
    assert kwargs["additional_config"].timeout.query == 30


def test_legacy_grpc_config_closes_client_on_connect_error(legacy_client):
    legacy_client.return_value.connect.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        WeaviateClient({"url": "http://localhost:8080", "grpc_config": {}})
    legacy_client.return_value.close.assert_called_once()


@pytest.mark.parametrize("secure", [True, False])
def test_legacy_grpc_channel_applies_options_and_credentials(monkeypatch, secure):
    import grpc  # pylint: disable=import-outside-toplevel

    secure_channel, insecure_channel = MagicMock(), MagicMock()
    monkeypatch.setattr(grpc, "secure_channel", secure_channel)
    monkeypatch.setattr(grpc, "insecure_channel", insecure_channel)
    credentials = object()

    params = weaviate_module.legacy_grpc_connection_params_class().from_grpc_config(
        ("w.internal", 443, secure),
        ("g.internal", 50051, secure),
        {"channel_options": [("grpc.keepalive_time_ms", 10000)], "credentials": credentials},
    )
    params._grpc_channel(proxies={"grpc": "http://proxy:3128"}, grpc_msg_size=1024, is_async=False)

    channel = secure_channel if secure else insecure_channel
    kwargs = channel.call_args.kwargs
    assert kwargs["target"] == "g.internal:50051"
    assert kwargs["options"] == [
        ("grpc.max_send_message_length", 1024),
        ("grpc.max_receive_message_length", 1024),
        ("grpc.default_authority", "g.internal"),
        ("grpc.http_proxy", "http://proxy:3128"),
        ("grpc.keepalive_time_ms", 10000),
    ]
    if secure:
        assert kwargs["credentials"] is credentials
        insecure_channel.assert_not_called()
    else:
        secure_channel.assert_not_called()


def test_legacy_grpc_channel_defaults_to_system_tls(monkeypatch):
    import grpc  # pylint: disable=import-outside-toplevel

    secure_channel = MagicMock()
    default_credentials = object()
    monkeypatch.setattr(grpc, "secure_channel", secure_channel)
    monkeypatch.setattr(grpc, "ssl_channel_credentials", lambda: default_credentials)
    params = weaviate_module.legacy_grpc_connection_params_class().from_grpc_config(
        ("w.internal", 443, True), ("g.internal", 443, True), {}
    )
    params._grpc_channel(proxies={}, grpc_msg_size=None, is_async=False)
    assert secure_channel.call_args.kwargs["credentials"] is default_credentials


def test_legacy_channel_options_defaults_then_grpc_config():
    params = weaviate_module.legacy_grpc_connection_params_class().from_grpc_config(
        ("w.internal", 8080, False),
        ("g.internal", 50051, False),
        {"channel_options": [("grpc.keepalive_time_ms", 10000)]},
    )
    options = params.channel_options(proxies={}, grpc_msg_size=None)
    assert options[:3] == [
        ("grpc.max_send_message_length", weaviate.connect.base.MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", weaviate.connect.base.MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.default_authority", "g.internal"),
    ]
    assert options[3:] == [("grpc.keepalive_time_ms", 10000)]


def test_tenant_and_quantization_from_connection_details(connectors):
    client = WeaviateClient(
        {"url": "http://localhost:8080", "tenant": "acme", "quantization": "RQ"}
    )
    assert (client.tenant, client.quantization) == ("acme", "rq")
    assert connectors["custom"].called


def test_tenant_and_quantization_from_env(connectors, monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_TENANT", "site-7")
    monkeypatch.setenv("DEEPFACE_WEAVIATE_QUANTIZATION", "bq")
    client = WeaviateClient("http://localhost:8080")
    assert (client.tenant, client.quantization) == ("site-7", "bq")


def test_tenant_with_existing_connection():
    client, _, _ = build_client(connection_details={"tenant": "acme"}, multi_tenancy=True)
    assert client.tenant == "acme"


@pytest.mark.parametrize("tenant", ["has space", "a" * 65, "semi;colon", "ümlaut"])
def test_invalid_tenant(connectors, tenant):
    with pytest.raises(ValueError, match="Invalid Weaviate tenant name"):
        WeaviateClient({"url": "http://localhost:8080", "tenant": tenant})


def test_invalid_quantization(connectors):
    with pytest.raises(ValueError, match="Unsupported Weaviate quantization"):
        WeaviateClient({"url": "http://localhost:8080", "quantization": "lvq"})


def test_default_creates_collection_without_tenancy_or_quantizer():
    client, _, _ = build_client()
    client.client.collections.exists.return_value = False
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    schema = client.client.collections.create_from_dict.call_args.args[0]
    assert "multiTenancyConfig" not in schema
    assert schema["vectorIndexConfig"] == {"distance": "l2-squared"}
    client.client.get_meta.assert_not_called()


@pytest.mark.parametrize(
    "quantization, expected",
    [
        ("rq", {"rq": {"enabled": True, "bits": 8}}),
        ("rq-8", {"rq": {"enabled": True, "bits": 8}}),
        ("rq-1", {"rq": {"enabled": True, "bits": 1}}),
        ("bq", {"bq": {"enabled": True}}),
        ("sq", {"sq": {"enabled": True}}),
        ("pq", {"pq": {"enabled": True}}),
        ("none", {"skipDefaultQuantization": True}),
    ],
)
def test_quantization_is_set_on_create(quantization, expected):
    client, _, _ = build_client(connection_details={"quantization": quantization})
    client.client.collections.exists.return_value = False
    client.initialize_database(model_name="Facenet", detector_backend="opencv", l2_normalized=True)
    schema = client.client.collections.create_from_dict.call_args.args[0]
    assert schema["vectorIndexConfig"] == {"distance": "cosine", **expected}


@pytest.mark.parametrize(
    "quantization, server_version", [("rq", "1.31.4"), ("rq-1", "1.32.9"), ("sq", "1.25.0")]
)
def test_quantization_requires_server_version(quantization, server_version):
    client, _, _ = build_client(
        connection_details={"quantization": quantization}, server_version=server_version
    )
    client.client.collections.exists.return_value = False
    with pytest.raises(ValueError, match="requires Weaviate"):
        client.initialize_database(model_name="Facenet", detector_backend="opencv")
    client.client.collections.create_from_dict.assert_not_called()


def test_existing_collection_with_other_quantizer_warns(monkeypatch):
    warnings = []
    monkeypatch.setattr(weaviate_module.logger, "warn", warnings.append)
    bq = type("_BQConfig", (), {})()
    client, _, _ = build_client(connection_details={"quantization": "rq"}, quantizer=bq)
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    client.client.collections.create_from_dict.assert_not_called()
    assert len(warnings) == 1 and "'bq'" in warnings[0]


def test_existing_collection_with_same_quantizer_does_not_warn(monkeypatch):
    warnings = []
    monkeypatch.setattr(weaviate_module.logger, "warn", warnings.append)
    rq = type("_RQConfig", (), {"bits": 8})()
    client, _, _ = build_client(connection_details={"quantization": "rq-8"}, quantizer=rq)
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    assert not warnings


def test_tenant_creates_multi_tenant_collection():
    client, _, _ = build_client(connection_details={"tenant": "acme"})
    client.client.collections.exists.return_value = False
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    schema = client.client.collections.create_from_dict.call_args.args[0]
    assert schema["multiTenancyConfig"] == {
        "enabled": True,
        "autoTenantCreation": True,
        "autoTenantActivation": True,
    }


def test_tenancy_requires_server_version():
    client, _, _ = build_client(connection_details={"tenant": "acme"}, server_version="1.24.9")
    client.client.collections.exists.return_value = False
    with pytest.raises(ValueError, match="multi-tenancy requires Weaviate 1.25"):
        client.initialize_database(model_name="Facenet", detector_backend="opencv")


def test_tenant_on_single_tenant_collection_is_rejected():
    client, _, _ = build_client(connection_details={"tenant": "acme"}, multi_tenancy=False)
    with pytest.raises(ValueError, match="created without multi-tenancy"):
        client.initialize_database(model_name="Facenet", detector_backend="opencv")


def test_multi_tenant_collection_requires_tenant():
    client, _, _ = build_client(multi_tenancy=True)
    with pytest.raises(ValueError, match="a tenant is required"):
        client.initialize_database(model_name="Facenet", detector_backend="opencv")


def test_operations_are_scoped_to_tenant():
    client, collection, batcher = build_client(
        connection_details={"tenant": "acme"}, multi_tenancy=True
    )
    client.insert_embeddings([embedding_record(1)])
    client.search_by_vector([0.0, 1.0], model_name="Facenet", detector_backend="opencv")
    client.fetch_all_embeddings("Facenet", "opencv", True, False)
    client.fetch_embedding("00000000-0000-0000-0000-000000000001", "Facenet", "opencv")
    assert collection.with_tenant.call_count == 4
    assert {c.args for c in collection.with_tenant.call_args_list} == {("acme",)}
    assert batcher.add_object.call_count == 1


def test_new_tenant_skips_existing_check_and_reads_return_empty():
    client, collection, batcher = build_client(
        connection_details={"tenant": "new"}, multi_tenancy=True, tenant_exists=False
    )
    assert client.insert_embeddings([embedding_record(1)]) == 1
    collection.query.fetch_objects.assert_not_called()
    assert batcher.add_object.call_count == 1
    assert not client.search_by_vector([0.0], model_name="Facenet", detector_backend="opencv")
    assert not client.fetch_all_embeddings("Facenet", "opencv", True, False)
    assert client.fetch_embedding("00000000-0000-0000-0000-000000000001", "Facenet") is None
    collection.query.near_vector.assert_not_called()


def test_tenant_and_quantization_from_env_uri_json(connectors, monkeypatch):
    monkeypatch.setenv(
        "DEEPFACE_WEAVIATE_URI",
        json.dumps({"url": "http://localhost:8080", "tenant": "acme", "quantization": "bq"}),
    )
    client = WeaviateClient()
    assert (client.tenant, client.quantization) == ("acme", "bq")


def test_tenant_from_env_uri_json_with_existing_connection(monkeypatch):
    monkeypatch.setenv("DEEPFACE_WEAVIATE_URI", json.dumps({"url": "x", "tenant": "acme"}))
    client, _, _ = build_client(multi_tenancy=True)
    assert client.tenant == "acme"


def test_invalid_options_do_not_open_a_connection(connectors):
    with pytest.raises(ValueError, match="Invalid Weaviate tenant name"):
        WeaviateClient({"url": "http://localhost:8080", "tenant": "bad name"})
    with pytest.raises(ValueError, match="Unsupported Weaviate quantization"):
        WeaviateClient({"url": "http://localhost:8080", "quantization": "lvq"})
    connectors["custom"].assert_not_called()


@pytest.mark.parametrize("tenant", ["", "   "])
def test_explicit_empty_tenant_is_rejected(connectors, tenant):
    with pytest.raises(ValueError, match="tenant is empty"):
        WeaviateClient({"url": "http://localhost:8080", "tenant": tenant})


def test_ids_are_scoped_to_tenant():
    ids = []
    for details in [None, {"tenant": "acme"}, {"tenant": "globex"}]:
        client, _, batcher = build_client(connection_details=details, multi_tenancy=bool(details))
        client.insert_embeddings([embedding_record(1)])
        ids.append(batcher.add_object.call_args.kwargs["uuid"])
    assert len(set(ids)) == 3


def test_fetch_embedding_checks_tenancy_mode():
    client, _, _ = build_client(multi_tenancy=True)
    with pytest.raises(ValueError, match="a tenant is required"):
        client.fetch_embedding("00000000-0000-0000-0000-000000000001", model_name="Facenet")


def test_quantization_none_skips_version_check():
    client, _, _ = build_client(connection_details={"quantization": "none"}, server_version="")
    client.client.collections.exists.return_value = False
    client.initialize_database(model_name="Facenet", detector_backend="opencv")
    client.client.get_meta.assert_not_called()
