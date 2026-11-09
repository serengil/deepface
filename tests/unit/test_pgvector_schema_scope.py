import sys
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from deepface.modules.database import pgvector


class SchemaScopeTests(TestCase):
    def test_each_connection_initializes_its_own_schema_once(self):
        modules = {key: MagicMock() for key in ("psycopg", "pgvector", "pgvector.psycopg")}
        first, second = MagicMock(), MagicMock()
        with patch.dict(sys.modules, modules), \
             patch.object(pgvector, "build_model", return_value=Mock(output_shape=128)):
            for conn in (first, second):
                conn.info.transaction_status = modules["psycopg"].pq.TransactionStatus.IDLE
                client = pgvector.PGVectorClient(connection=conn)
                client.initialize_database()
                client.initialize_database()
        for conn in (first, second):
            queries = [str(call.args[0]) for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list]
            self.assertEqual(sum("CREATE EXTENSION" in q for q in queries), 1)
            self.assertEqual(sum("CREATE TABLE" in q for q in queries), 1)
            self.assertEqual(sum("CREATE INDEX IF NOT EXISTS" in q for q in queries), 1)
            conn.rollback.assert_not_called()
            conn.commit.assert_not_called()
