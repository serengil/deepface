from unittest import TestCase
from unittest.mock import Mock, patch

from deepface.modules.database import pinecone


class ReadOnlyIndexTests(TestCase):
    def test_empty_gallery_does_not_require_provisioning_privileges(self):
        gallery = object.__new__(pinecone.PineconeClient)
        gallery.client = Mock()
        gallery.client.has_index.return_value = False
        with patch.object(pinecone, "build_model", side_effect=AssertionError("model loaded")):
            self.assertEqual(gallery.search_by_vector([1.0]), [])
            self.assertEqual(gallery.fetch_all_embeddings("Facenet", "skip", True, False), [])
        gallery.client.Index.assert_not_called()
        gallery.client.create_index.assert_not_called()
