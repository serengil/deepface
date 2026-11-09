from unittest import TestCase
from unittest.mock import Mock, patch

from deepface.api.src.modules.core import routes


class IntegerOptionTests(TestCase):
    def test_invalid_options_return_400_before_decoding_an_image(self):
        for handler, key in ((routes.represent, "max_faces"), (routes.register, "expand_percentage"),
                             (routes.search, "k")):
            for bad in ("no", 1.5, True, []):
                with self.subTest(route=handler.__name__, key=key, value=bad):
                    request = Mock(is_json=True)
                    request.get_json.return_value = {key: bad, "img": "image.jpg"}
                    with patch.object(routes, "request", request), \
                         patch.object(routes.blueprint, "variables", Mock(), create=True), \
                         patch.object(routes.blueprint, "container", Mock(), create=True), \
                         patch.object(routes, "extract_image_from_request") as decode:
                        body, status = handler()
                    self.assertEqual(status, 400)
                    self.assertIn(key, body["exception"])
                    decode.assert_not_called()

    def test_integer_strings_and_optional_nulls(self):
        request = Mock(is_json=True)
        request.get_json.return_value = {"expand_percentage": "10", "k": None}
        with patch.object(routes, "request", request):
            self.assertEqual(routes.read_options("expand_percentage", "k"), {"expand_percentage":10, "k":None})
