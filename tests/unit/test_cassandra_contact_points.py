from unittest import TestCase

from deepface.modules.database.cassandra import _parse_uri


class ContactPointTests(TestCase):
    def test_ipv6_and_dns_hosts_share_the_explicit_port(self):
        options, keyspace = _parse_uri(
            "cassandra://s%40m:p%3Ass@[2001:db8::1],node2:9142/gallery"
        )
        self.assertEqual(options, {
            "contact_points": ["2001:db8::1", "node2"], "port": 9142,
            "username": "s@m", "password": "p:ss",
        })
        self.assertEqual(keyspace, "gallery")

    def test_conflicting_ports_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same port"):
            _parse_uri("cassandra://one:9142,two:9242/gallery")

    def test_empty_contact_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "host"):
            _parse_uri("cassandra://one,,two/gallery")
