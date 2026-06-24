import os
import tempfile
import unittest
from pathlib import Path

from aw.autogame.stream_client.stream_client import StreamClient


class StreamClientDisconnectTests(unittest.TestCase):
    def test_disconnect_signal_waits_for_two_failed_reconnects_by_default(self):
        client = StreamClient.__new__(StreamClient)

        self.assertFalse(client._should_signal_stream_disconnect(1))
        self.assertFalse(client._should_signal_stream_disconnect(2))
        self.assertTrue(client._should_signal_stream_disconnect(3))

    def test_disconnect_signal_threshold_can_be_overridden(self):
        client = StreamClient.__new__(StreamClient)
        key = "AUTOGAME_STREAM_DISCONNECT_RETRIES_BEFORE_SIGNAL"
        original = os.environ.get(key)
        os.environ[key] = "3"
        try:
            self.assertFalse(client._should_signal_stream_disconnect(3))
            self.assertTrue(client._should_signal_stream_disconnect(4))
        finally:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def test_disconnect_signal_file_is_written_only_after_threshold(self):
        client = StreamClient.__new__(StreamClient)
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR")
            os.environ["AUTOGAME_RUN_ARCHIVE_DIR"] = tmpdir
            try:
                self.assertFalse(client._maybe_write_disconnect_signal("grpc_error", "first", 1))
                self.assertFalse((Path(tmpdir) / "stream_disconnect_signal.json").exists())

                self.assertFalse(client._maybe_write_disconnect_signal("grpc_error", "second", 2))
                self.assertFalse((Path(tmpdir) / "stream_disconnect_signal.json").exists())

                self.assertTrue(client._maybe_write_disconnect_signal("grpc_error", "third", 3))
                self.assertTrue((Path(tmpdir) / "stream_disconnect_signal.json").exists())
            finally:
                if original is None:
                    os.environ.pop("AUTOGAME_RUN_ARCHIVE_DIR", None)
                else:
                    os.environ["AUTOGAME_RUN_ARCHIVE_DIR"] = original

    def test_transient_message_does_not_match_launcher_disconnect_pattern(self):
        message = StreamClient._build_transient_disconnect_message("gRPC Error", 1, 1.0)

        self.assertIn("Transient", message)
        self.assertNotIn("[Stream] gRPC Error:", message)


if __name__ == "__main__":
    unittest.main()
