import unittest

from launcher import LauncherWindow


class LauncherHdcDisconnectTests(unittest.TestCase):
    def _make_window(self, screen_mode: str):
        window = LauncherWindow.__new__(LauncherWindow)
        window.current_plan = {"screen_mode": screen_mode}
        window.current_run_stream_disconnected = False
        window.current_run_stream_started = False
        window.current_run_sp_started = False
        window.marked_disconnects = []
        window._handle_sp_output = lambda text: None

        def mark(message, source):
            window.marked_disconnects.append((message, source))

        window._mark_stream_disconnected = mark
        return window

    def test_hdc_capture_failure_output_does_not_enter_stream_disconnect_recovery(self):
        window = self._make_window("1")

        LauncherWindow._handle_stream_output(
            window,
            "[HDC] Consecutive capture failures exceeded.\n",
        )

        self.assertEqual([], window.marked_disconnects)

    def test_grpc_disconnect_output_still_enters_stream_disconnect_recovery(self):
        window = self._make_window("0")

        LauncherWindow._handle_stream_output(
            window,
            "[Stream] gRPC Error: unavailable\n",
        )

        self.assertEqual(
            [("[Stream] gRPC Error: unavailable", "stdout")],
            window.marked_disconnects,
        )


if __name__ == "__main__":
    unittest.main()
