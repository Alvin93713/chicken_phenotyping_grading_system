import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inference_utils


class UbuntuCompatibilityTests(unittest.TestCase):
    def test_linux_camera_uses_platform_default_backend(self):
        import cv2

        capture = Mock()
        capture.isOpened.return_value = True
        with patch.object(inference_utils.sys, "platform", "linux"), \
             patch.object(cv2, "VideoCapture", return_value=capture) as video_capture:
            actual = inference_utils.open_camera(1)

        self.assertIs(actual, capture)
        video_capture.assert_called_once_with(1, cv2.CAP_ANY)


if __name__ == "__main__":
    unittest.main()
