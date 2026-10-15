import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.utils.git import read_checkout


class CheckoutOwnershipTests(unittest.TestCase):
    def test_different_owner_is_trusted_only_for_the_selected_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "repo"
            global_config = root / "gitconfig"
            global_config.write_text("")
            env = {"GIT_CONFIG_GLOBAL": str(global_config), "GIT_CONFIG_NOSYSTEM": "1"}
            with patch.dict(os.environ, env):
                subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
                with patch.dict(os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}):
                    rejected = subprocess.run(["git", "-C", str(checkout), "status"],
                                              capture_output=True, text=True)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("dubious ownership", rejected.stderr)
                    self.assertEqual(read_checkout(checkout, "status", "--porcelain"), "")
                    self.assertEqual(read_checkout(checkout, "rev-parse", "--show-toplevel"),
                                     str(checkout.resolve()))
                    # The exception does not persist into a subsequent plain Git call.
                    self.assertNotEqual(subprocess.run(
                        ["git", "-C", str(checkout), "status"], capture_output=True).returncode, 0)
                self.assertEqual(global_config.read_text(), "")


if __name__ == "__main__":
    unittest.main()
