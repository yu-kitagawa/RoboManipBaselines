"""Unit tests for verifying command execution in the AutoSuccessRateReport class."""

import unittest
from unittest.mock import patch

from robo_manip_baselines.scripts.AutoSuccessRateReport import AutoSuccessRateReport


class TestAutoSuccessRateReport(unittest.TestCase):
    """Unit test class for testing methods of the AutoSuccessRateReport class."""

    def setUp(self):
        """Initialize necessary setup before each test case."""
        super().setUp()
        self.auto_success_rate_report = AutoSuccessRateReport(
            policy="Act",
            env="MujocoUR5eCable",
            commit_id="0123456789abcdefghijklmnopqrstuvwxyz0123",
        )

    @patch.object(AutoSuccessRateReport, "exec_command")
    def test_exec_command_train(self, mock_exec_command):
        """Test case to check if the exec_command method is called correctly."""

        mock_exec_command.return_value = None

        command = [
            "/tmp/example1/venv/bin/python",
            "/tmp/example2/RoboManipBaselines/robo_manip_baselines/" + "bin/Train.py",
            "Act",
            "MujocoUR5eCable",
            "--dataset_dir",
            "/tmp/example3/dataset/",
            "--checkpoint_dir",
            "/tmp/example2/RoboManipBaselines/robo_manip_baselines/"
            + "checkpoint_dir/Act/MujocoUR5eCable",
        ]
        self.auto_success_rate_report.exec_command(command)  # Call the mocked method

        mock_exec_command.assert_called_once()  # Verify that the method was called exactly once
        mock_exec_command.assert_called_once_with(
            command
        )  # Verify that the method was called with the specified arguments

    def test_check_apt_packages_installed(self):
        package_names = AutoSuccessRateReport.APT_REQUIRED_PACKAGE_NAMES
        print(f"[{self.test_check_apt_packages_installed.__name__}] {package_names=}")
        self.auto_success_rate_report.check_apt_packages_installed(package_names)


if __name__ == "__main__":
    unittest.main()
