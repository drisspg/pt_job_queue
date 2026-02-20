from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ptq.provision import cancel_reservation, create_reservation, ensure_ssh_config


@pytest.fixture()
def mock_gpu_dev():
    """Mock the Config, ReservationManager, and auth returned by _get_config_and_manager."""
    config = MagicMock()
    manager = MagicMock()
    user_info = {"user_id": "testuser", "github_user": "testgh", "arn": "arn:aws:iam::testuser"}
    with (
        patch(
            "ptq.provision._get_config_and_manager",
            return_value=(config, manager),
        ),
        patch("ptq.provision._get_user_info", return_value=user_info),
    ):
        yield manager, user_info


class TestCreateReservation:
    def test_happy_path(self, mock_gpu_dev):
        manager, user_info = mock_gpu_dev
        manager.create_reservation.return_value = "res-123"
        manager.wait_for_reservation_completion.return_value = {
            "pod_name": "gpu-dev-abc12345",
            "ssh_command": "ssh dev@eager-raccoon.devservers.io",
            "fqdn": "eager-raccoon.devservers.io",
            "status": "active",
        }

        res_id, pod_name, conn_info = create_reservation(
            gpu_type="h100", duration_hours=2.0
        )

        assert res_id == "res-123"
        assert pod_name == "gpu-dev-abc12345"
        assert conn_info["status"] == "active"
        manager.create_reservation.assert_called_once()
        call_kwargs = manager.create_reservation.call_args[1]
        assert call_kwargs["user_id"] == "testuser"
        assert call_kwargs["github_user"] == "testgh"
        assert call_kwargs["gpu_type"] == "h100"
        assert call_kwargs["no_persistent_disk"] is True
        manager.wait_for_reservation_completion.assert_called_once_with("res-123")

    def test_passes_dockerfile_and_name(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.return_value = "res-456"
        manager.wait_for_reservation_completion.return_value = {
            "pod_name": "gpu-dev-xyz",
        }

        create_reservation(dockerfile="base64data==", name="my-job")

        kwargs = manager.create_reservation.call_args[1]
        assert kwargs["dockerfile"] == "base64data=="
        assert kwargs["name"] == "my-job"

    def test_create_returns_none_raises(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.return_value = None
        with pytest.raises(RuntimeError, match="no ID returned"):
            create_reservation()

    def test_wait_returns_none_raises(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.return_value = "res-789"
        manager.wait_for_reservation_completion.return_value = None
        with pytest.raises(RuntimeError, match="timed out"):
            create_reservation()

    def test_create_failure_propagates(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            create_reservation()

    def test_wait_failure_propagates(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.return_value = "res-789"
        manager.wait_for_reservation_completion.side_effect = TimeoutError(
            "Pod never became active"
        )
        with pytest.raises(TimeoutError):
            create_reservation()

    def test_empty_pod_name_fallback(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.create_reservation.return_value = "res-x"
        manager.wait_for_reservation_completion.return_value = {
            "ssh_command": "ssh dev@host.devservers.io",
        }

        _, pod_name, _ = create_reservation()
        assert pod_name == ""


class TestCancelReservation:
    def test_success(self, mock_gpu_dev):
        manager, user_info = mock_gpu_dev
        manager.cancel_reservation.return_value = True
        assert cancel_reservation("res-123") is True
        manager.cancel_reservation.assert_called_once_with("res-123", "testuser")

    def test_idempotent_on_error(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.cancel_reservation.side_effect = Exception("already cancelled")
        assert cancel_reservation("res-123") is False

    def test_idempotent_on_any_exception(self, mock_gpu_dev):
        manager, _ = mock_gpu_dev
        manager.cancel_reservation.side_effect = RuntimeError("network error")
        assert cancel_reservation("res-999") is False


class TestImportError:
    def test_missing_gpu_dev_cli(self):
        with (
            patch.dict(
                "sys.modules",
                {
                    "gpu_dev_cli": None,
                    "gpu_dev_cli.config": None,
                    "gpu_dev_cli.reservations": None,
                },
            ),
            patch(
                "ptq.provision._get_config_and_manager",
                side_effect=SystemExit("gpu-dev-cli is not installed"),
            ),
            pytest.raises(SystemExit, match="gpu-dev-cli is not installed"),
        ):
            create_reservation()


class TestEnsureSshConfig:
    def test_config_with_include(self, tmp_path):
        ssh_config = tmp_path / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        ssh_config.write_text("Host *\n  ServerAliveInterval 60\n\nInclude ~/.gpu-dev/*\n")
        with patch("ptq.provision.Path.home", return_value=tmp_path):
            assert ensure_ssh_config() is True

    def test_config_without_include(self, tmp_path):
        ssh_config = tmp_path / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        ssh_config.write_text("Host *\n  ServerAliveInterval 60\n")
        with patch("ptq.provision.Path.home", return_value=tmp_path):
            assert ensure_ssh_config() is False

    def test_no_ssh_config(self, tmp_path):
        with patch("ptq.provision.Path.home", return_value=tmp_path):
            assert ensure_ssh_config() is False
