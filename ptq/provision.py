"""GPU reservation lifecycle — thin wrapper around gpu_dev_cli (lazy import)."""

from __future__ import annotations

from pathlib import Path


def _get_config_and_manager():
    """Lazy import of gpu-dev-cli Config + ReservationManager."""
    try:
        from gpu_dev_cli.config import Config
        from gpu_dev_cli.reservations import ReservationManager
    except ImportError as e:
        raise SystemExit(
            "gpu-dev-cli is not installed. Install with: pip install gpu-dev-cli\n"
            "Or install ptq with GPU support: pip install 'ptq[gpu]'"
        ) from e
    config = Config()
    return config, ReservationManager(config)


def _get_user_info(config):
    """Get authenticated user info from gpu-dev-cli."""
    from gpu_dev_cli.auth import authenticate_user

    return authenticate_user(config)


def create_reservation(
    *,
    gpu_type: str = "h100",
    gpu_count: int = 1,
    duration_hours: float = 4.0,
    dockerfile: str | None = None,
    name: str | None = None,
) -> tuple[str, str, dict]:
    """Reserve a GPU pod and wait for it to become active.

    Args:
        gpu_type: GPU type (e.g., "h100", "a100").
        gpu_count: Number of GPUs.
        duration_hours: Max reservation duration in hours.
        dockerfile: Base64-encoded tar.gz build context for gpu-dev.
        name: Optional reservation name.

    Returns:
        (reservation_id, pod_name, conn_info) where conn_info contains
        SSH connection details including ssh_command, pod_name, fqdn.
    """
    config, manager = _get_config_and_manager()
    user_info = _get_user_info(config)

    reservation_id = manager.create_reservation(
        user_id=user_info["user_id"],
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        duration_hours=duration_hours,
        github_user=user_info["github_user"],
        no_persistent_disk=True,
        dockerfile=dockerfile,
        name=name,
    )

    if not reservation_id:
        raise RuntimeError("Failed to create reservation (no ID returned).")

    # Wait for pod to be active (creates SSH config automatically)
    conn_info = manager.wait_for_reservation_completion(reservation_id)
    if not conn_info:
        raise RuntimeError(
            f"Reservation {reservation_id} timed out waiting for pod."
        )

    pod_name = conn_info.get("pod_name", "")

    return reservation_id, pod_name, conn_info


def cancel_reservation(reservation_id: str) -> bool:
    """Cancel a GPU reservation. Idempotent — never raises."""
    try:
        config, manager = _get_config_and_manager()
        user_info = _get_user_info(config)
        return manager.cancel_reservation(reservation_id, user_info["user_id"])
    except Exception:
        return False


def ensure_ssh_config() -> bool:
    """Check that ~/.ssh/config includes gpu-dev SSH configuration."""
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return False
    content = ssh_config.read_text()
    return "Include ~/.gpu-dev/" in content or "Include ~/.gpu-dev/*" in content
