"""Dockerfile generation and build context tarball for gpu-dev provisioning."""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

DEFAULT_BASE_IMAGE = "308535385114.dkr.ecr.us-east-2.amazonaws.com/pytorch-gpu-dev-gpu-dev-image:latest"

SKILLS_DIR = Path(__file__).parent.parent / "skills"

MAX_CONTEXT_BYTES = 700 * 1024  # 700 KB SQS limit


def generate_dockerfile(
    *,
    skills_dir: Path | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
) -> str:
    """Generate a Dockerfile that bakes skills into the image."""
    skills = skills_dir or SKILLS_DIR
    lines = [
        f"FROM {base_image}",
        "",
        "# Copy Claude skills into personal scope (always loaded)",
        "COPY skills/ /home/dev/.claude/skills/",
        "RUN chown -R dev:dev /home/dev/.claude/skills/ 2>/dev/null || true",
    ]
    if skills.is_dir():
        for skill_dir in sorted(skills.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and skill_md.exists():
                lines.append(f"# Skill: {skill_dir.name}")
    return "\n".join(lines) + "\n"


def build_context_tarball(
    *,
    skills_dir: Path | None = None,
    extra_dockerfile: Path | None = None,
) -> str:
    """Build a tar.gz containing Dockerfile + skills/, return base64 string.

    If extra_dockerfile is provided, it is used instead of the generated one,
    but skills are still included in the build context.
    """
    skills = skills_dir or SKILLS_DIR

    if extra_dockerfile is not None:
        dockerfile_content = extra_dockerfile.read_text()
    else:
        dockerfile_content = generate_dockerfile(skills_dir=skills)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add Dockerfile
        dockerfile_bytes = dockerfile_content.encode()
        info = tarfile.TarInfo(name="Dockerfile")
        info.size = len(dockerfile_bytes)
        tar.addfile(info, io.BytesIO(dockerfile_bytes))

        # Add skills directory
        if skills.is_dir():
            for skill_path in sorted(skills.rglob("*")):
                if not skill_path.is_file():
                    continue
                rel = skill_path.relative_to(skills.parent)
                data = skill_path.read_bytes()
                info = tarfile.TarInfo(name=str(rel))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    compressed = buf.getvalue()
    if len(compressed) > MAX_CONTEXT_BYTES:
        raise ValueError(
            f"Build context too large: {len(compressed)} bytes "
            f"(max {MAX_CONTEXT_BYTES} bytes). "
            "Reduce skills content or use fewer files."
        )

    return base64.b64encode(compressed).decode()
