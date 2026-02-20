from __future__ import annotations

import base64
import io
import tarfile

import pytest

from ptq.docker import (
    MAX_CONTEXT_BYTES,
    build_context_tarball,
    generate_dockerfile,
)


@pytest.fixture()
def skills_tree(tmp_path):
    """Create a minimal skills directory for testing."""
    skills_dir = tmp_path / "skills"
    (skills_dir / "make-pr").mkdir(parents=True)
    (skills_dir / "make-pr" / "SKILL.md").write_text("# Make PR\nCreate a draft PR.")
    (skills_dir / "verify-fix").mkdir(parents=True)
    (skills_dir / "verify-fix" / "SKILL.md").write_text("# Verify Fix\nRun tests.")
    return skills_dir


class TestGenerateDockerfile:
    def test_contains_from_and_copy(self, skills_tree):
        result = generate_dockerfile(skills_dir=skills_tree)
        assert "FROM " in result
        assert "COPY skills/ /home/dev/.claude/skills/" in result
        assert "chown" in result

    def test_custom_base_image(self, skills_tree):
        result = generate_dockerfile(
            skills_dir=skills_tree, base_image="nvidia/cuda:12.4"
        )
        assert result.startswith("FROM nvidia/cuda:12.4\n")

    def test_lists_skill_names(self, skills_tree):
        result = generate_dockerfile(skills_dir=skills_tree)
        assert "make-pr" in result
        assert "verify-fix" in result

    def test_empty_skills_dir(self, tmp_path):
        empty = tmp_path / "skills"
        empty.mkdir()
        result = generate_dockerfile(skills_dir=empty)
        assert "COPY skills/" in result


class TestBuildContextTarball:
    def test_returns_valid_base64(self, skills_tree):
        b64 = build_context_tarball(skills_dir=skills_tree)
        raw = base64.b64decode(b64)
        assert len(raw) > 0

    def test_tarball_contains_dockerfile(self, skills_tree):
        b64 = build_context_tarball(skills_dir=skills_tree)
        raw = base64.b64decode(b64)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            names = tar.getnames()
        assert "Dockerfile" in names

    def test_tarball_contains_skills(self, skills_tree):
        b64 = build_context_tarball(skills_dir=skills_tree)
        raw = base64.b64decode(b64)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            names = tar.getnames()
        assert any("make-pr" in n for n in names)
        assert any("verify-fix" in n for n in names)

    def test_custom_dockerfile_override(self, skills_tree, tmp_path):
        custom = tmp_path / "Custom.Dockerfile"
        custom.write_text("FROM custom:latest\nRUN echo hi\n")
        b64 = build_context_tarball(
            skills_dir=skills_tree, extra_dockerfile=custom
        )
        raw = base64.b64decode(b64)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            df = tar.extractfile("Dockerfile")
            assert df is not None
            content = df.read().decode()
        assert "FROM custom:latest" in content

    def test_size_limit_enforced(self, tmp_path):
        import os

        skills = tmp_path / "skills"
        (skills / "big").mkdir(parents=True)
        # Use random bytes so gzip can't compress them below the limit
        (skills / "big" / "SKILL.md").write_bytes(os.urandom(MAX_CONTEXT_BYTES + 1))
        with pytest.raises(ValueError, match="Build context too large"):
            build_context_tarball(skills_dir=skills)
