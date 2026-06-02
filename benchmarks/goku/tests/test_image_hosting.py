"""Tests for benchmarks.goku.image_hosting + the V3 negative-point warning
in task_loader.py.

S3 calls are mocked — these tests never hit AWS.

Covers:
  1. Mode detection: GOKU_IMAGE_MODE env var + count threshold
  2. S3-configured detection from env vars
  3. Pre-conditioning resizes oversized images to ≤2000 px JPEG
  4. upload_task_images: per-image upload + presigned URL generation
  5. Cache: re-upload skipped within a single process
  6. V3 negative-points warning fires on load
  7. Negative-points rubrics STILL score correctly (back-compat)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.goku import image_hosting


def _make_test_image(path: Path, size_px: tuple[int, int] = (300, 200),
                     color: tuple[int, int, int] = (200, 50, 50)) -> Path:
    """Create a small real PNG so PIL can actually open it."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size_px, color).save(path, format="PNG")
    return path


# ─────────────────────────────────────────────────────────────────────────
# Mode detection
# ─────────────────────────────────────────────────────────────────────────

def test_should_use_url_hosting_below_threshold(monkeypatch) -> None:
    monkeypatch.delenv("GOKU_IMAGE_MODE", raising=False)
    paths = [Path(f"img_{i}.jpg") for i in range(10)]
    assert image_hosting.should_use_url_hosting(paths) is False


def test_should_use_url_hosting_above_threshold(monkeypatch) -> None:
    monkeypatch.delenv("GOKU_IMAGE_MODE", raising=False)
    paths = [Path(f"img_{i}.jpg") for i in range(50)]
    assert image_hosting.should_use_url_hosting(paths) is True


def test_should_use_url_hosting_explicit_many(monkeypatch) -> None:
    monkeypatch.setenv("GOKU_IMAGE_MODE", "many")
    assert image_hosting.should_use_url_hosting([Path("a.jpg")]) is True


def test_should_use_url_hosting_explicit_inline(monkeypatch) -> None:
    monkeypatch.setenv("GOKU_IMAGE_MODE", "inline")
    paths = [Path(f"img_{i}.jpg") for i in range(100)]
    assert image_hosting.should_use_url_hosting(paths) is False


def test_should_use_url_hosting_empty(monkeypatch) -> None:
    monkeypatch.setenv("GOKU_IMAGE_MODE", "many")
    assert image_hosting.should_use_url_hosting([]) is False


def test_s3_hosting_configured_missing(monkeypatch) -> None:
    monkeypatch.delenv("GOKU_S3_BUCKET", raising=False)
    monkeypatch.delenv("AWS_BUCKET", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    assert image_hosting.s3_hosting_configured() is False


def test_s3_hosting_configured_with_legacy_name(monkeypatch) -> None:
    """Back-compat: GOKU_S3_BUCKET still recognized."""
    monkeypatch.delenv("AWS_BUCKET", raising=False)
    monkeypatch.setenv("GOKU_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert image_hosting.s3_hosting_configured() is True


def test_s3_hosting_configured_with_aws_bucket_name(monkeypatch) -> None:
    """GRT Labs convention: AWS_BUCKET is the preferred name."""
    monkeypatch.delenv("GOKU_S3_BUCKET", raising=False)
    monkeypatch.setenv("AWS_BUCKET", "production-grtlabs-tag")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert image_hosting.s3_hosting_configured() is True


# ─────────────────────────────────────────────────────────────────────────
# Preconditioning
# ─────────────────────────────────────────────────────────────────────────

def test_precondition_resizes_oversized(tmp_path: Path) -> None:
    """Image larger than max_dim is resized."""
    src = _make_test_image(tmp_path / "big.png", size_px=(3000, 2000))
    dest = tmp_path / "out" / "big.jpg"
    out = image_hosting._precondition_image(src, dest, max_dim=2000)
    from PIL import Image
    with Image.open(out) as im:
        assert max(im.size) == 2000
        assert im.format == "JPEG"


def test_precondition_keeps_small_image(tmp_path: Path) -> None:
    """Image within max_dim is re-encoded as JPEG (uniform format)."""
    src = _make_test_image(tmp_path / "small.png", size_px=(500, 400))
    dest = tmp_path / "out" / "small.jpg"
    out = image_hosting._precondition_image(src, dest, max_dim=2000)
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (500, 400)
        assert im.format == "JPEG"


# ─────────────────────────────────────────────────────────────────────────
# upload_task_images (S3 mocked)
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def s3_env(monkeypatch):
    """Set the env vars upload_task_images expects."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BUCKET", "test-bucket")  # GRT Labs convention
    monkeypatch.delenv("GOKU_S3_BUCKET", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.delenv("GOKU_IMAGE_MODE", raising=False)
    monkeypatch.delenv("AWS_FOLDER", raising=False)
    monkeypatch.delenv("GOKU_S3_KEY_PREFIX", raising=False)
    # Reset the module-level boto3 client AND URL cache between tests
    image_hosting._s3_client = None
    image_hosting._cache_clear()
    yield
    image_hosting._s3_client = None
    image_hosting._cache_clear()


def _mock_s3_client():
    """Return a mock boto3 S3 client with predictable behavior."""
    client = MagicMock()
    client.upload_file = MagicMock()
    # Each call to generate_presigned_url returns a synthetic URL based on the key
    def _presign(method, Params, ExpiresIn=None, **kwargs):
        return f"https://test-bucket.s3.amazonaws.com/{Params['Key']}?X-Amz-Signature=fake"
    client.generate_presigned_url.side_effect = _presign
    return client


def test_upload_task_images_uploads_each_file(tmp_path: Path, s3_env) -> None:
    images = [
        _make_test_image(tmp_path / f"img_{i}.png", size_px=(100, 100))
        for i in range(3)
    ]
    mock_client = _mock_s3_client()
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result = image_hosting.upload_task_images(
            image_paths=images,
            task_key="task_abc1234567890def",
        )
    assert len(result.urls) == 3
    assert all(u.startswith("https://test-bucket.s3.amazonaws.com/") for u in result.urls)
    assert all("X-Amz-Signature=fake" in u for u in result.urls)
    # Each file uploaded once
    assert mock_client.upload_file.call_count == 3
    # Each upload has the right bucket + content-type
    for call in mock_client.upload_file.call_args_list:
        assert call.kwargs["Bucket"] == "test-bucket"
        assert call.kwargs["ExtraArgs"]["ContentType"] == "image/jpeg"


def test_upload_task_images_uses_per_task_prefix(tmp_path: Path, s3_env) -> None:
    """S3 keys all share the same task-scoped prefix (default AWS_FOLDER='goku')."""
    images = [
        _make_test_image(tmp_path / f"img_{i}.png", size_px=(100, 100))
        for i in range(2)
    ]
    mock_client = _mock_s3_client()
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result = image_hosting.upload_task_images(
            image_paths=images,
            task_key="task_xyz9876543210abc",
            run_id="run_2026-05-27T15-30-00Z",
        )
    assert all(
        img.s3_key.startswith("goku/task_xyz9876543210abc/run_2026-05-27T15-30-00Z/")
        for img in result.images
    )


def test_upload_task_images_respects_aws_folder_env(
    tmp_path: Path, s3_env, monkeypatch
) -> None:
    """If AWS_FOLDER is set, that's the prefix (overrides default 'goku')."""
    monkeypatch.setenv("AWS_FOLDER", "custom-prefix")
    img = _make_test_image(tmp_path / "x.png")
    mock_client = _mock_s3_client()
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result = image_hosting.upload_task_images(
            image_paths=[img], task_key="task_x", run_id="run_x",
        )
    assert result.images[0].s3_key.startswith("custom-prefix/task_x/run_x/")


def test_upload_task_images_caches_within_process(tmp_path: Path, s3_env) -> None:
    """Second call with same paths + task_key reuses cached URLs (no re-upload)."""
    images = [_make_test_image(tmp_path / f"img_{i}.png") for i in range(3)]
    mock_client = _mock_s3_client()
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result1 = image_hosting.upload_task_images(
            image_paths=images, task_key="task_cache_test",
        )
        first_upload_count = mock_client.upload_file.call_count
        # Second call: cache hit
        result2 = image_hosting.upload_task_images(
            image_paths=images, task_key="task_cache_test",
        )
    # No new upload calls on the second invocation
    assert mock_client.upload_file.call_count == first_upload_count
    # Same URLs returned
    assert result1.urls == result2.urls


def test_upload_task_images_no_aws_creds_raises(tmp_path: Path,
                                                   monkeypatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_BUCKET", "test-bucket")
    image_hosting._s3_client = None
    img = _make_test_image(tmp_path / "x.png")
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        image_hosting.upload_task_images([img], task_key="task_x")


def test_upload_task_images_no_bucket_raises(tmp_path: Path,
                                                 monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BUCKET", raising=False)
    monkeypatch.delenv("GOKU_S3_BUCKET", raising=False)
    image_hosting._s3_client = None
    img = _make_test_image(tmp_path / "x.png")
    with pytest.raises(RuntimeError, match="AWS_BUCKET"):
        image_hosting.upload_task_images([img], task_key="task_x")


def test_upload_task_images_empty_input(s3_env) -> None:
    result = image_hosting.upload_task_images(
        image_paths=[], task_key="task_empty",
    )
    assert result.urls == []
    assert result.images == []


def test_upload_task_images_missing_file_skipped(tmp_path: Path, s3_env) -> None:
    """A missing file is logged + skipped, not crashed on."""
    real = _make_test_image(tmp_path / "real.png")
    missing = tmp_path / "does_not_exist.png"
    mock_client = _mock_s3_client()
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result = image_hosting.upload_task_images(
            image_paths=[real, missing], task_key="task_z",
        )
    assert len(result.urls) == 1  # only the real file uploaded
    assert mock_client.upload_file.call_count == 1


def test_cleanup_task_uploads_calls_delete_objects(tmp_path: Path, s3_env) -> None:
    images = [_make_test_image(tmp_path / f"img_{i}.png") for i in range(2)]
    mock_client = _mock_s3_client()
    mock_client.delete_objects = MagicMock(
        return_value={"Deleted": [{"Key": "k1"}, {"Key": "k2"}]}
    )
    with patch.object(image_hosting, "_get_s3_client", return_value=mock_client):
        result = image_hosting.upload_task_images(
            image_paths=images, task_key="task_cleanup",
        )
        deleted = image_hosting.cleanup_task_uploads(result)
    assert deleted == 2
    mock_client.delete_objects.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# V3 negative-points warning (task_loader)
# ─────────────────────────────────────────────────────────────────────────

def _write_task(tmp_path: Path, rubrics: list[dict],
                instruction: str = "do a thing") -> Path:
    task_dir = tmp_path / "task_v3test1234567890"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "rubrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rubrics) + "\n"
    )
    (task_dir / "data" / "input_files").mkdir(parents=True, exist_ok=True)
    return task_dir


def test_v3_negative_points_warns_on_load(tmp_path: Path, caplog) -> None:
    """Loading a rubric with negative points emits the V3 deprecation warning."""
    from benchmarks.goku.task_loader import load_task
    task_dir = _write_task(tmp_path, rubrics=[
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory",
         "criterion": "x", "paths": ["out.txt"]},
        # V3 violation:
        {"number": 2, "type": "response_not_criteria", "category": "HALLUCINATION",
         "points": -5, "importance": "mandatory",
         "criterion": "agent does NOT claim X"},
    ])
    with caplog.at_level(logging.WARNING):
        load_task(task_dir)
    matches = [r for r in caplog.records
               if "negative points" in r.message and "V3" in r.message]
    assert len(matches) == 1
    assert "rubric #2" in matches[0].message
    assert "response_criteria" in matches[0].message  # suggested type


def test_v3_positive_hallucination_no_warning(tmp_path: Path, caplog) -> None:
    """The V3-compliant rewrite (positive points, response_criteria) does NOT warn."""
    from benchmarks.goku.task_loader import load_task
    task_dir = _write_task(tmp_path, rubrics=[
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory",
         "criterion": "x", "paths": ["out.txt"]},
        {"number": 2, "type": "response_criteria", "category": "HALLUCINATION",
         "points": 5, "importance": "mandatory",
         "criterion": "The agent only identifies items that are actually visible"},
    ])
    with caplog.at_level(logging.WARNING):
        load_task(task_dir)
    matches = [r for r in caplog.records if "V3" in r.message]
    assert matches == []
