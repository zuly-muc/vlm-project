"""VideoLoader: decode a real, committed highway clip and sample keyframes.

Needs OpenCV (the ``video`` extra). It skips cleanly where OpenCV is not
installed; the ``docker-smoke`` CI job exercises the same path inside the
shipping image, which bundles OpenCV.

The fixture ``data/video/highway.mp4`` is real motorway footage (see
``data/video/ATTRIBUTION.md``) so decoding runs on genuine content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from vlm_project.dataset.video_loader import VideoLoader  # noqa: E402

_FIXTURE = Path(__file__).parent / "data" / "video" / "highway.mp4"


def test_fixture_exists():
    assert _FIXTURE.is_file(), "committed video fixture is missing"


def test_samples_keyframes_and_writes_them(tmp_path: Path):
    loader = VideoLoader(path=str(_FIXTURE), frame_dir=str(tmp_path / "frames"), every_n=15)
    clips = list(loader.iter_clips())

    assert len(clips) == 1
    clip = clips[0]
    assert clip.clip_id == "highway"
    assert clip.loader == "video"
    assert clip.metadata["sampled_every_n"] == 15

    # ~90 frames sampled every 15th -> a handful of keyframes; be tolerant of a
    # one-frame decode difference across OpenCV builds.
    assert 4 <= len(clip.frames) <= 8
    # every sampled frame was written to disk as a real image
    for fr in clip.frames:
        assert Path(fr.image_path).is_file()
    # frame indices are strictly increasing
    idx = [fr.index for fr in clip.frames]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)


def test_stride_controls_keyframe_count(tmp_path: Path):
    sparse = list(VideoLoader(str(_FIXTURE), str(tmp_path / "a"), every_n=45).iter_clips())[0]
    dense = list(VideoLoader(str(_FIXTURE), str(tmp_path / "b"), every_n=10).iter_clips())[0]
    assert len(dense.frames) > len(sparse.frames)


def test_directory_input_discovers_the_video(tmp_path: Path):
    clips = list(VideoLoader(str(_FIXTURE.parent), str(tmp_path / "frames")).iter_clips())
    assert [c.clip_id for c in clips] == ["highway"]
