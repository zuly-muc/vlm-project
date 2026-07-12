"""NuScenesLoader against the injected fake devkit object (no real dataset)."""

from __future__ import annotations

from vlm_project.dataset.nuscenes_loader import NuScenesLoader


def test_iter_clips_yields_one_clip_per_scene(fake_nusc):
    loader = NuScenesLoader(nusc=fake_nusc, camera="CAM_FRONT")
    clips = list(loader.iter_clips())

    assert [c.clip_id for c in clips] == ["scene-0000", "scene-0001"]
    assert all(c.loader == "nuscenes" for c in clips)
    assert all(c.camera == "CAM_FRONT" for c in clips)


def test_walks_the_full_keyframe_chain(fake_nusc):
    loader = NuScenesLoader(nusc=fake_nusc)
    clip = next(iter(loader.iter_clips()))

    # Three linked keyframes, indexed 0..2, each with a real image path.
    assert [f.index for f in clip.frames] == [0, 1, 2]
    assert all(f.image_path for f in clip.frames)
    assert clip.metadata["n_keyframes"] == 3
    assert clip.frames[0].metadata["sample_token"] == "s0_k0"
