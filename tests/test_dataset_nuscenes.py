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
    assert clip.metadata["nuscenes_version"] == "v1.0-mini"
    assert clip.frames[0].metadata["sample_token"] == "s0_k0"


class _ConfigurableNusc:
    """A NuScenes stand-in with hand-built samples, for the defect cases below."""

    def __init__(self, scenes, samples, data, raise_on=()):
        self.scene = scenes
        self._samples = samples
        self._data = data
        self._raise_on = set(raise_on)

    def get(self, table, token):
        assert table == "sample"
        if token in self._raise_on:
            raise KeyError(f"corrupt sample {token}")
        return self._samples[token]

    def get_sample_data(self, cam_token):
        return self._data[cam_token], [], None


def _scene(name, first):
    return {"name": name, "token": f"tok-{name}", "first_sample_token": first}


def test_missing_camera_channel_skips_that_keyframe(make_image):
    samples = {
        "k0": {"token": "k0", "next": "k1", "data": {"CAM_FRONT": "k0cam"}},
        "k1": {"token": "k1", "next": "", "data": {}},  # no CAM_FRONT
    }
    nusc = _ConfigurableNusc([_scene("scene-0000", "k0")], samples,
                             {"k0cam": make_image("k0.jpg")})
    clip = next(iter(NuScenesLoader(nusc=nusc).iter_clips()))

    assert [f.index for f in clip.frames] == [0]  # k1 skipped, not fatal
    assert clip.metadata["n_keyframes"] == 1


def test_scene_with_no_usable_keyframe_yields_frameless_clip(make_image):
    samples = {"k0": {"token": "k0", "next": "", "data": {}}}  # only sample, no camera
    nusc = _ConfigurableNusc([_scene("scene-0000", "k0")], samples, {})
    clip = next(iter(NuScenesLoader(nusc=nusc).iter_clips()))

    assert clip.frames == []
    assert clip.metadata["n_keyframes"] == 0


def test_cycle_in_sample_chain_is_broken(make_image):
    samples = {
        "a": {"token": "a", "next": "b", "data": {"CAM_FRONT": "ac"}},
        "b": {"token": "b", "next": "a", "data": {"CAM_FRONT": "bc"}},  # loops back
    }
    nusc = _ConfigurableNusc([_scene("scene-0000", "a")], samples,
                             {"ac": make_image("a.jpg"), "bc": make_image("b.jpg")})
    clip = next(iter(NuScenesLoader(nusc=nusc).iter_clips()))

    assert [f.index for f in clip.frames] == [0, 1]  # a, b, then stops (no infinite loop)


def test_corrupt_scene_does_not_abort_iteration(make_image):
    samples = {"a": {"token": "a", "next": "", "data": {"CAM_FRONT": "ac"}}}
    nusc = _ConfigurableNusc(
        [_scene("bad", "boom"), _scene("good", "a")],
        samples, {"ac": make_image("a.jpg")}, raise_on=["boom"],
    )
    clips = list(NuScenesLoader(nusc=nusc).iter_clips())

    assert [c.clip_id for c in clips] == ["bad", "good"]  # bad scene skipped, not fatal
    assert clips[0].frames == []
    assert len(clips[1].frames) == 1
