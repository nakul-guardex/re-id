"""Enrollment exclusivity, office lock, and tripwire latch."""

import numpy as np

from multi_rtsp_reid.core.identity_manager import GlobalIdentityManager
from multi_rtsp_reid.core.quality_gate import ClearFrameQualityGate
from multi_rtsp_reid.core.tracklet_builder import Tracklet


PALETTE = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def _mgr():
    return GlobalIdentityManager(
        palette=PALETTE,
        match_threshold=0.70,
        match_margin=0.05,
        sticky_boost=0.0,
        max_exemplars=5,
    )


def test_two_balcony_people_get_two_ids_when_one_is_already_present():
    mgr = _mgr()
    a = _emb(1)
    b = a.copy()  # same clothes / similar crop — the reported failure
    first = mgr.match_camera_tracks(
        "office_balcony",
        [(1, a, np.zeros(4), None)],
        1.0,
        is_enrollment_cam=True,
        commit_exemplars=True,
    )
    assert len(first) == 1
    gid_a = first[1][0]

    second = mgr.match_camera_tracks(
        "office_balcony",
        [(2, b, np.zeros(4), None)],
        2.0,
        occupied_gids={gid_a},
        is_enrollment_cam=True,
        commit_exemplars=True,
    )
    assert 2 in second
    assert second[2][0] != gid_a
    assert len(mgr.identities) == 2


def test_same_frame_similar_people_are_not_collapsed():
    mgr = _mgr()
    a = _emb(3)
    b = a * 0.99
    b = b / np.linalg.norm(b)
    out = mgr.match_camera_tracks(
        "office_balcony",
        [(1, a, np.zeros(4), None), (2, b, np.zeros(4), None)],
        1.0,
        is_enrollment_cam=True,
        commit_exemplars=True,
    )
    assert out[1][0] != out[2][0]


def test_office_camera_does_not_mint_ids():
    mgr = _mgr()
    out = mgr.match_camera_tracks(
        "office_1",
        [(7, _emb(9), np.zeros(4), None)],
        1.0,
        is_enrollment_cam=False,
        commit_exemplars=False,
    )
    assert out == {}
    assert mgr.identities == {}


def test_office_lock_requires_the_same_gid():
    tr = Tracklet("office_1", 4)
    assert tr.note_match("GID_1", 0.8, 0.7, 3) is False
    assert tr.note_match("GID_2", 0.8, 0.7, 3) is False
    assert tr.consecutive_passes == 1
    assert tr.note_match("GID_2", 0.81, 0.7, 3) is False
    assert tr.note_match("GID_2", 0.82, 0.7, 3) is True
    assert tr.status == "CONFIRMED"
    assert tr.assigned_gid == "GID_2"


def test_tripwire_latches_after_first_crossing():
    gate = ClearFrameQualityGate(
        min_height=10,
        min_width=10,
        border_margin=0,
        solidity_range=(0.01, 1.0),
        min_confidence=0.1,
        min_blur_score=0.0,
        max_occlusion_iou=1.0,
        temporal_spacing_sec=0.0,
    )
    h, w = 100, 80
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[40:90, 20:60] = True
    box = np.array([20, 40, 60, 90], dtype=np.float32)

    import multi_rtsp_reid.config as cfg
    cfg.TRIPWIRE_Y_PCT = 0.5  # y=50, inside the mask

    first = gate.evaluate("office_balcony", 1, box, mask, 0.9, (h, w), frame, [], 1.0)
    assert first.passed, first.rejection_reasons

    cfg.TRIPWIRE_Y_PCT = 0.1  # line no longer on the body
    second = gate.evaluate("office_balcony", 1, box, mask, 0.9, (h, w), frame, [], 1.2)
    assert second.passed, second.rejection_reasons
