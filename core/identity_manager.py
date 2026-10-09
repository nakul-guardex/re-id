#!/usr/bin/env python3
"""
Global Identity Manager
=======================
Enrolls people on the balcony camera and matches those identities
on the other cameras. Two people in one frame never share an ID.
"""

import atexit
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class GlobalIdentity:
    """Represents a confirmed physical person across all cameras."""

    def __init__(self, gid: str, name: str, color: Tuple[int, int, int], initial_embedding: np.ndarray, avatar: Optional[np.ndarray] = None):
        self.gid = gid
        self.name = name
        self.color = color
        self.centroid = l2norm(initial_embedding.copy())
        self.exemplars: List[np.ndarray] = [initial_embedding.copy()]
        self.avatar_bgr: Optional[np.ndarray] = avatar.copy() if avatar is not None else None
        self.active_presence: Dict[str, Tuple[int, float]] = {}  # cam_id -> (local_tid, last_seen_ts)
        self.first_seen_ts: float = time.monotonic()
        self.last_seen_ts: float = time.monotonic()
        self.total_matches: int = 1

    def add_exemplar(self, embedding: np.ndarray, max_exemplars: int = 10, ema_alpha: float = 0.15):
        """Adds embedding using diversity sampling and updates running centroid."""
        normed = l2norm(embedding)
        # Update centroid via EMA
        self.centroid = l2norm((1.0 - ema_alpha) * self.centroid + ema_alpha * normed)
        self.last_seen_ts = time.monotonic()
        self.total_matches += 1

        if len(self.exemplars) < max_exemplars:
            self.exemplars.append(normed)
        else:
            # Farthest-point / diversity replacement: replace exemplar closest to the new one
            sims = [float(np.dot(normed, ex)) for ex in self.exemplars]
            most_redundant_idx = int(np.argmax(sims))
            if sims[most_redundant_idx] > 0.85:
                # Slight update to existing redundant exemplar
                self.exemplars[most_redundant_idx] = l2norm(0.7 * self.exemplars[most_redundant_idx] + 0.3 * normed)
            else:
                # Replace the most similar to preserve diversity
                self.exemplars[most_redundant_idx] = normed

    def compute_similarity(self, candidate_emb: np.ndarray) -> float:
        """
        Multi-scale similarity:
        Score = 0.6 * max(cand . exemplars) + 0.4 * (cand . centroid)
        """
        normed = l2norm(candidate_emb)
        cent_sim = float(np.dot(normed, self.centroid))
        max_ex_sim = max([float(np.dot(normed, ex)) for ex in self.exemplars], default=cent_sim)
        return float(0.6 * max_ex_sim + 0.4 * cent_sim)


class GlobalIdentityManager:
    """Matches balcony-enrolled identities onto the other cameras."""

    def __init__(
        self,
        palette: List[Tuple[int, int, int]],
        match_threshold: float = 0.48,
        match_margin: float = 0.03,
        sticky_boost: float = 0.08,
        max_exemplars: int = 5,
        ema_alpha: float = 0.15,
    ):
        self.palette = palette
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.sticky_boost = sticky_boost
        self.max_exemplars = max_exemplars
        self.ema_alpha = ema_alpha

        self.lock = threading.RLock()
        self.identities: Dict[str, GlobalIdentity] = {}
        self._next_id_counter = 1
        self.aliases: Dict[str, str] = {}

        # Live audit log of events
        self.event_log: List[Dict[str, Any]] = []
        self.debug_csv = None
        self._open_debug_csv()
        atexit.register(self.close)

    def _open_debug_csv(self):
        from ..config import LOGS_DIR
        path = LOGS_DIR / "matching_debug.csv"
        write_header = not path.exists()
        self.debug_csv = open(path, "a")
        if write_header:
            self.debug_csv.write("timestamp,camera,track_id,best_gid,actual_sim,second_best_sim,margin,is_match\n")
            self.debug_csv.flush()

    def close(self):
        with self.lock:
            if self.debug_csv is not None:
                try:
                    self.debug_csv.close()
                except Exception:
                    pass
                self.debug_csv = None

    def reset(self):
        """Clears the gallery for a new analysis session."""
        with self.lock:
            self.identities.clear()
            self._next_id_counter = 1
            self.aliases.clear()
            self.event_log.clear()
            self.close()
            self._open_debug_csv()

    def apply_match(
        self,
        gid: str,
        cam_id: str,
        tid: int,
        embedding: np.ndarray,
        capture_ts: float,
        score: float = 0.0,
        log: bool = False,
    ):
        """Commits a confirmed match: update exemplars and camera presence."""
        with self.lock:
            gid = self.resolve_gid(gid)
            ident = self.identities.get(gid)
            if ident is None:
                return
            ident.add_exemplar(embedding, max_exemplars=self.max_exemplars, ema_alpha=self.ema_alpha)
            ident.active_presence[cam_id] = (tid, capture_ts)
            if log:
                self.log_event(
                    "MATCH",
                    f"[{cam_id}] Track #{tid} matched {ident.name} ({gid}) | Sim: {score:.2f}",
                    {"cam_id": cam_id, "track_id": tid, "gid": gid, "sim": score},
                )

    def update_presence(self, gid: str, cam_id: str, tid: int, capture_ts: float):
        with self.lock:
            gid = self.resolve_gid(gid)
            ident = self.identities.get(gid)
            if ident is not None:
                ident.active_presence[cam_id] = (tid, capture_ts)

    def log_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        print(f"[{event_type}] {message}", flush=True)
        evt = {
            "type": event_type,
            "message": message,
            "timestamp": round(time.monotonic(), 2),
            "details": details or {},
        }
        self.event_log.append(evt)
        if len(self.event_log) > 200:
            self.event_log.pop(0)

    def resolve_gid(self, gid: str) -> str:
        """Union-Find path compression to resolve aliased/merged IDs."""
        with self.lock:
            curr = gid
            visited = []
            while curr in self.aliases:
                visited.append(curr)
                curr = self.aliases[curr]
            for v in visited:
                self.aliases[v] = curr
            return curr

    def similarity_to(self, gid: str, embedding: np.ndarray) -> float:
        with self.lock:
            resolved = self.resolve_gid(gid)
            ident = self.identities.get(resolved)
            if ident is None:
                return 0.0
            return ident.compute_similarity(embedding)

    def identity_view(self, gid: str) -> Optional[Tuple[str, str, Tuple[int, int, int]]]:
        """Thread-safe snapshot of (gid, name, bgr color) for rendering."""
        with self.lock:
            resolved = self.resolve_gid(gid)
            ident = self.identities.get(resolved)
            if ident is None:
                return None
            return resolved, ident.name, ident.color

    def match_camera_tracks(
        self,
        cam_id: str,
        tracks_with_embs: List[Tuple[int, np.ndarray, np.ndarray, Optional[str]]],
        capture_ts: float,
        occupied_gids: Optional[Set[str]] = None,
        is_enrollment_cam: bool = False,
        commit_exemplars: bool = False,
    ) -> Dict[int, Tuple[str, float]]:
        """
        In-Camera 1-to-1 Hungarian Matching.

        GIDs already visible on this camera (occupied_gids) are excluded so two
        people in the same frame cannot share an identity. Enrollment cameras
        mint a new GID for every unmatched track, including when the gallery is empty.
        """
        occupied = {self.resolve_gid(g) for g in (occupied_gids or set()) if g}
        MASKED = -10.0

        with self.lock:
            if not tracks_with_embs:
                return {}

            gids = list(self.identities.keys())
            assignments: Dict[int, Tuple[str, float]] = {}
            assigned_gids_in_this_cam = set(occupied)

            if not gids:
                if is_enrollment_cam:
                    for tid, emb, _, _ in tracks_with_embs:
                        new_gid = self._create_identity(emb)
                        assignments[tid] = (new_gid, 1.0)
                        assigned_gids_in_this_cam.add(new_gid)
                        self.identities[new_gid].active_presence[cam_id] = (tid, capture_ts)
                return assignments

            N_tracks = len(tracks_with_embs)
            M_gids = len(gids)

            sim_matrix = np.zeros((N_tracks, M_gids), dtype=np.float32)

            for i, (tid, emb, box, curr_gid) in enumerate(tracks_with_embs):
                curr_res = self.resolve_gid(curr_gid) if curr_gid else None
                for j, gid in enumerate(gids):
                    if gid in occupied:
                        sim_matrix[i, j] = MASKED
                        continue
                    sim = self.identities[gid].compute_similarity(emb)
                    if curr_res == gid:
                        sim += self.sticky_boost
                    sim_matrix[i, j] = sim

            cost_matrix = 1.0 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                tid, emb, box, curr_gid = tracks_with_embs[r]
                matched_gid = gids[c]
                raw_score = float(sim_matrix[r, c])
                if raw_score <= MASKED + 1 or matched_gid in occupied:
                    continue

                sticky = bool(curr_gid and self.resolve_gid(curr_gid) == matched_gid)
                actual_sim = raw_score - (self.sticky_boost if sticky else 0.0)

                valid_scores = [float(s) for s in sim_matrix[r, :] if s > MASKED + 1]
                valid_scores.sort(reverse=True)
                second_score = valid_scores[1] if len(valid_scores) > 1 else -1.0
                margin = raw_score - second_score if second_score > MASKED else raw_score + 1.0

                is_match = actual_sim >= self.match_threshold and margin >= self.match_margin

                try:
                    if self.debug_csv is not None:
                        self.debug_csv.write(
                            f"{time.time()},{cam_id},{tid},{matched_gid},{actual_sim:.4f},{second_score:.4f},{margin:.4f},{is_match}\n"
                        )
                        self.debug_csv.flush()
                except Exception:
                    pass

                if is_match:
                    assignments[tid] = (matched_gid, round(actual_sim, 3))
                    assigned_gids_in_this_cam.add(matched_gid)
                    occupied.add(matched_gid)
                    if commit_exemplars:
                        self.identities[matched_gid].add_exemplar(
                            emb, max_exemplars=self.max_exemplars, ema_alpha=self.ema_alpha
                        )
                        self.identities[matched_gid].active_presence[cam_id] = (tid, capture_ts)
                    if not curr_gid or self.resolve_gid(curr_gid) != matched_gid:
                        self.log_event(
                            "MATCH",
                            f"[{cam_id}] Track #{tid} matched {self.identities[matched_gid].name} ({matched_gid}) | Sim: {actual_sim:.2f}",
                            {"cam_id": cam_id, "track_id": tid, "gid": matched_gid, "sim": actual_sim},
                        )

            for i, (tid, emb, box, curr_gid) in enumerate(tracks_with_embs):
                if tid not in assignments and is_enrollment_cam:
                    # Ignore GIDs already taken in this frame. A high score against
                    # a neighbor must not block minting a new identity.
                    free_scores = [
                        float(sim_matrix[i, j])
                        for j, gid in enumerate(gids)
                        if gid not in occupied and float(sim_matrix[i, j]) > MASKED + 1
                    ]
                    best_sim = max(free_scores) if free_scores else 0.0
                    if best_sim < self.match_threshold:
                        new_gid = self._create_identity(emb)
                        assignments[tid] = (new_gid, 1.0)
                        assigned_gids_in_this_cam.add(new_gid)
                        occupied.add(new_gid)
                        self.identities[new_gid].active_presence[cam_id] = (tid, capture_ts)

            return assignments

    def _create_identity(self, initial_embedding: np.ndarray, avatar: Optional[np.ndarray] = None) -> str:
        """Internal helper to mint a new Global ID."""
        gid = f"GID_{self._next_id_counter}"
        name = f"Person #{self._next_id_counter}"
        color = self.palette[(self._next_id_counter - 1) % len(self.palette)]
        self._next_id_counter += 1

        identity = GlobalIdentity(gid=gid, name=name, color=color, initial_embedding=initial_embedding, avatar=avatar)
        self.identities[gid] = identity
        self.log_event("ENROLLED", f"Discovered and enrolled {name} ({gid})", {"gid": gid})
        return gid

    def get_gallery_cards(self) -> List[Dict[str, Any]]:
        """Returns JSON-serializable list of active person cards for Web UI."""
        with self.lock:
            cards = []
            now = time.monotonic()
            for gid, ident in sorted(self.identities.items(), key=lambda x: x[1].first_seen_ts):
                active_cams = [cid for cid, (_, ts) in ident.active_presence.items() if (now - ts) < 4.0]
                cards.append({
                    "gid": gid,
                    "name": ident.name,
                    "color_rgb": [int(ident.color[2]), int(ident.color[1]), int(ident.color[0])], # BGR to RGB
                    "exemplar_count": len(ident.exemplars),
                    "active_cameras": active_cams,
                    "is_currently_visible": len(active_cams) > 0,
                    "first_seen_sec": round(now - ident.first_seen_ts, 1),
                    "total_matches": ident.total_matches,
                })
            return cards
