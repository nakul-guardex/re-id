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
        """Cosine similarity against the identity's single stored vector."""
        normed = l2norm(candidate_emb)
        return float(np.dot(normed, self.centroid))


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
        Each track claims only its single best gallery identity.

        A claim is accepted when cosine similarity is at least match_threshold
        and beats the second-best identity by match_margin. If two tracks claim
        the same identity, the higher cosine wins and the other stays unmatched.
        A taken identity is never replaced by a track's second-best person.
        Enrollment cameras mint a new GID when no free identity clears the threshold.
        """
        occupied = {self.resolve_gid(g) for g in (occupied_gids or set()) if g}

        with self.lock:
            if not tracks_with_embs:
                return {}

            gids = list(self.identities.keys())
            assignments: Dict[int, Tuple[str, float]] = {}

            if not gids:
                if is_enrollment_cam:
                    for tid, emb, _, _ in tracks_with_embs:
                        new_gid = self._create_identity(emb)
                        assignments[tid] = (new_gid, 1.0)
                        self.identities[new_gid].active_presence[cam_id] = (tid, capture_ts)
                return assignments

            scored = []
            for tid, emb, _box, curr_gid in tracks_with_embs:
                pairs = [
                    (self.identities[gid].compute_similarity(emb), gid)
                    for gid in gids
                ]
                pairs.sort(key=lambda item: item[0], reverse=True)
                best_sim, best_gid = pairs[0]
                second_sim = pairs[1][0] if len(pairs) > 1 else -1.0
                margin = (best_sim - second_sim) if len(pairs) > 1 else (best_sim + 1.0)
                free_pairs = [(sim, gid) for sim, gid in pairs if gid not in occupied]
                best_free = free_pairs[0][0] if free_pairs else 0.0
                claimable = (
                    best_gid not in occupied
                    and best_sim >= self.match_threshold
                    and margin >= self.match_margin
                )
                scored.append({
                    "tid": tid,
                    "emb": emb,
                    "curr_gid": curr_gid,
                    "best_gid": best_gid,
                    "best_sim": float(best_sim),
                    "second_sim": float(second_sim),
                    "margin": float(margin),
                    "best_free": float(best_free),
                    "claimable": claimable,
                })

            claims: Dict[str, List[dict]] = {}
            for row in scored:
                self._log_match_row(
                    cam_id,
                    row["tid"],
                    row["best_gid"],
                    row["best_sim"],
                    row["second_sim"],
                    row["margin"],
                    row["claimable"],
                )
                if row["claimable"]:
                    claims.setdefault(row["best_gid"], []).append(row)

            winners = set()
            for gid, rows in claims.items():
                rows.sort(key=lambda row: row["best_sim"], reverse=True)
                winner = rows[0]
                winners.add(winner["tid"])
                score = winner["best_sim"]
                assignments[winner["tid"]] = (gid, round(score, 3))
                occupied.add(gid)
                if commit_exemplars:
                    self.identities[gid].add_exemplar(
                        winner["emb"], max_exemplars=self.max_exemplars, ema_alpha=self.ema_alpha
                    )
                    self.identities[gid].active_presence[cam_id] = (winner["tid"], capture_ts)
                curr = winner["curr_gid"]
                if not curr or self.resolve_gid(curr) != gid:
                    self.log_event(
                        "MATCH",
                        f"[{cam_id}] Track #{winner['tid']} matched {self.identities[gid].name} ({gid}) | Sim: {score:.2f}",
                        {"cam_id": cam_id, "track_id": winner["tid"], "gid": gid, "sim": score},
                    )

            if is_enrollment_cam:
                for row in scored:
                    if row["tid"] in assignments or row["tid"] in winners:
                        continue
                    # A track that wanted an existing identity stays unknown.
                    # A high score against someone already in this frame does not
                    # block a new balcony identity.
                    if row["claimable"] or row["best_free"] >= self.match_threshold:
                        continue
                    new_gid = self._create_identity(row["emb"])
                    assignments[row["tid"]] = (new_gid, 1.0)
                    occupied.add(new_gid)
                    self.identities[new_gid].active_presence[cam_id] = (row["tid"], capture_ts)

            return assignments

    def _log_match_row(self, cam_id, tid, best_gid, best_sim, second_sim, margin, is_match):
        try:
            if self.debug_csv is not None:
                second = f"{second_sim:.4f}" if second_sim >= 0 else ""
                self.debug_csv.write(
                    f"{time.time()},{cam_id},{tid},{best_gid},{best_sim:.4f},{second},{margin:.4f},{is_match}\n"
                )
                self.debug_csv.flush()
        except Exception:
            pass

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
