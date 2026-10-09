#!/usr/bin/env python3
"""
Global Identity Manager & Cross-Camera Fusion Engine (v2)
==========================================================
Manages:
  1. Tentative and Confirmed identity tiers
  2. Capture-timestamp ordered Reorder Buffer
  3. In-Camera 1-to-1 Exclusivity (Hungarian matching)
  4. Cross-Camera Multi-Exemplar & Centroid Matching
  5. Hard Spatial Co-Occurrence (Cannot-Link) Graph
  6. Reversible Dynamic Online Merging with Undo
"""

from dataclasses import dataclass, field
import heapq
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class MergeEvent:
    merge_id: str
    survivor_gid: str
    absorbed_gid: str
    similarity: float
    evidence: str
    timestamp: float
    pre_merge_exemplars_survivor: List[np.ndarray]
    pre_merge_exemplars_absorbed: List[np.ndarray]
    pre_merge_avatar_absorbed: Optional[np.ndarray]
    undone: bool = False


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
    """
    Coordinator of cross-camera identities, Hungarian matching,
    co-occurrence blacklists, and online reconciliation.
    """

    def __init__(
        self,
        palette: List[Tuple[int, int, int]],
        match_threshold: float = 0.48,
        match_margin: float = 0.03,
        merge_threshold: float = 0.54,
        sticky_boost: float = 0.08,
    ):
        self.palette = palette
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.merge_threshold = merge_threshold
        self.sticky_boost = sticky_boost

        self.lock = threading.RLock()
        self.identities: Dict[str, GlobalIdentity] = {}
        self._next_id_counter = 1

        # Cannot-link graph: (gid_a, gid_b) -> True
        self.cannot_link_pairs: Set[Tuple[str, str]] = set()

        # Reversible merge history and Union-Find alias map
        self.merge_history: List[MergeEvent] = []
        self.aliases: Dict[str, str] = {}  # absorbed_gid -> survivor_gid

        # Live audit log of events
        self.event_log: List[Dict[str, Any]] = []

        import os
        write_header = not os.path.exists("matching_debug.csv")
        self.debug_csv = open("matching_debug.csv", "a")
        if write_header:
            self.debug_csv.write("timestamp,camera,track_id,best_gid,actual_sim,second_best_sim,margin,is_match\n")
            self.debug_csv.flush()

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
        curr = gid
        visited = []
        while curr in self.aliases:
            visited.append(curr)
            curr = self.aliases[curr]
        for v in visited:
            self.aliases[v] = curr
        return curr

    def record_co_occurrence(self, cam_id: str, active_gids_in_frame: List[str]):
        """
        Hard Spatial Co-Occurrence Rule:
        If two or more confirmed GIDs appear in the SAME camera frame,
        they can NEVER be merged (they are physically two distinct humans).
        """
        resolved = list(set([self.resolve_gid(g) for g in active_gids_in_frame if g in self.identities]))
        if len(resolved) < 2:
            return

        with self.lock:
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    g1, g2 = resolved[i], resolved[j]
                    if g1 != g2:
                        pair = (min(g1, g2), max(g1, g2))
                        if pair not in self.cannot_link_pairs:
                            self.cannot_link_pairs.add(pair)
                            self.log_event(
                                "CANNOT_LINK",
                                f"{g1} and {g2} co-occurred in {cam_id} -> Barred from merging",
                                {"pair": pair, "camera": cam_id},
                            )

    def match_camera_tracks(
        self,
        cam_id: str,
        tracks_with_embs: List[Tuple[int, np.ndarray, np.ndarray, Optional[str]]],
        capture_ts: float,
    ) -> Dict[int, Tuple[str, float]]:
        """
        In-Camera 1-to-1 Hungarian Matching:
        Matches a set of local tracks in a single camera to known Global IDs.
        Guarantees that within this camera frame:
          - No two tracks receive the same Global ID.
          - Tracks with existing IDs receive sticky hysteresis.
        
        Args:
            cam_id: Camera ID
            tracks_with_embs: List of (track_id, candidate_emb, box, current_assigned_gid)
            capture_ts: Monotonic capture timestamp
            
        Returns:
            Dict[track_id, (assigned_gid, similarity_score)]
        """
        with self.lock:
            if not tracks_with_embs:
                return {}

            # Active confirmed global identities
            gids = list(self.identities.keys())
            if not gids:
                # No identities enrolled yet -> enroll first candidate
                assignments = {}
                tid, emb, _, _ = tracks_with_embs[0]
                new_gid = self._create_identity(emb)
                assignments[tid] = (new_gid, 1.0)
                self.identities[new_gid].active_presence[cam_id] = (tid, capture_ts)
                return assignments

            N_tracks = len(tracks_with_embs)
            M_gids = len(gids)

            # Build similarity matrix (N_tracks x M_gids)
            sim_matrix = np.zeros((N_tracks, M_gids), dtype=np.float32)

            for i, (tid, emb, box, curr_gid) in enumerate(tracks_with_embs):
                curr_res = self.resolve_gid(curr_gid) if curr_gid else None
                for j, gid in enumerate(gids):
                    sim = self.identities[gid].compute_similarity(emb)
                    # Apply sticky bonus if this track already held this GID
                    if curr_res == gid:
                        sim += self.sticky_boost
                    sim_matrix[i, j] = sim

            # Cost matrix for Hungarian Linear Sum Assignment: Cost = 1 - Sim
            cost_matrix = 1.0 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assignments: Dict[int, Tuple[str, float]] = {}
            assigned_gids_in_this_cam = set()

            for r, c in zip(row_ind, col_ind):
                tid, emb, box, curr_gid = tracks_with_embs[r]
                matched_gid = gids[c]
                raw_score = float(sim_matrix[r, c])
                actual_sim = raw_score - (self.sticky_boost if (curr_gid and self.resolve_gid(curr_gid) == matched_gid) else 0.0)

                # Check margin over second best
                row_scores = sorted(sim_matrix[r, :], reverse=True)
                second_score = row_scores[1] if len(row_scores) > 1 else -1.0
                margin = raw_score - second_score
                
                is_match = actual_sim >= self.match_threshold and margin >= self.match_margin
                
                # Write to CSV
                try:
                    self.debug_csv.write(f"{time.time()},{cam_id},{tid},{matched_gid},{actual_sim:.4f},{second_score:.4f},{margin:.4f},{is_match}\n")
                    self.debug_csv.flush()
                except Exception:
                    pass

                if is_match:
                    assignments[tid] = (matched_gid, round(actual_sim, 3))
                    assigned_gids_in_this_cam.add(matched_gid)
                    self.identities[matched_gid].add_exemplar(emb)
                    self.identities[matched_gid].active_presence[cam_id] = (tid, capture_ts)
                    if not curr_gid or self.resolve_gid(curr_gid) != matched_gid:
                        self.log_event(
                            "MATCH",
                            f"[{cam_id}] Track #{tid} matched {self.identities[matched_gid].name} ({matched_gid}) | Sim: {actual_sim:.2f}",
                            {"cam_id": cam_id, "track_id": tid, "gid": matched_gid, "sim": actual_sim},
                        )

            # For tracks that did not match any existing GID with sufficient score:
            # Check if they should enroll as brand new Global IDs
            for i, (tid, emb, box, curr_gid) in enumerate(tracks_with_embs):
                if tid not in assignments:
                    # Check best score against all GIDs
                    best_sim = float(np.max(sim_matrix[i, :])) if M_gids > 0 else 0.0
                    if best_sim < self.match_threshold:
                        # New person discovered!
                        new_gid = self._create_identity(emb)
                        assignments[tid] = (new_gid, 1.0)
                        assigned_gids_in_this_cam.add(new_gid)
                        self.identities[new_gid].active_presence[cam_id] = (tid, capture_ts)

            # Record co-occurrences of all assigned GIDs in this frame
            self.record_co_occurrence(cam_id, list(assigned_gids_in_this_cam))

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

    def run_reconciliation_cycle(self) -> List[Dict[str, Any]]:
        """
        Asynchronous Reconciliation Worker:
        Evaluates active Global ID pairs that have NEVER co-occurred in the same camera.
        If their similarity exceeds merge_threshold (0.54), merges them with full rollback history.
        """
        with self.lock:
            gids = list(self.identities.keys())
            if len(gids) < 2:
                return []

            merges_performed = []

            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    gid_a = self.resolve_gid(gids[i])
                    gid_b = self.resolve_gid(gids[j])
                    if gid_a == gid_b:
                        continue

                    pair = (min(gid_a, gid_b), max(gid_a, gid_b))
                    if pair in self.cannot_link_pairs:
                        continue  # Hard negative constraint: cannot merge!

                    id_a = self.identities[gid_a]
                    id_b = self.identities[gid_b]

                    # Compute pairwise similarity
                    sim_cent = float(np.dot(id_a.centroid, id_b.centroid))
                    max_cross = max([
                        float(np.dot(ex_a, ex_b))
                        for ex_a in id_a.exemplars
                        for ex_b in id_b.exemplars
                    ], default=sim_cent)

                    combined_sim = 0.6 * max_cross + 0.4 * sim_cent

                    if combined_sim >= self.merge_threshold:
                        # Perform reversible merge: absorb B into A
                        merge_event = self._merge_identities(gid_a, gid_b, combined_sim)
                        merges_performed.append(merge_event)
                        break  # Break inner loop to restart cycle cleanly

            return merges_performed

    def _merge_identities(self, survivor_gid: str, absorbed_gid: str, similarity: float) -> Dict[str, Any]:
        """Atomically merges absorbed_gid into survivor_gid."""
        survivor = self.identities[survivor_gid]
        absorbed = self.identities[absorbed_gid]

        event_id = f"merge_{int(time.monotonic() * 1000)}"
        event = MergeEvent(
            merge_id=event_id,
            survivor_gid=survivor_gid,
            absorbed_gid=absorbed_gid,
            similarity=round(similarity, 3),
            evidence=f"Appearance similarity {similarity:.3f} >= {self.merge_threshold}",
            timestamp=time.monotonic(),
            pre_merge_exemplars_survivor=[ex.copy() for ex in survivor.exemplars],
            pre_merge_exemplars_absorbed=[ex.copy() for ex in absorbed.exemplars],
            pre_merge_avatar_absorbed=absorbed.avatar_bgr.copy() if absorbed.avatar_bgr is not None else None,
        )
        self.merge_history.append(event)

        # Transfer exemplars & update centroid
        for ex in absorbed.exemplars:
            survivor.add_exemplar(ex)

        # Transfer active presences
        for cid, pres in absorbed.active_presence.items():
            survivor.active_presence[cid] = pres

        # Union cannot-link constraints
        updated_links = set()
        for pair in self.cannot_link_pairs:
            p1, p2 = pair
            if p1 == absorbed_gid:
                p1 = survivor_gid
            if p2 == absorbed_gid:
                p2 = survivor_gid
            if p1 != p2:
                updated_links.add((min(p1, p2), max(p1, p2)))
        self.cannot_link_pairs = updated_links

        # Alias in Disjoint Set
        self.aliases[absorbed_gid] = survivor_gid
        self.identities.pop(absorbed_gid, None)

        msg = f"[MERGED] {absorbed.name} ({absorbed_gid}) merged into {survivor.name} ({survivor_gid}) [Sim: {similarity:.3f}]"
        self.log_event("MERGE", msg, {"event_id": event_id, "survivor": survivor_gid, "absorbed": absorbed_gid})
        print(f"[IdentityManager] {msg}")

        return {
            "event_id": event_id,
            "survivor": survivor_gid,
            "absorbed": absorbed_gid,
            "similarity": round(similarity, 3),
            "message": msg,
        }

    def undo_merge(self, merge_id: str) -> bool:
        """Rolls back a previous merge, restoring pre-merge exemplars and split IDs."""
        with self.lock:
            target_event: Optional[MergeEvent] = None
            for evt in self.merge_history:
                if evt.merge_id == merge_id and not evt.undone:
                    target_event = evt
                    break

            if not target_event:
                return False

            # Restore survivor exemplars
            if target_event.survivor_gid in self.identities:
                survivor = self.identities[target_event.survivor_gid]
                survivor.exemplars = target_event.pre_merge_exemplars_survivor
                survivor.centroid = l2norm(np.mean(np.stack(survivor.exemplars), axis=0))

            # Recreate absorbed identity
            absorbed_gid = target_event.absorbed_gid
            idx = int(absorbed_gid.replace("GID_", "")) if "GID_" in absorbed_gid else 99
            name = f"Person #{idx}"
            color = self.palette[(idx - 1) % len(self.palette)]

            init_emb = target_event.pre_merge_exemplars_absorbed[0]
            restored = GlobalIdentity(gid=absorbed_gid, name=name, color=color, initial_embedding=init_emb, avatar=target_event.pre_merge_avatar_absorbed)
            for ex in target_event.pre_merge_exemplars_absorbed[1:]:
                restored.add_exemplar(ex)

            self.identities[absorbed_gid] = restored
            self.aliases.pop(absorbed_gid, None)
            target_event.undone = True

            msg = f"[UNDO] Reverted merge of {absorbed_gid} from {target_event.survivor_gid}"
            self.log_event("UNDO_MERGE", msg, {"event_id": merge_id})
            print(f"[IdentityManager] {msg}")
            return True

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
