"""Nightly memory consolidation: deduplicate and prune stale memories in Qdrant."""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from code_recall._mem0 import QDRANT_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.80
SITUATIONAL_MAX_AGE = timedelta(weeks=4)


def _discover_collections() -> tuple[str, ...]:
    """Auto-discover mem0 collections from Qdrant."""
    response = httpx.get(f"{QDRANT_URL}/collections")
    all_names = [entry["name"] for entry in response.json()["result"]["collections"]]
    mem0_collections = tuple(sorted(name for name in all_names if name.startswith("mem0_")))
    return mem0_collections


def main() -> None:
    """Entry point for nightly consolidation: deduplicate + prune stale situational facts."""
    collections = _discover_collections()
    logger.info("Discovered %d collections: %s", len(collections), ", ".join(collections))
    total_deleted = 0
    total_pruned = 0
    for collection in collections:
        total_deleted += _deduplicate_collection(collection)
        total_pruned += _prune_stale_situational(collection)

    logger.info(
        "Consolidation complete: %d duplicates removed, %d stale situational facts pruned",
        total_deleted,
        total_pruned,
    )


def _prune_stale_situational(collection: str) -> int:
    """Delete expired or stale situational facts.

    Situational facts with expires_at are pruned when past expiry.
    Situational facts without expires_at fall back to SITUATIONAL_MAX_AGE.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    age_cutoff = (datetime.now(timezone.utc) - SITUATIONAL_MAX_AGE).isoformat()

    response = httpx.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        json={
            "limit": 500,
            "with_payload": True,
            "filter": {
                "must": [
                    {"key": "temporal_scope", "match": {"value": "situational"}},
                ],
            },
        },
    )
    points = response.json().get("result", {}).get("points", [])

    stale_ids = []
    for point in points:
        payload = point.get("payload", {})
        expires_at = payload.get("expires_at")
        if expires_at:
            if expires_at < now_iso:
                stale_ids.append(point["id"])
                logger.info("  Pruning expired: %s (expires %s)", payload.get("data", "")[:60], expires_at[:10])
        else:
            timestamp = _get_timestamp(point)
            if timestamp and timestamp < age_cutoff:
                stale_ids.append(point["id"])
                logger.info("  Pruning stale: %s (sourced %s)", payload.get("data", "")[:60], timestamp[:10])

    if stale_ids:
        httpx.post(
            f"{QDRANT_URL}/collections/{collection}/points/delete",
            json={"points": stale_ids},
        )
        logger.info("[%s] Pruned %d stale/expired facts", collection, len(stale_ids))
    else:
        logger.info("[%s] No stale or expired facts", collection)

    return len(stale_ids)


def _deduplicate_collection(collection: str) -> int:
    """Remove duplicate points from a collection, keeping newer facts."""
    duplicates = _find_duplicates(collection)
    if not duplicates:
        logger.info("[%s] No duplicates found", collection)
        return 0

    logger.info("[%s] Found %d duplicate pairs", collection, len(duplicates))

    delete_ids = []
    for point_a, point_b, score in duplicates:
        older_id = _pick_older(point_a, point_b)
        newer_id = point_b["id"] if older_id == point_a["id"] else point_a["id"]
        logger.info(
            "  Duplicate (%.3f): keeping %s, deleting %s",
            score,
            str(newer_id)[:8],
            str(older_id)[:8],
        )
        delete_ids.append(older_id)

    if delete_ids:
        httpx.post(
            f"{QDRANT_URL}/collections/{collection}/points/delete",
            json={"points": delete_ids},
        )
        logger.info("[%s] Deleted %d duplicate points", collection, len(delete_ids))

    return len(delete_ids)


def _pick_older(point_a: dict, point_b: dict) -> str:
    """Return the ID of the older point (to be deleted)."""
    timestamp_a = _get_timestamp(point_a)
    timestamp_b = _get_timestamp(point_b)
    if timestamp_a <= timestamp_b:
        return point_a["id"]
    return point_b["id"]


def _get_timestamp(point: dict) -> str:
    """Extract the most recent timestamp from a point's payload."""
    payload = point.get("payload", {})
    timestamp = payload.get("sourced_at") or payload.get("updated_at") or payload.get("created_at") or ""
    return timestamp


def _find_duplicates(collection: str) -> list[tuple[dict, dict, float]]:
    """Find pairs of near-duplicate points by vector similarity."""
    points = _get_all_points(collection)
    logger.info("[%s] Scanning %d points for duplicates", collection, len(points))

    if len(points) < 2:
        return []

    points_by_id = {str(point["id"]): point for point in points}
    duplicates = []
    seen_pairs: set[tuple[str, str]] = set()

    for point in points:
        vector = point.get("vector")
        if not vector:
            continue
        response = httpx.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            json={"vector": vector, "limit": 5, "with_payload": True},
        )
        results = response.json().get("result", [])
        for match in results:
            if match["id"] == point["id"]:
                continue
            if match["score"] < SIMILARITY_THRESHOLD:
                continue
            id_pair = sorted([str(point["id"]), str(match["id"])])
            sorted_ids = (id_pair[0], id_pair[1])
            if sorted_ids in seen_pairs:
                continue
            seen_pairs.add(sorted_ids)
            match_point = points_by_id.get(str(match["id"]), match)
            duplicates.append((point, match_point, match["score"]))

    return duplicates


def _get_all_points(collection: str) -> list[dict]:
    """Scroll through all points in a Qdrant collection."""
    points = []
    offset = None
    while True:
        payload = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            payload["offset"] = offset
        response = httpx.post(f"{QDRANT_URL}/collections/{collection}/points/scroll", json=payload)
        data = response.json().get("result", {})
        batch = data.get("points", [])
        if not batch:
            break
        points.extend(batch)
        offset = data.get("next_page_offset")
        if offset is None:
            break
    return points
