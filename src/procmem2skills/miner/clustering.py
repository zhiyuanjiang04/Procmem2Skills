from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from procmem2skills.models import Trajectory, WorkflowCandidate, WorkflowCluster
from procmem2skills.normalization import normalize_text, operation_family, sequence_similarity


class WorkflowClusterer:
    def __init__(
        self,
        similarity_threshold: float = 0.34,
        structure_threshold: float = 0.5,
        cluster_backend: str = "auto",
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        dbscan_eps: float = 0.35,
        dbscan_min_samples: int = 2,
        embedding_strict: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.structure_threshold = structure_threshold
        self.cluster_backend = (cluster_backend or "auto").strip().lower()
        self.embedding_model = embedding_model or os.environ.get("PROCMEM_CLUSTER_EMBED_MODEL") or os.environ.get("OPENROUTER_EMBED_MODEL")
        self.embedding_base_url = (
            embedding_base_url
            or os.environ.get("PROCMEM_CLUSTER_EMBED_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.embedding_strict = bool(embedding_strict)

    def cluster(self, workflows: list[WorkflowCandidate], trajectories: list[Trajectory]) -> list[WorkflowCluster]:
        if not workflows:
            return []
        trajectory_by_episode = {trajectory.episode_id: trajectory for trajectory in trajectories}
        views = [_workflow_view(workflow, trajectory_by_episode) for workflow in workflows]
        if len(views) == 1:
            return [_cluster_from_members("cluster-1", views)]

        groups = self._cluster_with_embedding_dbscan(views)
        backend_used = "embedding-dbscan"
        if groups is None:
            if self.embedding_strict and self.cluster_backend in {"auto", "embedding-dbscan"}:
                raise RuntimeError("embedding DBSCAN clustering was required but unavailable")
            groups = self._cluster_with_lexical_union(views)
            backend_used = "lexical-union"

        clusters = []
        for cluster_index, members in enumerate(groups, start=1):
            cluster = _cluster_from_members(f"cluster-{cluster_index}", members)
            cluster.metadata = {**cluster.metadata, "backend": backend_used}
            clusters.append(cluster)
        return sorted(clusters, key=lambda cluster: (-cluster.support, cluster.cluster_id))

    def dedupe_workflows(self, workflows: list[WorkflowCandidate], clusters: list[WorkflowCluster]) -> list[WorkflowCandidate]:
        workflow_by_id = {workflow.workflow_id: workflow for workflow in workflows}
        deduped = []
        for cluster in clusters:
            for workflow_id in cluster.canonical_workflow_ids:
                workflow = workflow_by_id.get(workflow_id)
                if workflow is None:
                    continue
                workflow.metadata = {
                    **workflow.metadata,
                    "cluster_id": cluster.cluster_id,
                    "cluster_support": cluster.support,
                    "cluster_backend": cluster.metadata.get("backend"),
                }
                deduped.append(workflow)
        return deduped

    def _cluster_with_embedding_dbscan(self, views: list[dict]) -> list[list[dict]] | None:
        if self.cluster_backend in {"lexical", "heuristic", "tfidf"}:
            return None
        if not self.embedding_model:
            if self.embedding_strict:
                raise RuntimeError("embedding model is required for embedding DBSCAN clustering")
            return None
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key and not _is_local_embedding_url(self.embedding_base_url):
            if self.embedding_strict:
                raise RuntimeError("embedding API key is required when embedding endpoint is not local")
            return None

        try:
            matrix = _embed_documents(
                documents=[view["document"] for view in views],
                model=self.embedding_model,
                base_url=self.embedding_base_url,
                api_key=api_key or "",
            )
        except Exception:
            if self.embedding_strict:
                raise
            return None

        if matrix.shape[0] != len(views):
            if self.embedding_strict:
                raise RuntimeError("embedding matrix row count does not match workflow count")
            return None
        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
        ).fit_predict(matrix)

        grouped: dict[int, list[dict]] = defaultdict(list)
        singletons: list[list[dict]] = []
        for index, label in enumerate(labels):
            if int(label) == -1:
                singletons.append([views[index]])
                continue
            grouped[int(label)].append(views[index])
        ordered_groups = [grouped[key] for key in sorted(grouped)]
        ordered_groups.extend(singletons)
        return ordered_groups

    def _cluster_with_lexical_union(self, views: list[dict]) -> list[list[dict]]:
        documents = [view["document"] for view in views]
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(documents)
        similarity = cosine_similarity(matrix)
        parents = list(range(len(views)))

        for left in range(len(views)):
            for right in range(left + 1, len(views)):
                if self._should_link(views[left], views[right], similarity[left, right]):
                    _union(parents, left, right)

        groups: dict[int, list[dict]] = defaultdict(list)
        for index, view in enumerate(views):
            groups[_find(parents, index)].append(view)
        return list(groups.values())

    def _should_link(self, left: dict, right: dict, cosine_score: float) -> bool:
        if left["fingerprint"] == right["fingerprint"]:
            return True
        step_count_gap = abs(left["step_count"] - right["step_count"])
        if step_count_gap > max(1, min(left["step_count"], right["step_count"])):
            return False
        structure = _jaccard(left["step_keys"], right["step_keys"])
        family_overlap = _jaccard(left["action_families"], right["action_families"])
        sequence_score = sequence_similarity(left["action_families"], right["action_families"])
        same_tool_signature = left["tool_signature"] == right["tool_signature"]

        if cosine_score >= self.similarity_threshold and sequence_score >= self.structure_threshold:
            return True
        if same_tool_signature and structure >= self.structure_threshold and family_overlap >= self.structure_threshold:
            return True
        if same_tool_signature and cosine_score >= self.similarity_threshold and family_overlap >= self.structure_threshold:
            return True
        if cosine_score >= self.similarity_threshold + 0.2 and structure >= 0.34 and family_overlap >= 0.5:
            return True
        if same_tool_signature and sequence_score >= 0.9 and family_overlap >= 0.9:
            return True
        return False


def _embed_documents(*, documents: list[str], model: str, base_url: str, api_key: str, batch_size: int = 32) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        payload = {
            "model": model,
            "input": chunk,
        }
        request = urllib.request.Request(
            url=f"{base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=_embedding_headers(api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"embedding request failed with status {exc.code}: {detail}") from exc
        data = body.get("data") or []
        if not isinstance(data, list) or len(data) != len(chunk):
            raise RuntimeError("embedding response size mismatch")
        for item in sorted(data, key=lambda record: int(record.get("index", 0))):
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("invalid embedding payload")
            embeddings.append([float(value) for value in vector])
    return np.asarray(embeddings, dtype=np.float32)


def _embedding_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_local_embedding_url(base_url: str) -> bool:
    normalized = (base_url or "").strip().lower()
    return normalized.startswith("http://127.0.0.1") or normalized.startswith("http://localhost")


def _workflow_view(workflow: WorkflowCandidate, trajectory_by_episode: dict[str, Trajectory]) -> dict:
    episode_id = workflow.source_segment_id.split("-seg-", 1)[0]
    trajectory = trajectory_by_episode.get(episode_id)
    step_keys = [normalize_text(f"{step.tool}:{step.operation}") for step in workflow.steps]
    action_families = [operation_family(step.tool, step.operation) for step in workflow.steps]
    tool_signature = " ".join(step.tool or "tool" for step in workflow.steps)
    parts = [
        normalize_text(workflow.objective),
        normalize_text(workflow.trigger),
        " ".join(normalize_text(item) for item in workflow.preconditions),
        " ".join(step_keys),
        " ".join(action_families),
        " ".join(normalize_text(item) for item in workflow.verification),
        " ".join(normalize_text(item) for item in workflow.failure_modes),
    ]
    return {
        "workflow": workflow,
        "fingerprint": workflow.fingerprint,
        "step_keys": step_keys,
        "action_families": action_families,
        "tool_signature": tool_signature,
        "document": " ".join(part for part in parts if part),
        "benchmark": trajectory.benchmark.value if trajectory else "unknown",
        "trajectory": trajectory,
        "step_count": len(workflow.steps),
    }


def _cluster_from_members(cluster_id: str, members: list[dict]) -> WorkflowCluster:
    seen_fingerprints = set()
    canonical_workflow_ids = []
    benchmark_origins = set()
    harness_origins = set()
    agent_origins = set()
    task_origins = set()
    for member in members:
        workflow = member["workflow"]
        fingerprint = member["fingerprint"]
        if fingerprint not in seen_fingerprints:
            canonical_workflow_ids.append(workflow.workflow_id)
            seen_fingerprints.add(fingerprint)
        trajectory = member["trajectory"]
        if trajectory is None:
            continue
        benchmark_origins.add(trajectory.benchmark.value)
        harness_origins.add(trajectory.harness)
        agent_origins.add(trajectory.agent)
        task_origins.add(trajectory.task_id)
    return WorkflowCluster(
        cluster_id=cluster_id,
        member_workflow_ids=[member["workflow"].workflow_id for member in members],
        canonical_workflow_ids=canonical_workflow_ids,
        support=len(members),
        benchmark_origins=sorted(benchmark_origins),
        harness_origins=sorted(harness_origins),
        agent_origins=sorted(agent_origins),
        task_origins=sorted(task_origins),
        metadata={
            "unique_fingerprints": len(seen_fingerprints),
            "tool_signatures": sorted({member["tool_signature"] for member in members}),
        },
    )


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root != right_root:
        parents[right_root] = left_root
