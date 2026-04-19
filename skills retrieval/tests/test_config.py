from skills_retrieval.config import PoolSpec, RunConfig, TrialRecord


def test_pool_spec_roundtrip():
    spec = PoolSpec(task_id="sb_000", strategy="random", n=50, seed=0, representation="card")
    s = spec.model_dump_json()
    loaded = PoolSpec.model_validate_json(s)
    assert loaded == spec
    assert spec.pool_id == "sb_000__random__n50__s0__card"


def test_trial_record_has_all_fields():
    rec = TrialRecord(
        pool_id="sb_000__random__n50__s0__card",
        probe="awareness",
        model="claude-sonnet-4-6",
        raw_response="<skills>SKILL_000,SKILL_001,SKILL_002,SKILL_003,SKILL_004</skills>",
        extracted_ids=["SKILL_000", "SKILL_001", "SKILL_002", "SKILL_003", "SKILL_004"],
        format_status="clean",
        flags={},
        latency_ms=1234,
    )
    assert rec.probe == "awareness"
    assert len(rec.extracted_ids) == 5
