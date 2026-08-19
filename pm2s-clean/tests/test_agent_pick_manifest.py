import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_agent_pick as r


def test_manifest_rows_create_workspace_skills_and_set_metrics(tmp_path):
    skill_a = tmp_path / "skill-a" / "SKILL.md"
    skill_a.parent.mkdir()
    skill_a.write_text("# Skill A\nUseful for parsing meshes.\n", encoding="utf-8")

    manifest = tmp_path / "seed-42.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_name": "task-one",
                        "task_description": "Find the mass of a mesh.",
                        "neighbors": [
                            {
                                "neighbor_task_name": "mesh-analysis",
                                "source_skill_md": str(skill_a),
                                "role": "gt",
                                "skill_slug": "mesh-analysis",
                            },
                            {
                                "neighbor_task_name": "noise skill",
                                "source_skill_md": "# Noise\nNot relevant.\n",
                                "role": "noise",
                                "skill_slug": "noise skill",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = r.load_manifest_rows(manifest, benchmark="skillsbench")
    assert rows[0]["gt_skill_names"] == ["mesh-analysis"]
    assert [s["skill_name"] for s in rows[0]["candidate_skills"]] == ["mesh-analysis", "noise skill"]

    workspace = r.prepare_agent_pick_workspace(tmp_path / "work", rows[0])
    assert (workspace / "skills" / "mesh-analysis" / "SKILL.md").read_text(encoding="utf-8") == skill_a.read_text(encoding="utf-8")
    assert (workspace / "skills" / "noise-skill" / "SKILL.md").read_text(encoding="utf-8") == "# Noise\nNot relevant.\n"

    picked = r.parse_picked_skills('{"skills": ["mesh-analysis", "extra"]}')
    hit, precision, recall = r.gt_set_metrics(picked, ["mesh-analysis", "other-gt"])
    assert hit is True
    assert precision == 0.5
    assert recall == 0.5

    assert r.canonicalize_picked_skills(["noise-skill", "mesh-analysis"], rows[0]) == ["noise skill", "mesh-analysis"]
    expanded = r.expand_command_template(
        'printf \'{"skills":["mesh-analysis"]}\' --model {model_leaf}',
        model="google/gemini-3.1-pro-preview",
        prompt_file=tmp_path / "prompt.md",
        prompt="PROMPT",
    )
    assert expanded == ["printf", '{"skills":["mesh-analysis"]}', "--model", "gemini-3.1-pro-preview"]
