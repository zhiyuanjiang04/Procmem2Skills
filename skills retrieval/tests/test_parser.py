from skills_retrieval.parser import parse_response


def test_clean_selection():
    out = parse_response("<skill>SKILL_003</skill>", probe="selection")
    assert out["extracted_ids"] == ["SKILL_003"]
    assert out["format_status"] == "clean"
    assert out["flags"] == {}


def test_clean_awareness_five_items():
    out = parse_response("<skills>A,B,C,D,E</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D", "E"]
    assert out["format_status"] == "clean"


def test_awareness_fewer_than_five_flagged():
    out = parse_response("<skills>A,B</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B"]
    assert out["format_status"] == "warning"
    assert out["flags"].get("length_violation") is True


def test_awareness_duplicates_deduped_in_order():
    out = parse_response("<skills>A,B,A,C,D</skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D"]
    assert out["flags"].get("dup_violation") is True


def test_extracts_tag_when_surrounded_by_prose():
    out = parse_response("Let me analyze this. <skill>SKILL_007</skill> I picked this one because...", probe="selection")
    assert out["extracted_ids"] == ["SKILL_007"]
    assert out["format_status"] == "warning"


def test_parse_fail_on_missing_tag():
    out = parse_response("I cannot decide.", probe="selection")
    assert out["format_status"] == "fail"
    assert out["extracted_ids"] == []


def test_whitespace_around_ids_stripped():
    out = parse_response("<skills> A , B , C , D , E </skills>", probe="awareness")
    assert out["extracted_ids"] == ["A", "B", "C", "D", "E"]


def test_multiple_tags_take_first_warn():
    out = parse_response("<skill>X</skill> then also <skill>Y</skill>", probe="selection")
    assert out["extracted_ids"] == ["X"]
    assert out["flags"].get("multiple_tags") is True


def test_prefix_missing_normalised():
    """Model omits SKILL_ prefix and leading zeros; parser should recover."""
    out = parse_response("<skills>106,188,80,174,0</skills>", probe="awareness")
    assert out["extracted_ids"] == ["SKILL_106", "SKILL_188", "SKILL_080", "SKILL_174", "SKILL_000"]
    assert out["flags"].get("id_normalized") is True
    assert out["format_status"] == "warning"


def test_prefix_lowercase_normalised():
    """Model uses lowercase skill_ prefix; parser should normalise."""
    out = parse_response("<skill>skill_3</skill>", probe="selection")
    assert out["extracted_ids"] == ["SKILL_003"]
    assert out["flags"].get("id_normalized") is True


def test_non_skill_id_passthrough():
    """Non-SKILL_ IDs (e.g. gt IDs) pass through unchanged without normalisation flag."""
    out = parse_response("<skill>gt_sb_000_mesh</skill>", probe="selection")
    assert out["extracted_ids"] == ["gt_sb_000_mesh"]
    assert not out["flags"].get("id_normalized")
