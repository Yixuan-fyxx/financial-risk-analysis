from training.eval.scoring import load_general_probes, summarize


def test_load_general_probes_returns_prompt_reference_pairs():
    probes = load_general_probes()
    assert len(probes) >= 5
    for p in probes:
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert isinstance(p["reference"], str) and p["reference"]


def test_summarize_computes_mean_domain_score():
    result = summarize("stage1_sft", general_loss=1.2345, domain_scores=[0.8, 1.0, 0.6])
    assert result["checkpoint"] == "stage1_sft"
    assert result["general_capability_loss"] == 1.2345
    assert result["domain_verifier_score_mean"] == 0.8
    assert result["domain_verifier_scores"] == [0.8, 1.0, 0.6]


def test_summarize_handles_empty_domain_scores():
    result = summarize("base", general_loss=2.0, domain_scores=[])
    assert result["domain_verifier_score_mean"] is None
    assert result["domain_verifier_scores"] == []
