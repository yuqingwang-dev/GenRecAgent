import json

import pytest

from genrecagent.delivery.sharing import (
    credential_findings,
    public_metrics,
    render_replay,
    sanitize_environment,
)


def test_credential_audit_reports_locations_without_values():
    token = "sk-" + "x" * 32
    result = credential_findings("settings", "OPENAI_" + "API_KEY=" + token)
    assert result and result[0]["file"] == "settings" and result[0]["line"] == 1
    assert token not in json.dumps(result)
    assert not credential_findings("template", "OPENAI_" + "API_KEY=\n")
    assert not credential_findings("typed-source", "api_key: str | None = None")


def test_private_environment_is_redacted_but_localhost_and_scores_are_not():
    private = ".".join(["10", "8", "176", "95"])
    text = private + " /mnt/reviewer/project C:/Users/Person/env "
    text += "C:\\Users\\Person\\env 127.0.0.1 Recall@20=0.036560"
    sanitized = sanitize_environment(text)
    assert private not in sanitized and "Person" not in sanitized and "reviewer" not in sanitized
    assert "127.0.0.1" in sanitized and "0.036560" in sanitized


def test_replay_data_cannot_close_the_script_or_inject_html():
    hostile = "</script><script>alert('not a real action')</script>"
    template = "<script>const data = __GENREC_REPLAY_DATA__;</script>"
    html = render_replay(template, {"title": hostile})
    assert html.count("</script>") == 1 and "\\u003c" in html
    embedded = html.split("const data = ")[1].split(";</script>")[0]
    assert json.loads(embedded)["title"] == hostile
    with pytest.raises(ValueError):
        render_replay("no marker", {})


def test_public_recovery_summary_keeps_results_but_excludes_task_ids_and_traces():
    metric = {"tasks": 72, "contract_pass": 72, "feasible_tasks": 36, "feasible_filled": 36,
              "correct_clarifications": 36, "constraint_violations": 0,
              "policy_illegal_calls": 1, "generation_tokens": 10,
              "avg_execution_latency_ms": 3, "ablation": "none",
              "task_ids": ["private-task"], "user_id": "private-user"}
    report = {key: {} for key in (
        "scope", "users", "catalog_items", "retrieval_test", "dual_retrieval_validation_only",
        "sft_training", "training_code_commit", "training_seconds", "source_metric_sha256",
        "default_policy", "limits",
    )}
    report["policy_heldout_synthetic_subset"] = {"sft": metric}
    report["learned_chain"] = {"tokenizer": {"variant": "pca_ema"}}
    report["recovery_v3"] = {
        "scope": "synthetic normalized states", "status": "verified", "selection": {},
        "adapter_update": {}, "controller_used": False, "training_seconds": 1,
        "training_commit": "fixture", "source_sha256": {},
        "raw_results": {"compact_sft": {"metrics": metric,
                                        "independent_audit": {"trace": "private-user"}}},
    }
    result = public_metrics(report)
    assert result["recovery_v3"]["raw_results"]["compact_sft"]["contract_pass"] == 72
    assert result["learned_chain"]["tokenizer"]["variant"] == "pca_ema"
    assert "private-user" not in json.dumps(result) and "private-task" not in json.dumps(result)
