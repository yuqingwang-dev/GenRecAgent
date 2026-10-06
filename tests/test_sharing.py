import json

import pytest

from genrecagent.delivery.sharing import credential_findings, render_replay, sanitize_environment


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
