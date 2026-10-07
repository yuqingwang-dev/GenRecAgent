from scripts.evaluate_dense import optional_package_version


def test_unused_encoder_distribution_is_not_required_for_saved_index_evaluation():
    assert optional_package_version("genrecagent-definitely-not-an-installed-distribution") is None
    assert isinstance(optional_package_version("numpy"), str)
