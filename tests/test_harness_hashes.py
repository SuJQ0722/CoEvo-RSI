"""Keep historical hash values while sharing only Python normalization."""

import pytest

from src.meta_evolver.evolution import semantic_harness_hash
from src.meta_evolver.weights import harness_hash
from src.utils.python_source import normalized_python


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "__init__.py").write_text(
        '"""package doc"""\nfrom .helper import run\nclass Harness:\n'
        '    name = "candidate"\n    def run(self):\n        """doc"""\n        return run()\n'
    )
    (tmp_path / "helper.py").write_text("def run():\n    return 42\n")
    (tmp_path / "dynamic.py").write_text("VALUE = 7\n")
    (tmp_path / "prompt.txt").write_text("prompt payload\n")
    return tmp_path


def test_hashes_match_values_captured_before_refactoring(bundle):
    assert (
        semantic_harness_hash(bundle)
        == "3871f642581330b4f85c45e2bbe882b6546a33f964ba95af94dc09f73b8f193f"
    )
    assert (
        harness_hash(bundle) == "b19865eca54755c3b6a782bcb2d0a75893be16459b8b3015565457936987b250"
    )


@pytest.mark.parametrize("filename", ["prompt.txt", "dynamic.py"])
def test_assets_and_unreachable_code_change_only_execution_identity(bundle, filename):
    semantic, execution = semantic_harness_hash(bundle), harness_hash(bundle)
    (bundle / filename).write_text("VALUE = 8\n")
    assert semantic_harness_hash(bundle) == semantic
    assert harness_hash(bundle) != execution


def test_reachable_logic_changes_both_identities(bundle):
    semantic, execution = semantic_harness_hash(bundle), harness_hash(bundle)
    (bundle / "helper.py").write_text("def run():\n    return 43\n")
    assert semantic_harness_hash(bundle) != semantic
    assert harness_hash(bundle) != execution


def test_comments_docstrings_and_candidate_metadata_keep_identities(bundle):
    semantic, execution = semantic_harness_hash(bundle), harness_hash(bundle)
    path = bundle / "__init__.py"
    path.write_text(
        path.read_text().replace("package doc", "new doc").replace('"candidate"', '"other"')
        + "# comment\n"
    )
    (bundle / "lineage.json").write_text('{"generation":2}')
    (bundle / "evolution_report.md").write_text("new report")
    assert semantic_harness_hash(bundle) == semantic
    assert harness_hash(bundle) == execution
    with pytest.raises(SyntaxError):
        normalized_python("def invalid(")
