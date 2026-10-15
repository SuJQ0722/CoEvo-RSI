from src.domains.rbac.databases import BIRD_DATABASES, resolve_db_path
from src.domains.rbac.evaluator import evaluate


def test_mixed_database_lookup(tmp_path):
    for root, db in [('data/spider/database', 'spider_db'), (BIRD_DATABASES, 'bird_db')]:
        path = tmp_path / root / db / f'{db}.sqlite'
        path.parent.mkdir(parents=True)
        path.touch()
        assert resolve_db_path(None, db, project_root=tmp_path) == str(path)
    assert resolve_db_path(None, 'absent', project_root=tmp_path) is None


def test_subset_metrics_do_not_hide_regression():
    records = [
        {'sample_id': 's', 'test_type': 'spider:easy', 'gold': {'valid': 1},
         'prediction': {'access': 1, 'sql': 'SELECT 1'}},
        {'sample_id': 'b', 'test_type': 'bird:simple', 'gold': {'valid': 1},
         'prediction': {'access': 0}},
    ]
    result = evaluate(records)
    subsets = result.diagnostics['per_dataset']
    assert result.metrics['safe_ex'] == 0.5
    assert subsets['spider'] == {'total': 1, 'safe_ex': 1.0, 'ac_f1': 1.0}
    assert subsets['bird'] == {'total': 1, 'safe_ex': 0.0, 'ac_f1': 0.0}
