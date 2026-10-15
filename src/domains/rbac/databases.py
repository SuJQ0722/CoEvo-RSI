"""Shared SQL database lookup for mixed Spider/BIRD evolution."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BIRD_DATABASES = Path('data/RBAC-Text2SQL-Benchmark-official/data/Bird/dev_20251106/dev_databases')


def resolve_db_path(db_dir: str | None, db_id: str, *, project_root: Path = PROJECT_ROOT) -> str | None:
    if not db_id:
        return None
    roots = ([Path(db_dir)] if db_dir else []) + [
        project_root / 'data/spider/database',
        project_root / 'data/spider',
        project_root / BIRD_DATABASES,
    ]
    for root in roots:
        path = root / db_id / f'{db_id}.sqlite'
        if path.is_file():
            return str(path)
    return None
