"""Select the last round of the newest run, without falling back to stale runs."""

import re
import sys
from pathlib import Path


def latest_resume_round(output_root: Path) -> str:
    candidates = []
    for path in (output_root / 'rounds').glob('round_*'):
        match = re.fullmatch(r'round_(\d+)_(\d{8}T\d{6}Z(?:_[A-Za-z0-9_]+)?)', path.name)
        if path.is_dir() and match:
            candidates.append((match[2], int(match[1]), path))
    if not candidates:
        raise ValueError(f'No existing runs to resume in {output_root}/rounds')
    _, _, latest = max(candidates)
    if (latest / 'round_summary.json').is_file():
        raise ValueError(
            f'Latest run is complete at {latest.name}; start a new continuation '
            'without --resume. Older unfinished runs are not selected automatically.'
        )
    for artifact in ('snapshot.json', 'journal.json'):
        file = latest / artifact
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'Missing frozen resume artifact: {latest.name}/{artifact}')
    return latest.name


if __name__ == '__main__':
    try:
        print(latest_resume_round(Path(sys.argv[1])))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
