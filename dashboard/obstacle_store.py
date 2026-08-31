import json
import re

from django.conf import settings

_NAME_RE = re.compile(r'^[A-Za-z0-9_\- ]{1,64}$')


def _safe_path(name: str):
    if not _NAME_RE.match(name):
        raise ValueError('Invalid layout name.')
    settings.OBSTACLE_LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
    return settings.OBSTACLE_LAYOUTS_DIR / f'{name}.json'


def save_layout(name: str, cells: list) -> None:
    path = _safe_path(name)
    path.write_text(json.dumps({'cells': cells}))


def load_layout(name: str) -> list:
    path = _safe_path(name)
    if not path.exists():
        raise FileNotFoundError(name)
    data = json.loads(path.read_text())
    return data.get('cells', [])


def list_layouts() -> list:
    settings.OBSTACLE_LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in settings.OBSTACLE_LAYOUTS_DIR.glob('*.json'))
