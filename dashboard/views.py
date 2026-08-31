import json

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from . import obstacle_store
from .forms import TrainingConfigForm


def _list_checkpoints():
    settings.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(settings.CHECKPOINTS_DIR.glob('*.pth')):
        items.append({'name': path.stem, 'mtime': path.stat().st_mtime})
    items.sort(key=lambda item: item['mtime'], reverse=True)
    return items


@ensure_csrf_cookie
def index(request):
    context = {
        'form': TrainingConfigForm(),
        'checkpoints': _list_checkpoints(),
        'checkpoints_json': json.dumps(_list_checkpoints()),
        'saved_layouts_json': json.dumps(obstacle_store.list_layouts()),
    }
    return render(request, 'dashboard/index.html', context)


@require_GET
def checkpoints_json(request):
    return JsonResponse({'checkpoints': _list_checkpoints()})


@require_GET
def list_obstacle_layouts_json(request):
    return JsonResponse({'layouts': obstacle_store.list_layouts()})


@require_GET
def load_obstacle_layout_json(request, name):
    try:
        cells = obstacle_store.load_layout(name)
    except (FileNotFoundError, ValueError):
        return JsonResponse({'error': 'Layout not found.'}, status=404)
    return JsonResponse({'name': name, 'cells': cells})


@require_POST
def save_obstacle_layout(request):
    try:
        data = json.loads(request.body)
        name = data['name']
        cells = data['cells']
    except (KeyError, ValueError):
        return JsonResponse({'error': 'Invalid payload.'}, status=400)

    try:
        obstacle_store.save_layout(name, cells)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    return JsonResponse({'ok': True, 'name': name})
