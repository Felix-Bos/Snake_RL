import json

from channels.generic.websocket import AsyncWebsocketConsumer

from .forms import TrainingConfigForm
from .training_manager import manager

GROUP_NAME = 'training_updates'


class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(GROUP_NAME, self.channel_name)
        await self.accept()
        await self.send(text_data=json.dumps({'event': 'status_change', 'payload': manager.get_status()}))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(GROUP_NAME, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data)
        except (TypeError, ValueError):
            await self._send_error('Invalid message.')
            return

        action = data.get('action')

        if action == 'start_training':
            await self._handle_start_training(data.get('config') or {})
        elif action == 'stop':
            manager.stop()
        elif action == 'start_watch':
            await self._handle_start_watch(data)
        else:
            await self._send_error(f'Unknown action: {action}')

    async def _handle_start_training(self, raw_config):
        form = TrainingConfigForm(data=raw_config)
        if not form.is_valid():
            await self._send_error(f'Invalid training config: {form.errors.as_text()}')
            return

        config = form.cleaned_data
        config['obstacle_layout'] = raw_config.get('obstacle_layout')

        if not manager.start_training(config):
            await self._send_error('A training or watch run is already in progress.')

    async def _handle_start_watch(self, data):
        model_name = data.get('model_name')
        if not model_name:
            await self._send_error('No model selected.')
            return

        obs_type = data.get('obs_type', 'vector')
        obstacle_layout = data.get('obstacle_layout')
        n_episodes = int(data.get('n_episodes', 5))

        if not manager.start_watch(model_name, obs_type, obstacle_layout, n_episodes):
            await self._send_error('A training or watch run is already in progress.')

    async def _send_error(self, message):
        await self.send(text_data=json.dumps({'event': 'error', 'payload': {'message': message}}))

    # Handler name matches the "type" key used in group_send from training_loop.make_emitter().
    async def broadcast_event(self, event):
        await self.send(text_data=json.dumps({'event': event['event'], 'payload': event['payload']}))
