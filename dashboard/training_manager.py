"""Module-level singleton orchestrating the training/watch background thread.

Only one run (training or watch) can be active at a time, matching the
"local single-user tool" scope of this dashboard -- no task queue needed."""

import threading

import torch

from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from env import SnakeEnv

from .training_loop import MODEL_CONFIG, make_emitter, run_training, run_watch


def _device():
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.status = 'idle'  # idle | training | watching | stopping
        self.config = None
        self.agent = None
        self.error = None

    def get_status(self) -> dict:
        return {'status': self.status, 'config': self.config, 'error': self.error}

    def start_training(self, config: dict) -> bool:
        with self._lock:
            if self.status != 'idle':
                return False
            self.stop_event.clear()
            self.status = 'training'
            self.config = config
            self.error = None
            self.thread = threading.Thread(
                target=self._run_training_thread, args=(config,), daemon=True
            )
            self.thread.start()
            return True

    def start_watch(self, model_name: str, obs_type: str, obstacle_layout, n_episodes: int) -> bool:
        with self._lock:
            if self.status != 'idle':
                return False
            self.stop_event.clear()
            self.status = 'watching'
            self.error = None
            self.thread = threading.Thread(
                target=self._run_watch_thread,
                args=(model_name, obs_type, obstacle_layout, n_episodes),
                daemon=True,
            )
            self.thread.start()
            return True

    def stop(self):
        self.stop_event.set()
        if self.status == 'training':
            self.status = 'stopping'
        elif self.status == 'watching':
            self.status = 'stopping'

    def _build_env(self, obs_type, obstacles_count, no_food_steps, step_penalty, fixed_obstacles):
        no_food = None if not no_food_steps or no_food_steps <= 0 else no_food_steps
        n_obstacles = 0 if fixed_obstacles else obstacles_count
        return SnakeEnv(
            obs_type=obs_type,
            n_obstacles=n_obstacles,
            max_no_food_steps=no_food,
            step_penalty=step_penalty,
            fixed_obstacles=fixed_obstacles,
        )

    def _run_training_thread(self, config):
        emit = make_emitter()
        env = None
        try:
            fixed_obstacles = config.get('obstacle_layout') if config.get('use_custom_obstacles') else None
            env = self._build_env(
                config['obs'], config['obstacles'], config.get('no_food_steps', 0),
                config['step_penalty'], fixed_obstacles,
            )
            input_shape = env.observation_space.shape
            n_actions = env.action_space.n
            run_name = f"{config['algo']}_{config['obs']}_web"

            AgentCls = DDQNAgent if config['algo'] == 'DDQN' else DQNAgent
            agent = AgentCls(
                input_shape, n_actions, _device(), run_name,
                lr=config['lr'], batch_size=config['batch_size'], model_config=MODEL_CONFIG,
            )

            if config.get('resume') and config.get('resume_model'):
                agent.load(config['resume_model'])

            self.agent = agent
            emit('status_change', {'status': 'training', 'config': config, 'error': None})

            reason, best_eval_score = run_training(env, agent, config, emit, self.stop_event)
            emit('training_finished', {'reason': reason, 'best_eval_score': best_eval_score})
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            self.error = str(exc)
            emit('training_finished', {'reason': 'error', 'message': str(exc)})
        finally:
            if env is not None:
                env.close()
            self.status = 'idle'
            self.thread = None
            emit('status_change', {'status': 'idle', 'config': None, 'error': self.error})

    def _run_watch_thread(self, model_name, obs_type, obstacle_layout, n_episodes):
        emit = make_emitter()
        env = None
        try:
            env = self._build_env(obs_type, 10, 0, 0.01, obstacle_layout)
            input_shape = env.observation_space.shape
            n_actions = env.action_space.n

            # Network architecture is identical between DQN/DDQN (only train_step
            # differs, which is never called during inference) so DQNAgent is used
            # unconditionally for watch mode regardless of how the checkpoint was trained.
            agent = DQNAgent(input_shape, n_actions, _device(), 'watch_web', model_config=MODEL_CONFIG)
            agent.load(model_name)

            self.agent = agent
            emit('status_change', {'status': 'watching', 'config': {'model': model_name}, 'error': None})

            run_watch(env, agent, n_episodes, emit, self.stop_event)
            emit('training_finished', {'reason': 'stopped' if self.stop_event.is_set() else 'completed'})
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            emit('training_finished', {'reason': 'error', 'message': str(exc)})
        finally:
            if env is not None:
                env.close()
            self.status = 'idle'
            self.thread = None
            emit('status_change', {'status': 'idle', 'config': None, 'error': self.error})


manager = TrainingManager()
