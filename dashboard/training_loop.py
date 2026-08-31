"""Training/inference loops adapted from train.py's train_process/evaluate_process,
with tqdm/print replaced by an `emit(event_type, payload)` callback so a background
thread can stream live state to connected dashboard clients over Channels."""

import time

import numpy as np
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

GROUP_NAME = 'training_updates'

# Duplicated from train.py's get_model_config() on purpose, to keep this app
# fully decoupled from the CLI script.
MODEL_CONFIG = {
    'conv_layers': [
        {'out': 32, 'kernel': 8, 'stride': 4, 'activation': 'relu'},
        {'out': 64, 'kernel': 4, 'stride': 2, 'activation': 'relu'},
    ],
    'hidden_layers': [256, 256],
    'activations': ['relu', 'relu'],
}

GAME_STATE_MIN_INTERVAL = 0.05  # seconds, throttles game_state during fast training
SLOW_TRAINING_STEP_DELAY = 0.08  # seconds, sleep between steps in "slow" training preview
WATCH_STEP_DELAY = 0.08  # seconds, sleep between steps in watch mode so it's watchable


def make_emitter():
    channel_layer = get_channel_layer()

    def emit(event_type: str, payload: dict):
        async_to_sync(channel_layer.group_send)(
            GROUP_NAME,
            {'type': 'broadcast_event', 'event': event_type, 'payload': payload},
        )

    return emit


def _game_state_payload(env, score, step):
    return {
        'snake': list(env.snake),
        'head': list(env.head),
        'food': list(env.food),
        'obstacles': list(env.obstacles),
        'cols': env.cols,
        'rows': env.rows,
        'score': score,
        'step': step,
    }


def run_training(env, agent, config, emit, stop_event):
    """Adapted train_process(). Returns the reason training stopped."""
    epsilon = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    episodes = config['episodes']
    max_steps = config['max_steps']
    slow_preview = config.get('train_speed') == 'slow'

    run_name = (config.get('run_name') or '').strip()
    prefix = f'{run_name}_' if run_name else ''
    name_best = f'{prefix}best_model'
    name_final = f'{prefix}final_model'
    name_interrupted = f'{prefix}interrupted_model'

    scores = []
    best_eval_score = -float('inf')
    start_time = time.time()
    last_emit = 0.0
    reason = 'completed'

    for episode in range(episodes):
        if stop_event.is_set():
            reason = 'stopped'
            break

        obs, _ = env.reset()
        done = False
        score = 0.0
        steps = 0
        last_loss = None

        while not done:
            if stop_event.is_set():
                reason = 'stopped'
                break

            steps += 1
            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            if steps >= max_steps:
                truncated = True
                terminated = True

            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            score += reward

            if agent.steps_done % 1000 == 0 and hasattr(agent, 'update_target_net'):
                agent.update_target_net()

            if slow_preview:
                # Show every single step so the snake's movement reads as continuous,
                # at the cost of slowing training down to a watchable pace.
                emit('game_state', _game_state_payload(env, score, steps))
                time.sleep(SLOW_TRAINING_STEP_DELAY)
            else:
                now = time.time()
                if now - last_emit >= GAME_STATE_MIN_INTERVAL:
                    emit('game_state', _game_state_payload(env, score, steps))
                    last_emit = now

        if reason == 'stopped':
            break

        scores.append(score)
        epsilon = max(eps_end, epsilon * eps_decay)
        agent.log('Reward/Train', score)

        mean_score = float(np.mean(scores[-50:]))
        emit('metrics_update', {
            'episode': episode,
            'score': score,
            'mean_score_50': mean_score,
            'loss': last_loss,
            'epsilon': epsilon,
            'steps': steps,
            'elapsed_sec': time.time() - start_time,
            'buffer_size': len(agent.memory),
        })

        if episode > 0 and episode % 50 == 0:
            eval_score = run_watch(env, agent, n_episodes=3, emit=None, stop_event=stop_event, silent=True)
            agent.log('Reward/Eval', eval_score)

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save(name_best)

    if reason == 'stopped':
        agent.save(name_interrupted)
    else:
        agent.save(name_final)

    return reason, best_eval_score


def run_watch(env, agent, n_episodes, emit, stop_event, silent=False):
    """Adapted evaluate_process(). When silent=True, runs quietly (used for the
    intermediate eval pass inside training) and does not emit anything."""
    agent.set_mode('eval')
    scores = []

    for episode in range(n_episodes):
        if stop_event is not None and stop_event.is_set():
            break

        obs, _ = env.reset()
        done = False
        score = 0.0
        steps = 0

        while not done:
            if stop_event is not None and stop_event.is_set():
                break

            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1

            if not silent:
                emit('game_state', _game_state_payload(env, score, steps))
                time.sleep(WATCH_STEP_DELAY)

        scores.append(score)
        if not silent:
            emit('episode_finished', {'episode': episode, 'score': score})

    agent.set_mode('train')
    return float(np.mean(scores)) if scores else 0.0
