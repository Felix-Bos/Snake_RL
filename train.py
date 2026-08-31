import argparse
import os

# Workaround for duplicate libomp runtime on some macOS conda setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import time
from tqdm import tqdm
from env import SnakeEnv 
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent

# --- CONFIGURATION DU MODELE ---
def get_model_config():
    return {
        'conv_layers': [
            {'out': 32, 'kernel': 8, 'stride': 4, 'activation': 'relu'},
            {'out': 64, 'kernel': 4, 'stride': 2, 'activation': 'relu'},
        ],
        'hidden_layers': [256, 256], 
        'activations': ['relu', 'relu']
    }

def get_args():
    parser = argparse.ArgumentParser(description="Snake RL Trainer Global")
    
    # Paramètres d'apprentissage
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "DDQN"])
    parser.add_argument("--obs", type=str, default="vector", choices=["vector", "image"])
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre total d'épisodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Limite de pas par épisode")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00025)
    
    # Paramètres de l'environnement
    parser.add_argument("--obstacles", type=int, default=10)
    parser.add_argument("--no_food_steps", type=int, default=0,
                        help="Nombre max de pas sans manger avant troncature (0 = auto)")
    parser.add_argument("--step_penalty", type=float, default=0.01,
                        help="Penalite appliquee a chaque pas sans nourriture")
    
    # Modes d'exécution
    parser.add_argument("--mode", type=str, default="all", choices=["train", "eval", "all"])
    parser.add_argument("--render_train", action="store_true", help="Voir le jeu PENDANT l'entraînement (lent)")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Nombre de parties d'évaluation")
    parser.add_argument("--model", type=str, default="best_model", help="Nom du modèle à charger en eval (avec ou sans .pth)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre l'entrainement depuis un checkpoint")
    parser.add_argument("--resume_model", type=str, default="final_model",
                        help="Nom du checkpoint pour reprendre le train (avec ou sans .pth)")
    
    return parser.parse_args()

# --- FONCTION D'ENTRAINEMENT ---
def train_process(env, agent, args):
    print(f"--- Démarrage de l'entraînement ({args.algo}) ---")
    
    epsilon = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    
    scores = []
    best_eval_score = -float('inf')
    
    pbar = tqdm(range(args.episodes), unit="ep")

    for episode in pbar:
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            steps += 1
            # Action selection
            action = agent.select_action(obs, epsilon)
            
            # Step env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Vérification de la limite de pas
            if steps >= args.max_steps:
                truncated = True
                terminated = True
            
            done = terminated or truncated
            
            # Stockage & Apprentissage
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.train_step()
            
            obs = next_obs
            score += reward
            
            # Update target network
            if agent.steps_done % 1000 == 0 and hasattr(agent, 'update_target_net'):
                agent.update_target_net()

        scores.append(score)
        epsilon = max(eps_end, epsilon * eps_decay)
        
        agent.log("Reward/Train", score)

        # Mise à jour barre de progression
        mean_score = np.mean(scores[-50:])
        pbar.set_postfix({'score': score, 'mean': f"{mean_score:.1f}", 'eps': f"{epsilon:.2f}"})

        # --- Evaluation Intermédiaire ---
        if episode > 0 and episode % 50 == 0:
            # Mode silencieux pour l'eval intermédiaire
            eval_score = evaluate_process(env, agent, n_episodes=3, render=False)
            agent.log("Reward/Eval", eval_score)
            
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                # On sauvegarde avec un nom générique pour simplifier le chargement
                agent.save("best_model") 

    # Sauvegarde finale
    agent.save("final_model")
    print("Entraînement terminé.")

# --- FONCTION D'EVALUATION ---
def evaluate_process(env, agent, n_episodes=5, render=True):
    agent.set_mode("eval") # Désactive l'exploration
    scores = []
    
    delay = 0.05 if render else 0
    iterator = range(n_episodes)
    
    if render:
        print(f"--- Lancement de l'évaluation visuelle ({n_episodes} parties) ---")
        iterator = tqdm(iterator, desc="Evaluation")

    for _ in iterator:
        obs, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
            if render:
                env.render()
                time.sleep(delay)
        
        scores.append(score)

    agent.set_mode("train")
    return np.mean(scores)

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_config = get_model_config()

    # Création du dossier checkpoints s'il n'existe pas
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # 1. Config Environnement
    env_obs_type = 'image' if args.obs == 'image' else 'vector'
    no_food_steps = None if args.no_food_steps <= 0 else args.no_food_steps
    
    # Env temporaire pour dimensions
    tmp_env = SnakeEnv(
        obs_type=env_obs_type,
        n_obstacles=args.obstacles,
        max_no_food_steps=no_food_steps,
        step_penalty=args.step_penalty,
    )
    input_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    # 2. Agent Init
    run_name = f"{args.algo}_{args.obs}"
    if args.algo == "DQN":
        agent = DQNAgent(input_shape, n_actions, device, run_name, lr=args.lr, batch_size=args.batch_size, model_config=model_config)
    elif args.algo == "DDQN":
        agent = DDQNAgent(input_shape, n_actions, device, run_name, lr=args.lr, batch_size=args.batch_size, model_config=model_config)

    # 3. Execution
    
    # --- PHASE ENTRAINEMENT ---
    if args.mode in ["train", "all"]:
        if args.resume:
            resume_name = args.resume_model[:-4] if args.resume_model.endswith(".pth") else args.resume_model
            resume_path = f"checkpoints/{resume_name}.pth"
            if os.path.exists(resume_path):
                agent.load(resume_name)
                print(f"Reprise de l'entrainement depuis : {resume_name}.pth")
            else:
                print(f"Checkpoint introuvable ({resume_path}). Entrainement depuis zero.")

        # Choix du rendu : 'human' si demandé, sinon None (plus rapide)
        train_render_mode = "human" if args.render_train else None
        
        train_env = SnakeEnv(
            render_mode=train_render_mode,
            obs_type=env_obs_type,
            n_obstacles=args.obstacles,
            max_no_food_steps=no_food_steps,
            step_penalty=args.step_penalty,
        )
        
        try:
            train_process(train_env, agent, args)
        except KeyboardInterrupt:
            print("\nArrêt manuel ! Sauvegarde du modèle interrompu...")
            agent.save("interrupted_model")
        finally:
            train_env.close()

    # --- PHASE EVALUATION ---
    if args.mode in ["eval", "all"]:
        print("\nPréparez-vous à regarder le résultat...")
        eval_env = SnakeEnv(
            render_mode="human",
            obs_type=env_obs_type,
            n_obstacles=args.obstacles,
            max_no_food_steps=no_food_steps,
            step_penalty=args.step_penalty,
        )
        
        # Chargement du modèle
        model_name = args.model
        if args.mode == "eval":
            model_name_no_ext = model_name[:-4] if model_name.endswith(".pth") else model_name
            if not os.path.exists(f"checkpoints/{model_name_no_ext}.pth"):
                print(f"Attention: {model_name_no_ext}.pth introuvable. Essai avec final_model.pth...")
                model_name_no_ext = "final_model"
        else:
            model_name_no_ext = "best_model"

        try:
            agent.load(model_name_no_ext)
            print(f"Modele charge : {model_name_no_ext}.pth")
        except Exception as exc:
            print(f"Aucun modele trouve ou erreur de chargement: {exc}. Utilisation d'un agent aleatoire.")

        mean_score = evaluate_process(eval_env, agent, n_episodes=args.eval_episodes, render=True)
        print(f"Score moyen final : {mean_score}")
        eval_env.close()