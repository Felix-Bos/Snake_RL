# Snake_RL

Snake_RL est un projet de reinforcement learning pour entrainer un agent a
jouer a Snake avec DQN ou DDQN.

Le projet propose deux manieres de l'utiliser :

- un script CLI `train.py` pour entrainer ou evaluer un agent ;
- un dashboard Django/Channels pour lancer un entrainement depuis le navigateur
  et suivre le snake, les rewards et la loss en temps reel.

## Fonctionnalites

- Environnement Snake custom compatible Gymnasium.
- Agents DQN et DDQN.
- Observations vectorielles ou image.
- Sauvegarde de checkpoints PyTorch.
- Logs TensorBoard.
- Dashboard web avec WebSocket.
- Visualisation du snake en direct.
- Editeur de layouts d'obstacles.

## Prerequis

- Python 3.11 ou plus recent.
- `pip`.
- Un environnement macOS, Linux ou Windows avec support Python/PyTorch.

Sur macOS, le script utilise automatiquement `mps` si PyTorch le detecte.
Sinon, il utilise le CPU.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Le fichier `.env` peut rester tel quel en local.

## Lancer un entrainement CLI

Mode entrainement rapide sans rendu :

```bash
.venv/bin/python train.py --mode train --algo DQN --obs vector --episodes 300
```

Mode entrainement avec affichage pygame :

```bash
.venv/bin/python train.py --mode train --algo DQN --obs vector --episodes 300 --render_train
```

## Evaluer un modele

Charger le meilleur modele disponible :

```bash
.venv/bin/python train.py --mode eval --algo DQN --obs vector --eval_episodes 5 --model best_model
```

Charger le modele final :

```bash
.venv/bin/python train.py --mode eval --algo DQN --obs vector --model final_model
```

Si `best_model.pth` n'existe pas, le script tente automatiquement de charger
`final_model.pth`.

## Lancer train + eval

```bash
.venv/bin/python train.py --mode all --algo DQN --obs vector --episodes 300
```

## TensorBoard

```bash
.venv/bin/python -m tensorboard.main --logdir runs --port 6006
```

Puis ouvrez :

```text
http://localhost:6006
```

## Dashboard web

Le dashboard permet de configurer un entrainement, de suivre les metriques en
direct, de regarder le snake jouer et de creer des cartes d'obstacles.

Premiere utilisation :

```bash
.venv/bin/python manage.py migrate
```

Lancement du serveur ASGI :

```bash
.venv/bin/python -m uvicorn snake_dashboard.asgi:application --reload
```

Puis ouvrez :

```text
http://127.0.0.1:8000/
```

`manage.py runserver` n'est pas recommande pour ce projet, car le dashboard
utilise des WebSockets via Django Channels. Utilisez `uvicorn`.

## Artefacts generes

Ces dossiers/fichiers sont crees pendant l'utilisation et ne doivent pas etre
commites :

- `checkpoints/`
- `runs/`
- `db.sqlite3`
- `obstacle_layouts/*.json`
- `.venv/`
- `.env`

Le dossier `obstacle_layouts/` contient un `.gitkeep` pour garder le dossier
present dans le repo sans publier les layouts locaux.

## Tests et checks

```bash
.venv/bin/python manage.py check
.venv/bin/python manage.py test
.venv/bin/python -m pip check
```

Note : le projet ne contient pas encore de tests unitaires reels. La commande
`manage.py test` valide surtout que la configuration Django se charge
correctement.

## Structure

```text
agents/              Implementations DQN/DDQN
dashboard/           Application Django du dashboard
networks_utils/      Architectures reseau PyTorch
replay_utils/        Replay buffer
snake_dashboard/     Configuration Django/ASGI
env.py               Environnement Snake
train.py             Script CLI d'entrainement/evaluation
requirements.txt     Dependances Python
```
