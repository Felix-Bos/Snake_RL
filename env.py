import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from enum import Enum

class Direction(Enum):
    """
    Enumeration representing the four possible movement directions.
    """
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeEnv(gym.Env):
    """
    Custom Environment that follows the Gymnasium interface for the Snake game.
    
    This environment supports two observation types:
    1. 'vector': A 1-D vector of size 11 containing relative information (dangers, direction, food).
    2. 'image': A 3-D matrix (1, Height, Width) representing the grid state visually.

    Attributes:
        metadata (dict): specific metadata for the gym environment (render modes, fps).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, grid_size=20, width=640, height=480, obs_type='vector', n_obstacles=10,
                 max_no_food_steps=None, step_penalty=0.01, fixed_obstacles=None):
        """
        Initializes the Snake Environment.

        Args:
            render_mode (str, optional): The render mode ('human' or 'rgb_array'). Defaults to None.
            grid_size (int, optional): The size of one grid cell in pixels. Defaults to 20.
            width (int, optional): Window width in pixels. Defaults to 640.
            height (int, optional): Window height in pixels. Defaults to 480.
            obs_type (str, optional): Type of observation to return ('vector' or 'image'). Defaults to 'vector'.
            n_obstacles (int, optional): Number of static walls/obstacles to place randomly. Defaults to 10.
            fixed_obstacles (list, optional): List of [x, y] cells to use as a fixed obstacle
                layout instead of random placement. Defaults to None (random placement).
        """
        super(SnakeEnv, self).__init__()

        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.n_obstacles = n_obstacles
        self.step_penalty = abs(float(step_penalty))
        self.fixed_obstacles = fixed_obstacles
        
        # Calculate grid dimensions (e.g., 32x24)
        self.cols = width // grid_size
        self.rows = height // grid_size
        self.max_no_food_steps = max_no_food_steps if max_no_food_steps is not None else (self.cols * self.rows * 2)

        # Action Space: 0=Right, 1=Down, 2=Left, 3=Up
        self.action_space = spaces.Discrete(4)

        # --- DEFINITION OF OBSERVATION SPACE ---
        if self.obs_type == 'image':
            # IMAGE: Matrix (1 channel, Height, Width)
            # Values: 0=Empty, 1=Body, 2=Head, 3=Food, 4=Wall
            self.observation_space = spaces.Box(
                low=0, high=4, shape=(1, self.rows, self.cols), dtype=np.uint8
            )
        else:
            # VECTOR: 11 boolean/float values (Danger L/S/R, Current Dir, Food Pos)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(11,), dtype=np.float32
            )

        self.window = None
        self.clock = None
        
        self.snake = []
        self.head = []
        self.food = []
        self.obstacles = []
        self.direction = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        This includes placing the snake in the middle, generating obstacles,
        placing food, and resetting score/steps.

        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options (unused).

        Returns:
            tuple: (observation, info)
                - observation (np.array): The initial observation (vector or image).
                - info (dict): Diagnostic information (empty for now).
        """
        super().reset(seed=seed)
        
        # 1. Init Snake at the center
        self.direction = Direction.RIGHT
        self.head = [self.cols // 2, self.rows // 2]
        self.snake = [self.head, 
                      [self.head[0]-1, self.head[1]], 
                      [self.head[0]-2, self.head[1]]]
        
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_steps = (self.cols * self.rows) * 100
        
        # 2. Place Obstacles and Food
        self._place_obstacles()
        self._place_food()
        
        if self.render_mode == "human":
            self._init_pygame()
            
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The direction to move (0=Right, 1=Down, 2=Left, 3=Up).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (np.array): The next state of the game.
                - reward (float): The reward obtained (+10 food, -10 death, etc.).
                - terminated (bool): True if the snake died.
                - truncated (bool): True if max_steps reached.
                - info (dict): Diagnostic information.
        """
        self.steps += 1
        self.steps_since_food += 1
        
        # Direction Logic (Prevent 180-degree turns)
        if action == 0 and self.direction != Direction.LEFT: self.direction = Direction.RIGHT
        elif action == 1 and self.direction != Direction.UP: self.direction = Direction.DOWN
        elif action == 2 and self.direction != Direction.RIGHT: self.direction = Direction.LEFT
        elif action == 3 and self.direction != Direction.DOWN: self.direction = Direction.UP

        # Movement
        x, y = self.head
        if self.direction == Direction.RIGHT: x += 1
        elif self.direction == Direction.LEFT: x -= 1
        elif self.direction == Direction.DOWN: y += 1
        elif self.direction == Direction.UP: y -= 1
        
        self.head = [x, y]
        
        # Check Collisions
        reward = 0
        terminated = False
        truncated = False
        
        # Collision with Wall, Self, or Obstacle
        if self._is_collision(self.head):
            terminated = True
            reward = -10
        else:
            self.snake.insert(0, self.head)
            
            # Check Food
            if self.head == self.food:
                self.score += 1
                reward = 10
                self.steps_since_food = 0
                self._place_food()
            else:
                self.snake.pop()
                reward = -self.step_penalty

        # Truncated if infinite loop or max steps reached
        if self.steps > self.max_steps:
            truncated = True
            reward = -5

        # Anti-loop: coupe les episodes qui tournent en rond sans atteindre de nourriture.
        if self.steps_since_food >= self.max_no_food_steps and not terminated:
            truncated = True
            reward = min(reward, -5)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Constructs the observation based on 'obs_type'.

        Returns:
            np.array: 
                - If 'image': shape (1, Rows, Cols), values 0-4.
                - If 'vector': shape (11,), values 0 or 1.
        """
        # --- CASE 1: IMAGE MODE (For CNN) ---
        if self.obs_type == 'image':
            grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
            
            # Draw Snake body
            for pt in self.snake:
                if 0 <= pt[1] < self.rows and 0 <= pt[0] < self.cols:
                    grid[pt[1], pt[0]] = 1  
            
            # Draw Head
            if 0 <= self.head[1] < self.rows and 0 <= self.head[0] < self.cols:
                grid[self.head[1], self.head[0]] = 2 

            # Draw Food
            if 0 <= self.food[1] < self.rows and 0 <= self.food[0] < self.cols:
                grid[self.food[1], self.food[0]] = 3 
            
            # Draw Obstacles
            for pt in self.obstacles:
                 if 0 <= pt[1] < self.rows and 0 <= pt[0] < self.cols:
                    grid[pt[1], pt[0]] = 4
            
            # Add channel dimension for PyTorch: (Channels, Height, Width)
            return np.expand_dims(grid, axis=0)
            
        # --- CASE 2: VECTOR MODE (11 values) ---
        else:
            head = self.snake[0]
            
            # Adjacent points to check for collisions
            point_l = [head[0] - 1, head[1]]
            point_r = [head[0] + 1, head[1]]
            point_u = [head[0], head[1] - 1]
            point_d = [head[0], head[1] + 1]
            
            dir_l = self.direction == Direction.LEFT
            dir_r = self.direction == Direction.RIGHT
            dir_u = self.direction == Direction.UP
            dir_d = self.direction == Direction.DOWN

            state = [
                # 1-3. DANGERS (Straight, Right, Left)
                (dir_r and self._is_collision(point_r)) or 
                (dir_l and self._is_collision(point_l)) or 
                (dir_u and self._is_collision(point_u)) or 
                (dir_d and self._is_collision(point_d)),

                (dir_u and self._is_collision(point_r)) or 
                (dir_d and self._is_collision(point_l)) or 
                (dir_l and self._is_collision(point_u)) or 
                (dir_r and self._is_collision(point_d)),

                (dir_d and self._is_collision(point_r)) or 
                (dir_u and self._is_collision(point_l)) or 
                (dir_r and self._is_collision(point_u)) or 
                (dir_l and self._is_collision(point_d)),
                
                # 4-7. CURRENT DIRECTIONS
                dir_l, dir_r, dir_u, dir_d,
                
                # 8-11. FOOD DIRECTION
                self.food[0] < head[0], # Food Left
                self.food[0] > head[0], # Food Right
                self.food[1] < head[1], # Food Up
                self.food[1] > head[1]  # Food Down
            ]
            
            return np.array(state, dtype=np.float32)

    def _is_collision(self, pt):
        """
        Checks if a point collides with the boundary, the snake body, or an obstacle.

        Args:
            pt (list): The [x, y] coordinate to check.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # Wall bounds
        if pt[0] >= self.cols or pt[0] < 0 or pt[1] >= self.rows or pt[1] < 0:
            return True
        # Self collision (ignoring head if checking future move)
        if pt in self.snake[1:]:
            return True
        # Obstacle collision
        if pt in self.obstacles:
            return True
        return False

    def _place_obstacles(self):
        """Randomly places `n_obstacles` on the grid, avoiding the snake and duplicates.

        If `fixed_obstacles` was provided at construction time, that layout is used
        as-is (clipped to the current grid bounds) instead of a random placement.
        """
        if self.fixed_obstacles is not None:
            self.obstacles = [
                [x, y] for x, y in self.fixed_obstacles
                if 0 <= x < self.cols and 0 <= y < self.rows
            ]
            return

        self.obstacles = []
        for _ in range(self.n_obstacles):
            while True:
                x = random.randint(0, self.cols - 1)
                y = random.randint(0, self.rows - 1)
                if [x, y] not in self.snake and [x, y] not in self.obstacles:
                    if abs(x - self.head[0]) + abs(y - self.head[1]) > 2:
                        self.obstacles.append([x, y])
                        break

    def _place_food(self):
        """Randomly places food on the grid, avoiding snake and obstacles."""
        while True:
            x = random.randint(0, self.cols - 1)
            y = random.randint(0, self.rows - 1)
            if [x, y] not in self.snake and [x, y] not in self.obstacles:
                self.food = [x, y]
                break

    def _init_pygame(self):
        """Initializes the Pygame window and clock."""
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Snake RL")
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _render_frame(self):
        """Updates the Pygame display with the current game state."""
        if not pygame.get_init() or not pygame.display.get_init():
            self._init_pygame()
        elif self.window is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill((0, 0, 0))
        
        # Draw Obstacles (Grey)
        for pt in self.obstacles:
            pygame.draw.rect(self.window, (100, 100, 100), 
                             pygame.Rect(pt[0]*self.grid_size, pt[1]*self.grid_size, self.grid_size, self.grid_size))

        # Draw Snake (Green)
        for pt in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0), 
                             pygame.Rect(pt[0]*self.grid_size, pt[1]*self.grid_size, self.grid_size, self.grid_size))
        
        # Draw Head (Cyan)
        pygame.draw.rect(self.window, (0, 255, 255), 
                         pygame.Rect(self.head[0]*self.grid_size, self.head[1]*self.grid_size, self.grid_size, self.grid_size))

        # Draw Food (Red)
        pygame.draw.rect(self.window, (255, 0, 0), 
                         pygame.Rect(self.food[0]*self.grid_size, self.food[1]*self.grid_size, self.grid_size, self.grid_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        """Gymnasium render API."""
        if self.render_mode == "human":
            self._render_frame()
            return None

        if self.render_mode == "rgb_array":
            if self.window is None:
                self._init_pygame()
            self._render_frame()
            frame = pygame.surfarray.array3d(self.window)
            return np.transpose(frame, (1, 0, 2))

        return None

    def close(self):
        """Closes the Pygame window and quits the application."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            
            
if __name__ == "__main__":
    env = SnakeEnv(render_mode="human", obs_type='image', n_obstacles=15)
    
    epoch = 100
    
    for i in range(epoch):
        obs, info = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # print(obs) # Commented out to avoid flooding the console
            done = terminated or truncated

    env.close()