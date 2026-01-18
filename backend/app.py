# ============================================
# BACKEND: Python Flask API with PPO (Stable-Baselines3)
# File: backend/app.py
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Warehouse Configuration
GRID_SIZE = 20
CELL_TYPES = {
    'EMPTY': 0,
    'SHELF': 1,
    'OBSTACLE': 2,
    'ROBOT': 3
}

class WarehouseEnv(gym.Env):
    """Custom Gymnasium Environment for Warehouse Navigation with PPO"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, grid, start_pos, target_pos, avoided_shelves=None):
        super(WarehouseEnv, self).__init__()
        
        self.grid = np.array(grid)
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.current_pos = self.start_pos.copy()
        self.avoided_shelves = avoided_shelves or set()
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [robot_x, robot_y, target_x, target_y, dx, dy, distance]
        # Enhanced observation for better PPO learning
        self.observation_space = spaces.Box(
            low=-GRID_SIZE, high=GRID_SIZE, shape=(7,), dtype=np.float32
        )
        
        self.max_steps = GRID_SIZE * 3
        self.current_step = 0
        self.visited_positions = set()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.current_step = 0
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.current_pos))
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Enhanced observation with relative position and distance"""
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        return np.array([
            self.current_pos[0],
            self.current_pos[1],
            self.target_pos[0],
            self.target_pos[1],
            dx,
            dy,
            distance
        ], dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        # Action mapping
        moves = [
            (0, -1),  # UP
            (0, 1),   # DOWN
            (-1, 0),  # LEFT
            (1, 0)    # RIGHT
        ]
        
        dx, dy = moves[action]
        new_pos = self.current_pos + np.array([dx, dy], dtype=np.float32)
        
        # Calculate previous distance
        prev_dist = np.linalg.norm(self.current_pos - self.target_pos)
        
        # Initialize reward
        reward = 0
        terminated = False
        truncated = False
        
        # Check if move is valid
        if self._is_valid_move(int(new_pos[0]), int(new_pos[1])):
            self.current_pos = new_pos
            
            # Calculate new distance
            new_dist = np.linalg.norm(self.current_pos - self.target_pos)
            
            # Reward shaping for PPO
            # Strong reward for getting closer
            distance_reward = (prev_dist - new_dist) * 5
            reward += distance_reward
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.1
            
            # Penalty for revisiting positions (encourage exploration)
            pos_tuple = tuple(self.current_pos)
            if pos_tuple in self.visited_positions:
                reward -= 0.5
            else:
                self.visited_positions.add(pos_tuple)
                reward += 0.2  # Small bonus for exploring new cells
            
            # Check if reached target
            if np.allclose(self.current_pos, self.target_pos, atol=0.1):
                reward += 200  # Large reward for reaching goal
                terminated = True
        else:
            # Strong penalty for invalid moves
            reward = -15
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            # Penalty for not reaching target in time
            reward -= 50
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _is_valid_move(self, x, y):
        # Check boundaries
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return False
        
        # Check obstacles
        if self.grid[int(y)][int(x)] == CELL_TYPES['OBSTACLE']:
            return False
        
        # Check avoided shelves
        if f"{int(x)},{int(y)}" in self.avoided_shelves:
            return False
        
        return True
    
    def render(self):
        pass


def train_ppo_agent(env, total_timesteps=30000, verbose=0):
    """
    Train PPO agent on the warehouse environment
    Optimized for M1 Mac 8GB RAM
    """
    # PPO hyperparameters optimized for speed and performance
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,           # Reduced for memory efficiency
        batch_size=64,         # Smaller batch for M1 8GB
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Encourage exploration
        verbose=verbose,
        device='cpu',          # Use CPU (better for M1)
        tensorboard_log=None
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    return model


def find_path_with_ppo(env, model, max_steps=None):
    """Use trained PPO model to find path"""
    if max_steps is None:
        max_steps = env.max_steps
    
    obs, _ = env.reset()
    path = [env.current_pos.copy().tolist()]
    
    for _ in range(max_steps):
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        path.append(env.current_pos.copy().tolist())
        
        if terminated:
            break
        
        if truncated:
            # If truncated, try to get closer anyway
            break
    
    return path


def generate_warehouse():
    """Generate warehouse layout with shelves and obstacles"""
    grid = [[CELL_TYPES['EMPTY'] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    shelves = []
    
    # Create organized shelf rows
    shelf_id = 0
    for row in range(2, GRID_SIZE - 2, 4):
        for col in range(2, GRID_SIZE - 2, 3):
            if random.random() > 0.1:
                grid[row][col] = CELL_TYPES['SHELF']
                grid[row][col + 1] = CELL_TYPES['SHELF']
                
                shelf_name = f"SHELF_{chr(65 + shelf_id)}"
                shelves.append({
                    'id': shelf_name,
                    'x': col,
                    'y': row,
                    'skus': [f"SKU{shelf_id * 3 + i + 1}" for i in range(3)]
                })
                shelf_id += 1
    
    # Add random obstacles
    for _ in range(8):
        x = random.randint(2, GRID_SIZE - 3)
        y = random.randint(2, GRID_SIZE - 3)
        if grid[y][x] == CELL_TYPES['EMPTY']:
            grid[y][x] = CELL_TYPES['OBSTACLE']
    
    return grid, shelves


# Store warehouse state
warehouse_state = {
    'grid': None,
    'shelves': None
}


@app.route('/api/warehouse/generate', methods=['GET'])
def generate_warehouse_endpoint():
    """Generate new warehouse layout"""
    grid, shelves = generate_warehouse()
    warehouse_state['grid'] = grid
    warehouse_state['shelves'] = shelves
    
    return jsonify({
        'grid': grid,
        'shelves': shelves
    })


@app.route('/api/warehouse/optimize', methods=['POST'])
def optimize_path():
    """Optimize path using PPO reinforcement learning"""
    data = request.json
    selected_skus = data.get('selectedSKUs', [])
    avoided_shelves = set(data.get('avoidedShelves', []))
    
    if not warehouse_state['grid'] or not warehouse_state['shelves']:
        return jsonify({'error': 'Warehouse not initialized'}), 400
    
    if not selected_skus:
        return jsonify({'error': 'No SKUs selected'}), 400
    
    # Find target positions for selected SKUs
    targets = []
    for sku in selected_skus:
        for shelf in warehouse_state['shelves']:
            if sku in shelf['skus']:
                targets.append({'x': shelf['x'], 'y': shelf['y'], 'sku': sku})
                break
    
    if not targets:
        return jsonify({'error': 'No valid targets found'}), 400
    
    # Find optimal path through all targets using PPO
    full_path = []
    current_pos = [0, 0]  # Start position
    training_info = []
    
    for idx, target in enumerate(targets):
        target_pos = [target['x'], target['y']]
        
        # Create environment
        env = WarehouseEnv(
            warehouse_state['grid'],
            current_pos,
            target_pos,
            avoided_shelves
        )
        
        # Verify environment is valid
        try:
            check_env(env, warn=True)
        except Exception as e:
            print(f"Environment check warning: {e}")
        
        # Train PPO model
        # Reduced timesteps for faster training on M1
        timesteps = 20000 if idx == 0 else 15000  # First target needs more training
        model = train_ppo_agent(env, total_timesteps=timesteps, verbose=0)
        
        # Find path using trained model
        segment_path = find_path_with_ppo(env, model)
        
        # Add to full path (avoid duplicates)
        if full_path:
            full_path.extend(segment_path[1:])
        else:
            full_path = segment_path
        
        current_pos = target_pos
        
        training_info.append({
            'target': target['sku'],
            'timesteps': timesteps,
            'path_length': len(segment_path)
        })
    
    # Convert numpy arrays to regular Python lists
    full_path = [[float(x), float(y)] for x, y in full_path]
    
    return jsonify({
        'path': full_path,
        'totalSteps': len(full_path),
        'itemsCount': len(targets),
        'trainingInfo': training_info,
        'algorithm': 'PPO (Proximal Policy Optimization)'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'algorithm': 'PPO',
        'framework': 'stable-baselines3'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🤖 Warehouse RL Path Optimizer with PPO")
    print("=" * 60)
    print("Algorithm: PPO (Proximal Policy Optimization)")
    print("Framework: Stable-Baselines3")
    print("Environment: Custom Gymnasium")
    print("Server: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=8080)