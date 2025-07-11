import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ChainReactionEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, size=6):
        self.size = size
        self.action_space = spaces.Discrete(self.size * self.size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.size, self.size), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.owner = np.zeros((self.size, self.size), dtype=int)  # 0 = no one, 1 = p1, 2 = p2
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        obs[0] = self.board / 6.0  # Normalize
        obs[1] = self.owner / 2.0  # Normalize
        obs[2] = np.full((self.size, self.size), self.current_player / 2.0)
        return obs
    
    def action_masks(self):
        # Return a 1D np.array of booleans or 0/1s
        # True or 1 for valid actions, False or 0 for invalid ones
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            if self.is_valid_action(a):
                mask[a] = True        
        return mask
    
    def is_valid_action(self, action):
        x, y = divmod(action, self.size)

        # Invalid move penalty
        if self.owner[x, y] not in [0, self.current_player]:
            return False
        
        return True
        
    def step(self, action):
        if self.done:
            raise Exception("Game over")

        x, y = divmod(action, self.size)

        # Invalid move penalty
        if self.owner[x, y] not in [0, self.current_player]:
            return self._get_obs(), -1.0, False, False, {}

        self._apply_move(x, y)

        # Check end condition
        player_cells = np.sum(self.owner == self.current_player)
        opponent_cells = np.sum((self.owner > 0) & (self.owner != self.current_player))

        if opponent_cells == 0 and player_cells > 1:
            self.done = True
            return self._get_obs(), 1.0, True, False, {}

        # Swap player
        self.current_player = 3 - self.current_player  # 1 ↔ 2
        return self._get_obs(), 0.0, False, False, {}

    def _apply_move(self, x, y):
        self.board[x, y] += 1
        self.owner[x, y] = self.current_player
        self._check_chain_reaction(x, y)

    def _check_chain_reaction(self, x, y):
        if self.board[x, y] <= 6:
            return

        self.board[x, y] = 1

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.board[nx, ny] += 1
                self.owner[nx, ny] = self.current_player
                self._check_chain_reaction(nx, ny)

    def render(self):
        print("Player:", self.current_player)
        for i in range(self.size):
            row = []
            for j in range(self.size):
                val = self.board[i, j]
                owner = self.owner[i, j]
                char = f"{val}"
                if owner == 1:
                    char = f"\033[91m{val}\033[0m"  # red
                elif owner == 2:
                    char = f"\033[94m{val}\033[0m"  # blue
                row.append(char)
            print(" ".join(row))
        print()

    def close(self):
        pass
