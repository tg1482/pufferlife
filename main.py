"""Game of Life with PufferLib - Retro Raylib Visualization"""

import numpy as np
import gymnasium
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import pufferlib
from pufferlib.ocean import render
import pufferlib.emulation
import pufferlib.vector


@dataclass
class GameOfLifeRule:
    """Conway's Game of Life rule representation"""

    name: str
    birth: List[int]  # Neighbor counts for birth
    survival: List[int]  # Neighbor counts for survival
    color: Tuple[int, int, int]  # RGB color


# Predefined rules with sick retro colors
RULES = {
    "Conway": GameOfLifeRule("Conway's Game of Life", [3], [2, 3], (0, 255, 85)),
    "HighLife": GameOfLifeRule("HighLife", [3, 6], [2, 3], (255, 85, 255)),
    "Seeds": GameOfLifeRule("Seeds", [2], [], (255, 255, 85)),
    "Coral": GameOfLifeRule("Coral", [3], [4, 5, 6, 7, 8], (255, 85, 85)),
    "Maze": GameOfLifeRule("Maze", [3], [1, 2, 3, 4, 5], (85, 255, 255)),
    "Replicator": GameOfLifeRule(
        "Replicator", [1, 3, 5, 7], [1, 3, 5, 7], (85, 255, 85)
    ),
    "DayNight": GameOfLifeRule(
        "Day & Night", [3, 6, 7, 8], [3, 4, 6, 7, 8], (170, 85, 255)
    ),
}

# Retro Game of Life colors - proper 80s aesthetic
RETRO_COLORS = np.array(
    [
        [6, 24, 24],  # Deep background
        [0, 255, 85],  # Neon green - alive cells
        [255, 85, 255],  # Hot pink - recently born
        [255, 255, 85],  # Electric yellow - dying
        [85, 255, 255],  # Cyan - stable
        [255, 85, 85],  # Red - overcrowded
        [85, 255, 85],  # Light green - survivors
        [170, 85, 255],  # Purple - special
    ],
    dtype=np.uint8,
)


class GameOfLifePufferEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        width=128,
        height=128,
        rule_name="Conway",
        density=0.3,
        max_steps=10_000_000,
        render_mode="human",
        buf=None,
        seed=0,
        rl_mode: str = "rule",  # "rule" (user-chosen fixed rule) or "learn" (agent learns masks)
        reward_mode: str = "default",  # "default" or "stability_entropy"
    ):

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_rule = RULES[rule_name]
        self.step_count = 0
        self.generation = 0
        self.density = density
        # Auto-evolution interval (lower = faster). Adjustable in render with -/=
        self.auto_step_interval = 5
        self.paused = False
        self.draw_mode = False
        self.rl_mode = rl_mode
        self.reward_mode = reward_mode

        # Observation: flattened grid + rule info + stats
        obs_size = width * height + 8  # grid + rule params + stats
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )

        # Actions
        if self.rl_mode == "learn":
            # 18 binary decisions (9 for birth, 9 for survival)
            self.single_action_space = gymnasium.spaces.MultiDiscrete([2] * 18)
            # initialize with Conway
            self.learn_birth_mask = np.zeros(9, dtype=np.uint8)
            self.learn_survival_mask = np.zeros(9, dtype=np.uint8)
            for b in [3]:
                self.learn_birth_mask[b] = 1
            for s in [2, 3]:
                self.learn_survival_mask[s] = 1
        else:
            # "rule" mode: agent does nothing; rule is chosen externally/user. Single no-op action.
            self.single_action_space = gymnasium.spaces.Discrete(1)

        self.num_agents = num_envs
        super().__init__(buf)

        # Initialize grids for each environment
        self.grids = np.zeros((num_envs, height, width), dtype=np.uint8)
        self.prev_grids = np.zeros((num_envs, height, width), dtype=np.uint8)
        self.rule_indices = np.zeros(num_envs, dtype=np.int32)
        self.generations = np.zeros(num_envs, dtype=np.int32)
        self.live_counts = np.zeros(num_envs, dtype=np.int32)

        # Map rule names to indices and set initial index
        rule_name_to_index = {name: idx for idx, name in enumerate(RULES.keys())}
        initial_idx = rule_name_to_index.get(rule_name, 0)
        self.rule_indices[:] = initial_idx

        # Initialize renderer if in human mode
        self.renderer = None
        if render_mode == "human":
            self.renderer = render.GridRender(
                width,
                height,
                screen_width=1200,
                screen_height=800,
                colors=RETRO_COLORS,
                fps=60,
                name="🧬 GAME OF LIFE - RETRO MODE 🧬",
            )

        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset all environments
        for i in range(self.num_agents):
            self.grids[i] = np.random.choice(
                [0, 1],
                size=(self.height, self.width),
                p=[1 - self.density, self.density],
            ).astype(np.uint8)
            self.prev_grids[i] = self.grids[i].copy()
            self.generations[i] = 0

        self.step_count = 0
        self._update_observations()

        return self.observations, [{}] * self.num_agents

    def step(self, actions):
        for i, action in enumerate(actions):
            if self.rl_mode == "learn":
                # Expect action as length-18 vector of 0/1
                a = np.asarray(action).astype(np.int32)
                birth_mask = a[:9]
                survival_mask = a[9:18]
                self.learn_birth_mask = birth_mask
                self.learn_survival_mask = survival_mask
                # Update current rule for display/info
                birth_list = [k for k in range(9) if birth_mask[k] == 1]
                survival_list = [k for k in range(9) if survival_mask[k] == 1]
                self.current_rule = GameOfLifeRule(
                    name="Learned",
                    birth=birth_list,
                    survival=survival_list,
                    color=(170, 85, 255),
                )

        # Auto-evolve every few steps for natural progression
        if (not self.paused) and (
            self.step_count % self.auto_step_interval == 0
        ):  # Auto-evolve
            for i in range(self.num_agents):
                self.prev_grids[i] = self.grids[i].copy()
                self.grids[i] = self._evolve_grid(self.grids[i])
                self.generations[i] += 1
                self.live_counts[i] = np.sum(self.grids[i])

        self.step_count += 1

        # Calculate rewards based on selected mode
        if self.reward_mode == "stability_entropy":
            self._calculate_rewards_stability_entropy()
        else:
            self._calculate_rewards()

        # Episode termination
        terminated = (
            False if self.max_steps is None else self.step_count >= self.max_steps
        )
        self.terminals[:] = terminated
        self.truncations[:] = False

        self._update_observations()

        info = [
            {
                "generation": self.generations[i],
                "live_cells": self.live_counts[i],
                "rule": self.current_rule.name,
                "density": self.live_counts[i] / (self.width * self.height),
            }
            for i in range(self.num_agents)
        ]

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def _evolve_grid(self, grid):
        """Apply Game of Life rules with current rule set"""
        new_grid = np.zeros_like(grid)

        # Pad grid for neighbor counting
        padded = np.pad(grid, 1, mode="constant", constant_values=0)

        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                # Count neighbors
                neighbors = np.sum(padded[i - 1 : i + 2, j - 1 : j + 2]) - padded[i, j]

                if self.rl_mode == "learn":
                    # Use learned masks
                    if padded[i, j] == 1:  # Alive cell
                        if self.learn_survival_mask[neighbors] == 1:
                            new_grid[i - 1, j - 1] = 1
                    else:  # Dead cell
                        if self.learn_birth_mask[neighbors] == 1:
                            new_grid[i - 1, j - 1] = 1
                else:
                    if padded[i, j] == 1:  # Alive cell
                        if neighbors in self.current_rule.survival:
                            new_grid[i - 1, j - 1] = 1
                    else:  # Dead cell
                        if neighbors in self.current_rule.birth:
                            new_grid[i - 1, j - 1] = 1

        return new_grid

    def _calculate_rewards(self):
        """Reward interesting patterns and diversity"""
        for i in range(self.num_agents):
            reward = 0.0

            # Reward for maintaining life
            live_count = self.live_counts[i]
            if live_count > 0:
                reward += 0.1

            # Reward for pattern changes (evolution)
            if not np.array_equal(self.grids[i], self.prev_grids[i]):
                reward += 0.2

            # Penalty for stagnation or death
            if live_count == 0:
                reward -= 0.5
            elif np.array_equal(self.grids[i], self.prev_grids[i]):
                reward -= 0.1

            # Bonus for optimal density (not too sparse, not too dense)
            density = live_count / (self.width * self.height)
            if 0.1 <= density <= 0.4:
                reward += 0.1

            self.rewards[i] = reward

    def _calculate_rewards_stability_entropy(self):
        eps = 1e-8
        N = self.width * self.height
        for i in range(self.num_agents):
            grid = self.grids[i]
            prev = self.prev_grids[i]
            alive = np.sum(grid)
            alive_frac = alive / N
            change_rate = np.count_nonzero(grid ^ prev) / N
            p1 = alive_frac
            p0 = 1.0 - p1
            H_global = -(p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps))

            # Lightweight block entropy approximation (3x3 neighborhoods)
            padded = np.pad(grid, 1, mode="constant")
            counts = np.zeros(512, dtype=np.int32)
            for r in range(1, padded.shape[0] - 1):
                for c in range(1, padded.shape[1] - 1):
                    block = padded[r - 1 : r + 2, c - 1 : c + 2]
                    idx = 0
                    bit = 1
                    # little-endian bit pack of 3x3
                    for br in range(3):
                        for bc in range(3):
                            if block[br, bc]:
                                idx |= bit
                            bit <<= 1
                    counts[idx] += 1
            hist = counts / np.maximum(1, counts.sum())
            H_block = -np.sum(hist * np.log2(hist + eps))

            # Clustering proxy 1: edge agreement ratio (4-neighborhood)
            vert_same = (grid[1:, :] == grid[:-1, :]).sum()
            horz_same = (grid[:, 1:] == grid[:, :-1]).sum()
            total_edges = (self.height - 1) * self.width + self.height * (
                self.width - 1
            )
            cluster_edges = (vert_same + horz_same) / max(1, total_edges)

            # Clustering proxy 2: mean alive neighbors (8-neighborhood), normalized
            if alive > 0:
                P = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
                P[1:-1, 1:-1] = grid
                neigh = (
                    P[0:-2, 0:-2]
                    + P[0:-2, 1:-1]
                    + P[0:-2, 2:]
                    + P[1:-1, 0:-2]
                    + P[1:-1, 2:]
                    + P[2:, 0:-2]
                    + P[2:, 1:-1]
                    + P[2:, 2:]
                )
                alive_neigh_mean = (
                    float(neigh[grid == 1].mean()) if (grid == 1).any() else 0.0
                )
                cluster_alive = alive_neigh_mean / 8.0
            else:
                cluster_alive = 0.0

            # Target density around 50%
            density_term = 1.0 - abs(alive_frac - 0.5) / 0.5  # in [0,1], peak at 0.5

            # Compose reward
            reward = 0.0
            reward += 0.4 * density_term
            reward += 0.3 * (1.0 - change_rate)  # stability
            reward += 0.2 * cluster_edges
            reward += 0.1 * cluster_alive
            reward -= 0.25 * (H_block / 9.0)  # discourage high local entropy
            self.rewards[i] = float(reward)

    def _update_observations(self):
        """Update observation space with grid + metadata"""
        for i in range(self.num_agents):
            # Flatten grid
            grid_flat = self.grids[i].flatten()

            # Rule parameters
            rule_info = np.array(
                [
                    self.rule_indices[i],
                    len(self.current_rule.birth),
                    len(self.current_rule.survival),
                    self.generations[i] % 256,  # Keep it in uint8 range
                    self.live_counts[i] % 256,
                    int(self.density * 100),
                    self.step_count % 256,
                    0,  # Reserved
                ],
                dtype=np.uint8,
            )

            # Combine
            self.observations[i] = np.concatenate([grid_flat, rule_info])

    def render(self):
        """Retro visualization with rule display"""
        if self.renderer is None:
            return None

        # Use first environment for rendering
        display_grid = self.grids[0].copy()

        # Add some visual flair - color coding based on cell states
        # 0 = dead, 1 = alive, 2-7 = special states for visual appeal
        for i in range(display_grid.shape[0]):
            for j in range(display_grid.shape[1]):
                if display_grid[i, j] == 1:
                    # Color based on neighbor count for visual variety
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % display_grid.shape[0], (
                                j + dj
                            ) % display_grid.shape[1]
                            neighbors += display_grid[ni, nj]

                    # Map neighbors to colors for retro effect
                    if neighbors <= 1:
                        display_grid[i, j] = 5  # Red - lonely
                    elif neighbors in [2, 3]:
                        display_grid[i, j] = 1  # Green - stable
                    elif neighbors >= 4:
                        display_grid[i, j] = 2  # Pink - crowded

        frame = self.renderer.render(display_grid, end_drawing=False)

        # Overlay rule info and stats (retro style)
        from raylib import rl

        # Rule name and generation
        rule_text = f"RULE: {self.current_rule.name}".encode()
        gen_text = f"GEN: {self.generations[0]:05d}".encode()
        pop_text = f"POP: {self.live_counts[0]:05d}".encode()
        density_text = (
            f"DENSITY: {self.live_counts[0]/(self.width*self.height)*100:.1f}%".encode()
        )

        # Retro green text overlay
        puff_text = [0, 255, 85, 255]
        rl.DrawText(rule_text, 10, self.renderer.screen_height - 120, 24, puff_text)
        rl.DrawText(gen_text, 10, self.renderer.screen_height - 95, 20, puff_text)
        rl.DrawText(pop_text, 10, self.renderer.screen_height - 70, 20, puff_text)
        rl.DrawText(density_text, 10, self.renderer.screen_height - 45, 20, puff_text)

        # Rule change instructions
        instructions = [
            "1-7: CHANGE RULE",
            "R: RESET",
            "SPACE: STEP",
            "B: BLANK CANVAS",
            "P: PAUSE/RESUME",
            "LMB/RMB: DRAW/ERASE",
            "ESC: EXIT",
        ]

        for i, instruction in enumerate(instructions):
            rl.DrawText(
                instruction.encode(),
                self.renderer.screen_width - 200,
                10 + i * 25,
                18,
                [255, 255, 85, 255],
            )

        # Handle input for interactive rule changes
        if rl.IsKeyPressed(rl.KEY_ONE):
            self.current_rule = RULES["Conway"]
            self.rule_indices[0] = 0
        elif rl.IsKeyPressed(rl.KEY_TWO):
            self.current_rule = RULES["HighLife"]
            self.rule_indices[0] = 1
        elif rl.IsKeyPressed(rl.KEY_THREE):
            self.current_rule = RULES["Seeds"]
            self.rule_indices[0] = 2
        elif rl.IsKeyPressed(rl.KEY_FOUR):
            self.current_rule = RULES["Coral"]
            self.rule_indices[0] = 3
        elif rl.IsKeyPressed(rl.KEY_FIVE):
            self.current_rule = RULES["Maze"]
            self.rule_indices[0] = 4
        elif rl.IsKeyPressed(rl.KEY_SIX):
            self.current_rule = RULES["Replicator"]
            self.rule_indices[0] = 5
        elif rl.IsKeyPressed(rl.KEY_SEVEN):
            self.current_rule = RULES["DayNight"]
            self.rule_indices[0] = 6
        elif rl.IsKeyPressed(rl.KEY_R):
            self.grids[0] = np.random.choice(
                [0, 1],
                size=(self.height, self.width),
                p=[1 - self.density, self.density],
            ).astype(np.uint8)
            self.generations[0] = 0
        elif rl.IsKeyPressed(rl.KEY_SPACE):
            self.grids[0] = self._evolve_grid(self.grids[0])
            self.generations[0] += 1

        # Blank canvas and pause controls
        if rl.IsKeyPressed(rl.KEY_B):
            self.grids[0][:] = 0
            self.prev_grids[0][:] = 0
            self.generations[0] = 0
            self.live_counts[0] = 0
            self.draw_mode = True
            self.paused = True
        if rl.IsKeyPressed(rl.KEY_P):
            self.paused = not self.paused

        # Speed control: '=' speeds up (decrease interval), '-' slows down (increase interval)
        if rl.IsKeyPressed(rl.KEY_EQUAL):
            self.auto_step_interval = max(1, self.auto_step_interval - 1)
        if rl.IsKeyPressed(rl.KEY_MINUS):
            self.auto_step_interval = min(120, self.auto_step_interval + 1)

        # Show speed control hint and current speed
        speed_text = f"SPEED: every {self.auto_step_interval} steps ( - / = )".encode()
        rl.DrawText(speed_text, 10, self.renderer.screen_height - 20, 18, puff_text)

        # Mouse drawing (click/drag) mapped through camera to grid
        mouse_pos = rl.GetMousePosition()
        world_pos = rl.GetScreenToWorld2D(mouse_pos, self.renderer.camera)
        j = int(world_pos.x)
        i = int(world_pos.y)
        if 0 <= i < self.height and 0 <= j < self.width:
            if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_LEFT):
                self.grids[0, i, j] = 1
                display_grid[i, j] = 1
                self.live_counts[0] = int(np.sum(self.grids[0]))
            elif rl.IsMouseButtonDown(rl.MOUSE_BUTTON_RIGHT):
                self.grids[0, i, j] = 0
                display_grid[i, j] = 0
                self.live_counts[0] = int(np.sum(self.grids[0]))

        rl.EndDrawing()
        return frame

    def close(self):
        if self.renderer:
            from raylib import rl

            rl.CloseWindow()


if __name__ == "__main__":
    # Demo the retro Game of Life
    print("🧬 INITIALIZING RETRO GAME OF LIFE 🧬")
    print("Rules available:", list(RULES.keys()))

    env = pufferlib.vector.make(
        GameOfLifePufferEnv,
        num_envs=1,
        backend=pufferlib.vector.Serial,
        env_kwargs={
            "width": 64,
            "height": 64,
            "rule_name": "Conway",
            "density": 0.3,
            "render_mode": "human",
            "max_steps": None,
        },
    )

    print("Environment created! Starting simulation...")
    print("\nControls:")
    print("1-7: Change rules")
    print("R: Reset grid")
    print("SPACE: Manual step")
    print("ESC: Exit")
    print("WASD: Navigate, Q/E: Zoom")

    observations, infos = env.reset()

    # Run the visualization
    step = 0
    start_time = time.time()

    try:
        while True:
            # Single valid action in 'rule' mode is no-op (index 0)
            actions = [0]

            # # Occasionally inject some randomness
            # if step % 100 == 0:
            #     actions = [
            #         np.random.randint(0, env.driver_env.width * env.driver_env.height)
            #     ]

            observations, rewards, terminals, truncations, infos = env.step(actions)

            # Render the retro visualization
            frame = env.driver_env.render()

            step += 1

            if terminals[0] or truncations[0]:
                observations, infos = env.reset()
                step = 0
                print(f"Episode finished! Restarting...")

            # Print stats every 60 steps
            if step % 60 == 0:
                elapsed = time.time() - start_time
                gen = (
                    getattr(env.driver_env, "generations", [0])[0]
                    if hasattr(env.driver_env, "generations")
                    else 0
                )
                live = (
                    getattr(env.driver_env, "live_counts", [0])[0]
                    if hasattr(env.driver_env, "live_counts")
                    else 0
                )
                rule = getattr(
                    getattr(env.driver_env, "current_rule", None), "name", "N/A"
                )
                print(
                    f"Step: {step}, Gen: {gen}, "
                    f"Live: {live}, "
                    f"Rule: {rule}, "
                    f"FPS: {step/elapsed:.1f}"
                )

    except KeyboardInterrupt:
        print("\n🧬 Shutting down Game of Life... 🧬")
    finally:
        env.driver_env.close()
