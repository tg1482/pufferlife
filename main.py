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
        max_steps=1_000_000,
        render_mode="human",
        buf=None,
        seed=0,
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

        # Observation: flattened grid + rule info + stats
        obs_size = width * height + 8  # grid + rule params + stats
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=8, shape=(obs_size,), dtype=np.uint8
        )

        # Actions: toggle cell, change rule, reset, step
        self.single_action_space = gymnasium.spaces.Discrete(
            width * height + len(RULES) + 3
        )

        self.num_agents = num_envs
        super().__init__(buf)

        # Initialize grids for each environment
        self.grids = np.zeros((num_envs, height, width), dtype=np.uint8)
        self.prev_grids = np.zeros((num_envs, height, width), dtype=np.uint8)
        self.rule_indices = np.zeros(num_envs, dtype=np.int32)
        self.generations = np.zeros(num_envs, dtype=np.int32)
        self.live_counts = np.zeros(num_envs, dtype=np.int32)

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
            if action < self.width * self.height:
                # Toggle cell
                row = action // self.width
                col = action % self.width
                self.grids[i, row, col] = 1 - self.grids[i, row, col]
            elif action < self.width * self.height + len(RULES):
                # Change rule
                rule_idx = action - (self.width * self.height)
                self.rule_indices[i] = rule_idx
                self.current_rule = list(RULES.values())[rule_idx]
            elif action == self.width * self.height + len(RULES):
                # Reset grid
                self.grids[i] = np.random.choice(
                    [0, 1],
                    size=(self.height, self.width),
                    p=[1 - self.density, self.density],
                ).astype(np.uint8)
                self.generations[i] = 0
            elif action == self.width * self.height + len(RULES) + 1:
                # Single step evolution
                self.grids[i] = self._evolve_grid(self.grids[i])
                self.generations[i] += 1
            # action + 2 is no-op

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

        # Calculate rewards based on interesting patterns
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
            # Mostly no-op actions, let auto-evolution handle progression
            actions = [len(RULES) + 2]  # No-op action

            # Occasionally inject some randomness
            if step % 100 == 0:
                actions = [
                    np.random.randint(0, env.driver_env.width * env.driver_env.height)
                ]

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
