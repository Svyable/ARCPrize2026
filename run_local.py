"""Run the agent on the bundled mock games: python run_local.py"""
from arcagent import ExplorerAgent
from arcagent.mock_env import ClickGame, MazeGame
from arcagent.runner import play, score

games = {"maze": MazeGame(), "maze+hud": MazeGame(hud_bar=True), "click": ClickGame()}
for name, game in games.items():
    res = play(game, ExplorerAgent(seed=0, max_actions=3000))
    print(f"{name:9s} state={res['state']:9s} levels={res['levels_completed']}/{game.n_levels} "
          f"actions={res['actions']:4d} per_level={res['per_level']} "
          f"score={score(res, game.optimal):.2f} (baseline=optimal path)")
