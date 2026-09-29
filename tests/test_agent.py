import numpy as np

from arcagent import ExplorerAgent
from arcagent.mock_env import ClickGame, MazeGame
from arcagent.runner import play
from arcagent.vision import background_color, click_candidates, components


# ---- vision ---------------------------------------------------------------
def test_components_and_candidates():
    g = np.zeros((16, 16), dtype=np.uint8)
    g[2:4, 2:4] = 2
    g[10:12, 8:10] = 3
    assert background_color(g) == 0
    assert len(components(g, skip_color=0)) == 2
    pts = click_candidates(g)
    assert pts[0] in [(2, 2), (3, 2), (2, 3), (3, 3)] or pts[0][0] in (8, 9)
    assert len(pts) == len(set(pts))
    assert all(0 <= x < 16 and 0 <= y < 16 for x, y in pts)


# ---- end-to-end on mock games ----------------------------------------------
def test_solves_maze():
    res = play(MazeGame(), ExplorerAgent(max_actions=2000))
    assert res["state"] == "WIN" and res["levels_completed"] == 3


def test_solves_maze_with_shrinking_hud_bar():
    res = play(MazeGame(hud_bar=True), ExplorerAgent(max_actions=3000))
    assert res["state"] == "WIN"


def test_solves_click_game_near_optimally():
    g = ClickGame()
    res = play(g, ExplorerAgent(max_actions=500))
    assert res["state"] == "WIN"
    assert res["actions"] <= 2 * sum(g.optimal)


def test_deterministic_given_seed():
    a = play(MazeGame(), ExplorerAgent(seed=3, max_actions=2000))
    b = play(MazeGame(), ExplorerAgent(seed=3, max_actions=2000))
    assert a == b


def test_never_repeats_a_fatal_action():
    """After a GAME_OVER the fatal (state, action) edge must not be tried again."""
    game = MazeGame()
    agent = ExplorerAgent(max_actions=2000)
    deaths = []
    orig = agent._observe_death

    def spy():
        if agent.pending is not None:
            deaths.append((agent.cur_raw, agent.pending[1], agent.score))
        orig()

    agent._observe_death = spy
    play(game, agent)
    assert len(deaths) == len(set(deaths))


def test_stops_at_action_budget():
    res = play(MazeGame(), ExplorerAgent(max_actions=10))
    assert res["actions"] == 10


# ---- robustness ----------------------------------------------------------
def test_accepts_dict_frames_and_layered_frames():
    agent = ExplorerAgent()
    grid = np.zeros((8, 8), dtype=int).tolist()
    m = agent.choose_action([], {"frame": [grid, grid], "state": "NOT_PLAYED", "score": 0, "available_actions": [1, 2]})
    assert m.action_id == 0  # must RESET first
    m = agent.choose_action([], {"frame": [grid], "state": "NOT_FINISHED", "score": 0, "available_actions": [1, 2]})
    assert m.action_id in (1, 2)


def test_garbage_frame_degrades_to_random_not_crash():
    agent = ExplorerAgent()
    m = agent.choose_action([], {"frame": "garbage", "state": "NOT_FINISHED", "available_actions": [1, 6]})
    assert m.reason == "fallback" and agent.errors == 1


def test_no_available_actions_listed_uses_default():
    agent = ExplorerAgent()
    grid = np.zeros((8, 8), dtype=int).tolist()
    m = agent.choose_action([], {"frame": [grid], "state": "NOT_FINISHED", "score": 0})
    assert 1 <= m.action_id <= 5


def test_win_ends_run():
    agent = ExplorerAgent()
    grid = np.zeros((4, 4), dtype=int).tolist()
    frame = {"frame": [grid], "state": "WIN", "score": 3}
    agent.choose_action([], frame)
    assert agent.is_done([], frame)


def test_big_board_with_many_decoys_is_solved_quickly():
    from arcagent.mock_big import Big

    res = play(Big(njunk=150), ExplorerAgent(max_actions=500))
    assert res["state"] == "WIN" and res["actions"] <= 40
