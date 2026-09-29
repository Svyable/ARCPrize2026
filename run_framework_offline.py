"""Run the Explorer through the real ARC-AGI-3-Agents Swarm on local games (no network).

    python run_framework_offline.py ls20,tn36 [max_actions]

Needs the framework deps (arc-agi, arcengine, langchain, langgraph, smolagents, pillow<=11.3).
"""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ROOT / "environment_files")
os.chdir(ROOT / "ARC-AGI-3-Agents")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from agents import AVAILABLE_AGENTS, Swarm  # noqa: E402

AVAILABLE_AGENTS["explorer"].MAX_ACTIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
card = Swarm("explorer", "http://localhost", sys.argv[1].split(",")).main()
print(f"levels {card.total_levels_completed}/{card.total_levels}  actions {card.total_actions}")
