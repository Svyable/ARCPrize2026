import logging, sys, numpy as np
from arc_agi import Arcade, OperationMode
logging.disable(logging.CRITICAL)
CH=".123456789abcdef"
def show(g):
    return "\n".join("".join(CH[v] if v else "." for v in row) for row in g)
if __name__=="__main__":
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("q"))
    for gid in sys.argv[1:]:
        env=arc.make(gid); f=env.observation_space
        print("=====",gid,f.available_actions); print(show(f.frame[-1]))
