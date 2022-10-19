from .plugin import __version__ # import plugin and version constant
from .endpoints import rank, learn_ranking, sensitivity, learn_prediction   # must happen last to avoid import problems
