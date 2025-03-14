from textwrap import dedent
from typing import Any, Mapping, Optional, List, Dict, Tuple

from flask import Flask

from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

# plugin identifying constants (in extra module to avoid circular dependencies)
_plugin_name = "es-optimizer"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__) # full identifier including version


# just importing the plugin class creates the plugin instance in EsOptimizer.instance
class EsOptimizer(QHAnaPluginBase):
    """An optimizer using an evolutionary strategy."""

    name = _plugin_name
    version = __version__
    description = "An optimizer using an evolutionary strategy."
    tags = ["ES", "optimizer"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        from .api import PLUGIN_BLP # import blueprint only after plugin instance was created
        return PLUGIN_BLP

    def get_requirements(self):
        return "pymcdm==1.0.5\nscipy~=1.10\nscikit-learn==1.1\nplotly~=5.18.0\npandas~=1.5.0"
