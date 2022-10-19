from http import HTTPStatus

from flask.helpers import url_for
from flask.views import MethodView

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
)
from qhana_plugin_runner.api.util import SecurityBlueprint

from .plugin import EsOptimizer

# Blueprint to register API endpoints with
PLUGIN_BLP = SecurityBlueprint( #SecurityBlueprint for eventual JWT support
    EsOptimizer.instance.identifier,  # blueprint name
    __name__,  # module import name!
    description="ES-Optimizer plugin API.",
)


@PLUGIN_BLP.route("/")
class PluginView(MethodView):
    """Root resource of this plugin."""

    @PLUGIN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            # human readable title and description
            title=EsOptimizer.instance.name,
            description=PLUGIN_BLP.description,
            # machine-readable identifying name and version
            name=EsOptimizer.instance.identifier,
            version=EsOptimizer.instance.version,
            # pluin type: "processing"|"visualizing"|"conversion" (actual values still WIP!! confirm with current documentation)
            type=PluginType.simple,
            # tags describing the plugin, e.g. ml:autoencoder, ml:svm
            tags=[],
            # the main plugin entry point
            entry_point=EntryPoint(
                # entry point for headless (non-gui) applications
                href=url_for(f"{PLUGIN_BLP.name}.ProcessView"),
                # micro frontend entry point
                ui_href="",
                # definition of (required) input data
                data_input=[],
                # definition of output data
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
        )
