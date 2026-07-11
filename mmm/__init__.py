"""Sommmelier: AI-driven Marketing Mix Modeling."""

from pkgutil import extend_path

# Meridian's optional protobuf schema also publishes under the ``mmm`` namespace.
# Extend this package path so ``mmm.v1`` and Sommmelier's modules can coexist.
__path__ = extend_path(__path__, __name__)

__version__ = "0.1.0"
