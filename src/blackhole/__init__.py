"""event-horizon: a physically based Schwarzschild black hole ray tracer."""
from .camera import Camera
from .geodesic import B_CRIT, ISCO, PHOTON_SPHERE, trace
from .render import Scene

__all__ = ["Camera", "Scene", "trace", "B_CRIT", "ISCO", "PHOTON_SPHERE"]
__version__ = "0.1.0"
