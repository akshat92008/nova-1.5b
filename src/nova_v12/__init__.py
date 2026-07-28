"""Nova V12: a verified local atomic patch executor and training toolkit."""

from .protocol import parse_response
from .runner import NovaRunner, OllamaBackend
from .schema import AtomicTask, EscalationResponse, PatchResponse

__all__ = [
    "AtomicTask",
    "EscalationResponse",
    "NovaRunner",
    "OllamaBackend",
    "PatchResponse",
    "parse_response",
]

__version__ = "12.0.0.dev0"
