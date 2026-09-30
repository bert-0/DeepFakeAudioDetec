"""Captura de áudio ao vivo para análise contínua (RF04–RF07 da APS).

Captura a saída de áudio do sistema (loopback) em vez de integrar com o Teams,
então funciona com qualquer app de chamada.
"""

from .sources import (
    AudioSource,
    CaptureError,
    FileSource,
    WasapiLoopbackSource,
    abrir_fonte,
)
from .stream import JanelaDeslizante

__all__ = ["AudioSource", "CaptureError", "FileSource", "WasapiLoopbackSource",
           "abrir_fonte", "JanelaDeslizante"]
