"""Captura de áudio ao vivo para análise contínua (RF04–RF07 da APS).

O detector foi construído para arquivos, mas o cenário do cliente é uma chamada
em andamento. Em vez de pedir o áudio ao Microsoft Teams — o que exigiria um bot
de mídia em C#/.NET, hospedagem em Azure e consentimento de administrador do
tenant — este módulo captura a **saída de áudio do sistema** (loopback). É o que
sai da caixa de som, então funciona com Teams, Meet, Zoom ou qualquer outro,
sem publicar aplicativo nenhum.
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
