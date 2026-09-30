"""
cliente_puente.py — Cliente del puente G1 para la aplicación LSC UDI.

Implementa exactamente la interfaz que ConectorG1 espera de `_sdk`:
    init(ip, puerto), set_control_mode(modo), set_joint_positions(dict, speed=..., duracion=...),
    get_state(), close()
y por debajo habla por TCP/JSON con robot_sim/puente_g1.py, que corre en
Linux (WSL en casa, Ubuntu en la oficina, o el PC conectado al G1 real).

La app no necesita el SDK de Unitree instalado: solo este archivo.
"""

import json
import logging
import socket
import threading

log = logging.getLogger("lsc_bridge.cliente_puente")


class ClientePuenteG1:
    def __init__(self, timeout: float = 5.0):
        self._sock = None
        self._f = None
        self._lock = threading.Lock()
        self._timeout = timeout

    # ── Interfaz esperada por ConectorG1 ─────────────────────────────
    def init(self, ip: str, puerto: int):
        try:
            self._sock = socket.create_connection((ip, int(puerto)), timeout=self._timeout)
            self._sock.settimeout(self._timeout)
            self._f = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        except OSError as e:
            raise ConnectionError(f"No se pudo conectar al puente G1 en {ip}:{puerto}: {e}")

        resp = self._enviar({"op": "ping"})
        if not resp.get("ok"):
            raise ConnectionError(f"El puente respondió con error: {resp}")
        log.info(f"Puente G1 conectado en {ip}:{puerto} (robot listo: {resp.get('listo')})")

    def set_control_mode(self, modo: str):
        # El puente solo trabaja en posición articular; se acepta por compatibilidad.
        log.debug(f"set_control_mode({modo}) — ignorado por el puente")

    def set_joint_positions(self, joints: dict, speed: float = 1.0, duracion: float = None):
        # `speed` es la velocidad normalizada de ConectorG1 (0-1). Si no llega
        # duración explícita, se deriva: a más velocidad, menos tiempo.
        if duracion is None:
            duracion = max(0.3, 1.0 / max(float(speed), 0.1))
        resp = self._enviar({"op": "set_joint_positions",
                             "joints": joints, "duracion": float(duracion)})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "error desconocido del puente"))
        return resp

    def get_state(self) -> dict:
        resp = self._enviar({"op": "get_state"})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "error de telemetría"))
        return resp

    def reposo(self, duracion: float = 1.5):
        return self._enviar({"op": "reposo", "duracion": duracion})

    def close(self):
        try:
            if self._f:
                self._enviar({"op": "close"})
        except Exception:
            pass
        finally:
            for obj in (self._f, self._sock):
                try:
                    obj.close()
                except Exception:
                    pass
            self._f = self._sock = None

    # ── Interno ───────────────────────────────────────────────────────
    def _enviar(self, msg: dict) -> dict:
        if self._f is None:
            raise ConnectionError("Puente G1 no conectado")
        with self._lock:
            self._f.write(json.dumps(msg) + "\n")
            self._f.flush()
            linea = self._f.readline()
        if not linea:
            raise ConnectionError("El puente cerró la conexión")
        return json.loads(linea)
