"""
reproducir_sena.py — Fase 4 del entorno de simulación LSC UDI.

Reproduce una o varias señas en el Unitree G1 (simulador o robot real)
a partir de las posturas definidas en robot/conector_g1.py (POSTURAS_G1),
enviando LowCmd por DDS con el SDK2 de Unitree.

Uso (simulador, robot colgado con banda elástica):
    python reproducir_sena.py Hola
    python reproducir_sena.py Hola Gracias Si --pausa 0.8

Uso (robot real):
    python reproducir_sena.py Hola --dominio 0 --interfaz eth0 --real

Flujo por seña: interpola desde la postura actual hasta la postura de la
seña en `comando.duracion` segundos, la mantiene `--pausa` segundos, y al
terminar la lista vuelve a la postura de reposo.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Raíz del proyecto (LSC-Mafe/) para importar robot/conector_g1.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.conector_g1 import POSTURAS_G1, ComandoArticular  # noqa: E402

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,  # noqa: E402
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402

G1_NUM_MOTOR = 29
MODE_PR = 0

KP = [60, 60, 60, 100, 40, 40,
      60, 60, 60, 100, 40, 40,
      60, 40, 40,
      40, 40, 40, 40, 40, 40, 40,
      40, 40, 40, 40, 40, 40, 40]
KD = [1, 1, 1, 2, 1, 1,
      1, 1, 1, 2, 1, 1,
      1, 1, 1,
      1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1]

# ── Mapeo ComandoArticular -> índice de motor del G1 (29 DOF) ─────────────
MAPEO = {
    "hombro_izq_pitch": 15, "hombro_izq_roll": 16, "codo_izq": 18,
    "muñeca_izq_pitch": 20, "muñeca_izq_yaw": 21,
    "hombro_der_pitch": 22, "hombro_der_roll": 23, "codo_der": 25,
    "muñeca_der_pitch": 27, "muñeca_der_yaw": 28,
}

# Signo por campo. Convención de POSTURAS_G1: pitch negativo = brazo al
# frente (coincide con el G1). En roll/yaw el brazo derecho del G1 es espejo
# del izquierdo; si al probar el brazo derecho se mueve al revés de lo
# esperado, cambia aquí el signo y anótalo en docs/instalacion_sim.md.
SIGNOS = {
    "hombro_izq_pitch": +1, "hombro_izq_roll": +1, "codo_izq": +1,
    "muñeca_izq_pitch": +1, "muñeca_izq_yaw": +1,
    "hombro_der_pitch": +1, "hombro_der_roll": -1, "codo_der": +1,
    "muñeca_der_pitch": +1, "muñeca_der_yaw": -1,
}

INDICES_BRAZOS = sorted(MAPEO.values())


def comando_a_objetivos(cmd: ComandoArticular) -> dict:
    """Convierte un ComandoArticular en {índice_motor: ángulo_rad}."""
    return {idx: SIGNOS[campo] * getattr(cmd, campo) for campo, idx in MAPEO.items()}


class ControlG1:
    """Mantiene la postura del G1 a 500 Hz y permite mover joints con
    interpolación suave. Misma interfaz para simulador y robot real."""

    def __init__(self, dt: float = 0.002):
        self.dt = dt
        self.low_state = None
        self.mode_machine = 0
        self.listo = False
        self.postura_base = None

        self._lock = threading.Lock()
        self._q_from = None
        self._q_to = None
        self._t0 = 0.0
        self._dur = 1.0
        self._q_des = None

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()

    # ── Arranque ──────────────────────────────────────────────────────
    def iniciar(self):
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_lowstate, 10)

        print("Esperando LowState del robot...")
        while not self.listo:
            time.sleep(0.1)

        q0 = np.array(self.postura_base)
        q0[:15] = 0.0  # piernas y cintura rectas (pelvis anclada al mundo)
        self._q_from = q0.copy()
        self._q_to = q0.copy()
        self._q_des = q0.copy()
        self._t0 = time.time()
        self._hilo = RecurrentThread(interval=self.dt, target=self._tick, name="control")
        self._hilo.Start()
        print("Control activo (500 Hz).")

    def _on_lowstate(self, msg: LowState_):
        self.low_state = msg
        if not self.listo:
            self.mode_machine = msg.mode_machine
            self.postura_base = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.listo = True

    # ── Bucle de control ──────────────────────────────────────────────
    def _tick(self):
        with self._lock:
            s = min((time.time() - self._t0) / self._dur, 1.0)
            ratio = 0.5 * (1.0 - np.cos(np.pi * s))  # perfil coseno
            self._q_des = self._q_from + ratio * (self._q_to - self._q_from)
            q_des = self._q_des

        self.low_cmd.mode_pr = MODE_PR
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1
            m.tau = 0.0
            m.dq = 0.0
            m.kp = KP[i]
            m.kd = KD[i]
            m.q = float(q_des[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # ── API pública ───────────────────────────────────────────────────
    def ir_a(self, objetivos: dict, duracion: float, esperar: bool = True):
        """Lleva los joints indicados a sus ángulos en `duracion` s.
        Los joints no indicados conservan su consigna actual. Bloqueante."""
        with self._lock:
            self._q_from = self._q_des.copy()
            self._q_to = self._q_des.copy()
            for idx, q in objetivos.items():
                self._q_to[idx] = q
            self._dur = max(duracion, 0.05)
            self._t0 = time.time()
        if esperar:
            time.sleep(self._dur)

    def reposo(self, duracion: float = 1.5):
        """Brazos a la postura en que estaba el robot al arrancar."""
        self.ir_a({i: self.postura_base[i] for i in INDICES_BRAZOS}, duracion)

    def q_actual(self, idx: int) -> float:
        return self.low_state.motor_state[idx].q

    def reporte(self, objetivos: dict) -> str:
        errs = [np.degrees(self.q_actual(i) - q) for i, q in objetivos.items()]
        return f"error máx {max(abs(e) for e in errs):4.1f}°"


def main():
    p = argparse.ArgumentParser(description="Reproducir señas LSC en el G1")
    p.add_argument("senas", nargs="*", help="nombres de señas de POSTURAS_G1, ej. Hola Gracias")
    p.add_argument("--pausa", type=float, default=0.5, help="segundos manteniendo cada seña")
    p.add_argument("--interfaz", default="lo")
    p.add_argument("--dominio", type=int, default=1, help="1 = simulador, 0 = robot real")
    p.add_argument("--real", action="store_true",
                   help="liberar el controlador de alto nivel (solo robot físico)")
    p.add_argument("--listar", action="store_true", help="listar señas disponibles y salir")
    args = p.parse_args()

    if args.listar:
        print("Señas disponibles:", ", ".join(sorted(POSTURAS_G1.keys())))
        return

    desconocidas = [s for s in args.senas if s not in POSTURAS_G1]
    if desconocidas:
        print(f"Señas sin postura definida: {desconocidas}")
        print("Disponibles:", ", ".join(sorted(POSTURAS_G1.keys())))
        sys.exit(1)

    if args.real:
        input("ROBOT REAL: asegúrate de que no hay obstáculos. Enter para continuar...")

    ChannelFactoryInitialize(args.dominio, args.interfaz)

    if args.real:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result and result.get("name"):
            msc.ReleaseMode()
            _, result = msc.CheckMode()
            time.sleep(1)

    ctrl = ControlG1()
    ctrl.iniciar()

    try:
        for i, nombre in enumerate(args.senas, 1):
            cmd = POSTURAS_G1[nombre]
            objetivos = comando_a_objetivos(cmd)
            print(f"[{i}/{len(args.senas)}] {nombre}  ({cmd.duracion:.1f} s)")
            ctrl.ir_a(objetivos, cmd.duracion)
            time.sleep(args.pausa)
            print(f"      alcanzada: {ctrl.reporte(objetivos)}")
        print("Volviendo a reposo...")
        ctrl.reposo()
        print("Listo. Ctrl+C para salir (el robot mantiene la postura).")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    main()
