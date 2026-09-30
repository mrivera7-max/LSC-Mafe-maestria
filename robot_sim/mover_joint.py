"""
mover_joint.py — Fase 3 del entorno de simulación LSC UDI.

Lleva UN joint del Unitree G1 (29 DOF) a un ángulo objetivo con
interpolación suave, manteniendo el resto de joints en su postura actual,
y verifica por LowState que el joint llegó.

Uso (simulador, robot colgado con banda elástica):
    python mover_joint.py --joint LeftShoulderPitch --angulo 60
    python mover_joint.py --joint LeftElbow --angulo 45 --duracion 2

Uso (robot real, cuando llegue el momento):
    python mover_joint.py --joint LeftShoulderPitch --angulo 30 --dominio 0 --interfaz eth0 --real

Por defecto: dominio 1 e interfaz "lo" (los del simulador unitree_mujoco).
"""

import argparse
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

G1_NUM_MOTOR = 29

# Ganancias PD tomadas del ejemplo oficial g1_low_level_example.py
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

# Índices de los joints de brazos (G1 29 DOF). Piernas/cintura no se tocan.
JOINTS = {
    "LeftShoulderPitch": 15, "LeftShoulderRoll": 16, "LeftShoulderYaw": 17,
    "LeftElbow": 18, "LeftWristRoll": 19, "LeftWristPitch": 20, "LeftWristYaw": 21,
    "RightShoulderPitch": 22, "RightShoulderRoll": 23, "RightShoulderYaw": 24,
    "RightElbow": 25, "RightWristRoll": 26, "RightWristPitch": 27, "RightWristYaw": 28,
}

MODE_PR = 0  # control por Pitch/Roll en tobillos y muñecas (el que usa el ejemplo)


class MovedorJoint:
    def __init__(self, idx_joint: int, angulo_rad: float, duracion: float):
        self.idx = idx_joint
        self.q_objetivo = angulo_rad
        self.duracion = duracion
        self.dt = 0.002  # 500 Hz, igual que el ejemplo

        self.t = 0.0
        self.q_inicial = None      # ángulo del joint al arrancar
        self.postura_base = None   # copia de todos los q al arrancar
        self.low_state = None
        self.mode_machine = 0
        self.listo = False

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()

    # ── DDS ──────────────────────────────────────────────────────────────
    def init(self):
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_lowstate, 10)

    def _on_lowstate(self, msg: LowState_):
        self.low_state = msg
        if not self.listo:
            self.mode_machine = msg.mode_machine
            self.postura_base = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.q_inicial = self.postura_base[self.idx]
            self.listo = True

    # ── Control ──────────────────────────────────────────────────────────
    def start(self):
        print("Esperando LowState del robot...")
        while not self.listo:
            time.sleep(0.1)
        print(f"Joint {self.idx}: {np.degrees(self.q_inicial):6.1f}° -> "
              f"{np.degrees(self.q_objetivo):6.1f}° en {self.duracion} s")
        self.hilo = RecurrentThread(interval=self.dt, target=self._escribir_cmd,
                                    name="control")
        self.hilo.Start()

    def _escribir_cmd(self):
        self.t += self.dt
        # Perfil coseno: arranque y llegada suaves, sin picos de velocidad.
        s = min(self.t / self.duracion, 1.0)
        ratio = 0.5 * (1.0 - np.cos(np.pi * s))
        q_des = self.q_inicial + ratio * (self.q_objetivo - self.q_inicial)

        self.low_cmd.mode_pr = MODE_PR
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1          # habilitado
            m.tau = 0.0
            m.dq = 0.0
            m.kp = KP[i]
            m.kd = KD[i]
            m.q = q_des if i == self.idx else self.postura_base[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def q_actual(self) -> float:
        return self.low_state.motor_state[self.idx].q


def main():
    p = argparse.ArgumentParser(description="Mover un joint del G1 a un ángulo")
    p.add_argument("--joint", default="LeftShoulderPitch", choices=JOINTS.keys())
    p.add_argument("--angulo", type=float, default=60.0, help="grados")
    p.add_argument("--duracion", type=float, default=3.0, help="segundos")
    p.add_argument("--interfaz", default="lo")
    p.add_argument("--dominio", type=int, default=1, help="1 = simulador, 0 = robot real")
    p.add_argument("--real", action="store_true",
                   help="liberar el controlador de alto nivel (solo robot físico)")
    args = p.parse_args()

    if args.real:
        input("ROBOT REAL: asegúrate de que no hay obstáculos. Enter para continuar...")

    ChannelFactoryInitialize(args.dominio, args.interfaz)

    if args.real:
        # En el robot físico hay que soltar el controlador interno antes de LowCmd.
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result and result.get("name"):
            msc.ReleaseMode()
            _, result = msc.CheckMode()
            time.sleep(1)

    mov = MovedorJoint(JOINTS[args.joint], np.radians(args.angulo), args.duracion)
    mov.init()
    mov.start()

    # Monitor: ángulo real vs objetivo, cada segundo.
    try:
        for _ in range(int(args.duracion) + 5):
            time.sleep(1.0)
            err = np.degrees(mov.q_actual() - mov.q_objetivo)
            print(f"t={mov.t:4.1f}s  q={np.degrees(mov.q_actual()):6.1f}°  "
                  f"objetivo={args.angulo:6.1f}°  error={err:+5.1f}°")
        print("Fin. El robot queda manteniendo la postura hasta que cierres (Ctrl+C).")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDetenido.")
        sys.exit(0)


if __name__ == "__main__":
    main()
