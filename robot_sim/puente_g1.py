"""
puente_g1.py — Fase 5 del entorno LSC UDI.

Puente entre la aplicación de escritorio (Windows o Ubuntu) y el Unitree G1
(simulador MuJoCo o robot real). Escucha comandos JSON por TCP y los ejecuta
con ControlG1 (SDK2 / DDS).

Uso (simulador, en WSL o Ubuntu con `sim` corriendo en otra terminal):
    python puente_g1.py
Uso (robot real):
    python puente_g1.py --dominio 0 --interfaz eth0 --real

Protocolo: una línea JSON por mensaje, respuesta una línea JSON.
    {"op": "ping"}
    {"op": "set_joint_positions", "joints": {"right_shoulder_pitch": -0.5, ...},
     "duracion": 1.0}
    {"op": "get_state"}
    {"op": "reposo"}
    {"op": "close"}
Los ángulos van en radianes con la convención de robot/conector_g1.py.
"""

import argparse
import json
import socketserver
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproducir_sena import ControlG1, INDICES_BRAZOS  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

# Nombres que usa la app (config.json / ConectorG1) -> índice del G1 y signo.
# Mismos signos que SIGNOS en reproducir_sena.py (brazo derecho espejo en roll/yaw).
JOINTS_APP = {
    "left_shoulder_pitch":  (15, +1), "left_shoulder_roll":  (16, +1),
    "left_shoulder_yaw":    (17, +1), "left_elbow":          (18, +1),
    "left_wrist_roll":      (19, +1), "left_wrist_pitch":    (20, +1),
    "left_wrist_yaw":       (21, +1),
    "right_shoulder_pitch": (22, +1), "right_shoulder_roll": (23, -1),
    "right_shoulder_yaw":   (24, -1), "right_elbow":         (25, +1),
    "right_wrist_roll":     (26, -1), "right_wrist_pitch":   (27, +1),
    "right_wrist_yaw":      (28, -1),
}
IDX_A_NOMBRE = {idx: n for n, (idx, _) in JOINTS_APP.items()}

ctrl: ControlG1 = None  # se crea en main()


def _estado() -> dict:
    ls = ctrl.low_state
    joints = {IDX_A_NOMBRE[i]: JOINTS_APP[IDX_A_NOMBRE[i]][1] * ctrl.q_actual(i)
              for i in INDICES_BRAZOS if i in IDX_A_NOMBRE}
    # El simulador no reporta batería; en el robot real estos campos existen.
    return {
        "battery_level": getattr(getattr(ls, "bms_state", None), "soc", 0),
        "temperature":   max((m.temperature[0] for m in ls.motor_state[:29]
                              if hasattr(m, "temperature")), default=0),
        "voltage":       getattr(ls, "power_v", 0.0),
        "current":       getattr(ls, "power_a", 0.0),
        "joints":        joints,
    }


def procesar(msg: dict) -> dict:
    op = msg.get("op")
    if op == "ping":
        return {"ok": True, "robot": "g1", "listo": ctrl.listo}

    if op == "set_joint_positions":
        objetivos = {}
        for nombre, q in msg.get("joints", {}).items():
            if nombre in JOINTS_APP:
                idx, signo = JOINTS_APP[nombre]
                objetivos[idx] = signo * float(q)
        if not objetivos:
            return {"ok": False, "error": "sin joints reconocidos"}
        dur = float(msg.get("duracion", 1.0))
        ctrl.ir_a(objetivos, dur, esperar=False)
        return {"ok": True, "n_joints": len(objetivos), "duracion": dur}

    if op == "get_state":
        return {"ok": True, **_estado()}

    if op == "reposo":
        ctrl.reposo(float(msg.get("duracion", 1.5)))
        return {"ok": True}

    if op == "close":
        ctrl.reposo(1.5)
        return {"ok": True, "bye": True}

    return {"ok": False, "error": f"op desconocida: {op}"}


class Manejador(socketserver.StreamRequestHandler):
    def handle(self):
        print(f"[puente] cliente conectado: {self.client_address[0]}")
        for linea in self.rfile:
            linea = linea.decode("utf-8").strip()
            if not linea:
                continue
            try:
                resp = procesar(json.loads(linea))
            except Exception as e:  # nunca tumbar el puente por un mensaje malo
                resp = {"ok": False, "error": str(e)}
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            if resp.get("bye"):
                break
        print(f"[puente] cliente desconectado: {self.client_address[0]}")


class Servidor(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global ctrl
    p = argparse.ArgumentParser(description="Puente TCP -> Unitree G1 (SDK2)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--puerto", type=int, default=8080)
    p.add_argument("--interfaz", default="lo")
    p.add_argument("--dominio", type=int, default=1, help="1 = simulador, 0 = robot real")
    p.add_argument("--real", action="store_true")
    args = p.parse_args()

    ChannelFactoryInitialize(args.dominio, args.interfaz)

    if args.real:
        input("ROBOT REAL: asegúrate de que no hay obstáculos. Enter para continuar...")
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

    with Servidor((args.host, args.puerto), Manejador) as srv:
        print(f"[puente] escuchando en {args.host}:{args.puerto}  "
              f"(dominio {args.dominio}, interfaz {args.interfaz}). Ctrl+C para salir.")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[puente] cerrando, robot a reposo...")
            ctrl.reposo(1.5)


if __name__ == "__main__":
    main()
