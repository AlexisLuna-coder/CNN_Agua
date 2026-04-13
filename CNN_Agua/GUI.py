"""
CCN_AGUA  –  Sistema de Análisis de Contaminación Hídrica
==========================================================
Arquitectura (senior):
  • Hilo principal    → Tkinter event-loop  (ÚNICO hilo que toca la UI)
  • CameraThread      → captura frames de OpenCV y los pone en una Queue
  • AnalysisWorker    → ejecuta el análisis local en un ThreadPoolExecutor
  • _poll_queue()     → método periódico (after 30 ms) que consume la Queue
                        y despacha actualizaciones de UI de forma segura.

Reglas de oro aplicadas:
  1. Nunca tocar widgets desde un hilo que no sea el principal.
  2. Comunicación hilo→UI exclusivamente via queue + after().
  3. Events (threading.Event) para señalizar parada de hilos.
  4. ThreadPoolExecutor para análisis: limita concurrencia y reutiliza hilos.
  5. Daemon threads: se destruyen automáticamente al cerrar la app.
  6. Sin dependencias externas de API  →  análisis 100% local (HSV).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable
import sys

from keras.models import load_model
import numpy as np

# ── Dependencias ─────────────────────────────────────────────────────────────
missing = []

try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False
    missing.append("Pillow           →  pip install Pillow")

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    missing.append("opencv-python    →  pip install opencv-python")

if missing:
    print("Faltan dependencias críticas:")
    for m in missing:
        print(f"   {m}")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  MENSAJES ENTRE HILOS  (tipados, sin ambigüedad)
# ════════════════════════════════════════════════════════════════════════════

class MsgType(Enum):
    FRAME        = auto()
    CAM_ERROR    = auto()
    CAM_STOPPED  = auto()
    ANAL_START   = auto()
    ANAL_RESULT  = auto()
    ANAL_ERROR   = auto()
    STATUS       = auto()


@dataclass
class Message:
    type:    MsgType
    payload: object = None


# ════════════════════════════════════════════════════════════════════════════
#  HILO DE CÁMARA
# ════════════════════════════════════════════════════════════════════════════

class CameraThread(threading.Thread):
    """
    Hilo daemon que captura frames de OpenCV y los publica en out_q.
      • stop()    → termina el bucle limpiamente
      • capture() → señaliza que el próximo frame es una captura
    """

    FRAME_INTERVAL = 0.033   # ~30 fps

    def __init__(self, out_q: queue.Queue, device: int = 0):
        super().__init__(daemon=True, name="CameraThread")
        self._q        = out_q
        self._device   = device
        self._stop_evt = threading.Event()
        self._cap_evt  = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def capture(self):
        """Señaliza que el próximo frame debe marcarse como capturado."""
        self._cap_evt.set()

    def run(self):
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            self._q.put(Message(MsgType.CAM_ERROR, "No se pudo abrir la cámara."))
            return
        try:
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                is_capture = self._cap_evt.is_set()
                if is_capture:
                    self._cap_evt.clear()

                self._q.put(Message(
                    MsgType.FRAME,
                    {"frame": frame, "capture": is_capture}
                ))

                elapsed = time.monotonic() - t0
                sleep_t = max(0.0, self.FRAME_INTERVAL - elapsed)
                if sleep_t:
                    time.sleep(sleep_t)
        finally:
            cap.release()
            self._q.put(Message(MsgType.CAM_STOPPED))


# ════════════════════════════════════════════════════════════════════════════
#  MOTOR DE ANÁLISIS DE AGUA
# ════════════════════════════════════════════════════════════════════════════
class AnalizadorCNN:
    def __init__(self):
        self.altura = 200
        self.anchura = 200

        # 🔥 Cargar modelo UNA sola vez
        self.modelo = load_model("CNN_Imagenes/Modelo/cnn.h5")
        self.modelo.load_weights("CNN_Imagenes/Modelo/cnn_pesos.weights.h5")

        # ⚠️ ORDEN EXACTO de clases (IMPORTANTE)
        self.clases = [
            "AGUA_CON_PARTICULAS_ESTANCADA-Suspended_Stanqued_Matter",
            "AGUA_HIDROCARBURO-Hydrocarbons_Pollution",
            "AGUA_LIMPIA-Clean_Water",
            "AGUA_PLASTICO-Surface_Water_Pollution",
            "AGUA_SIN_OXIGENO-Oxygen_Depletion_Pollution",
            "AGUA_SUBMARINA_CONTAMINADA-UNDERWATER_PLASTIC_POLLUTION",
            "AGUA_SUCIA-GROUNDWATER_POLLUTION"
        ]

    def analizar(self, frame_bgr):
        import cv2
        from datetime import datetime

        # 🔄 Convertir BGR → RGB
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 📏 Resize
        frame = cv2.resize(frame, (self.anchura, self.altura))

        # 🔢 Normalizar
        frame = frame / 255.0

        # 📦 Expandir dimensión
        frame = np.expand_dims(frame, axis=0)

        # 🤖 Predicción
        pred = self.modelo.predict(frame)
        idx = np.argmax(pred[0])
        confianza = float(pred[0][idx])

        clase = self.clases[idx]

        # 🎨 COLOR POR CLASE
        colores = {
            "AGUA_LIMPIA-Clean_Water": "#52B788",
            "AGUA_HIDROCARBURO-Hydrocarbons_Pollution": "#E63946",
            "AGUA_PLASTICO-Surface_Water_Pollution": "#F4A261",
            "AGUA_SIN_OXIGENO-Oxygen_Depletion_Pollution": "#E63946",
            "AGUA_CON_PARTICULAS_ESTANCADA-Suspended_Stanqued_Matter": "#FFD166",
            "AGUA_SUBMARINA_CONTAMINADA-UNDERWATER_PLASTIC_POLLUTION": "#F77F00",
            "AGUA_SUCIA-GROUNDWATER_POLLUTION": "#9D4EDD"
        }

        return {
            "nivel": clase,
            "color_ui": colores.get(clase, "#00B4D8"),
            "hallazgos": [
                f"Clasificación: {clase}",
                f"Confianza: {round(confianza * 100, 2)} %"
            ],
            "metricas": {
                "Probabilidad": round(confianza, 4)
            },
            "ts": datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
        }
class AnalizadorAgua:
    """
    Evalúa un frame BGR con métricas del espacio de color HSV.
    Para conectar un modelo real: reemplazar solo _clasificar().
    """

    DIAGNOSTICOS = [
        {
            "nivel":     "CRÍTICO",
            "color_ui":  "#E63946",
            "hallazgos": [
                "AGUA CONTAMINADA POR HIDROCARBURO",
                "RESIDUOS ORGÁNICOS DETECTADOS",
                "pH FUERA DE RANGO SEGURO  (< 5 o > 9)",
            ],
        },
        {
            "nivel":     "ALTO",
            "color_ui":  "#F4A261",
            "hallazgos": [
                "SEDIMENTOS ANÓMALOS DETECTADOS",
                "TURBIDEZ ELEVADA  (NTU > 100)",
                "POSIBLE PRESENCIA DE ALGAS NOCIVAS",
            ],
        },
        {
            "nivel":     "MEDIO",
            "color_ui":  "#FFD166",
            "hallazgos": [
                "LEVE COLORACIÓN IRREGULAR",
                "PARTÍCULAS EN SUSPENSIÓN",
                "MONITOREO CONTINUO RECOMENDADO",
            ],
        },
        {
            "nivel":     "BAJO",
            "color_ui":  "#52B788",
            "hallazgos": [
                "AGUA DENTRO DE PARÁMETROS NORMALES",
                "TRANSPARENCIA ACEPTABLE",
                "SIN CONTAMINANTES VISIBLES",
            ],
        },
    ]

    def analizar(self, frame_bgr) -> dict:
        """Recibe frame BGR; devuelve dict con resultado. ~1.2 s de procesado."""
        time.sleep(1.2)

        hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mean_h = float(hsv[:, :, 0].mean())
        mean_s = float(hsv[:, :, 1].mean())
        mean_v = float(hsv[:, :, 2].mean())

        idx  = self._clasificar(mean_h, mean_s, mean_v)
        diag = self.DIAGNOSTICOS[idx]

        return {
            "nivel":     diag["nivel"],
            "color_ui":  diag["color_ui"],
            "hallazgos": diag["hallazgos"],
            "metricas": {
                "Tono H":       round(mean_h, 1),
                "Saturación S": round(mean_s, 1),
                "Brillo V":     round(mean_v, 1),
            },
            "ts": datetime.now().strftime("%H:%M:%S  %d/%m/%Y"),
        }

    @staticmethod
    def _clasificar(h: float, s: float, v: float) -> int:
        if s > 120 and not (90 < h < 140):
            return 0   # CRÍTICO
        if v < 60:
            return 1   # ALTO
        if s > 60:
            return 2   # MEDIO
        return 3       # BAJO


# ════════════════════════════════════════════════════════════════════════════
#  WORKER DE ANÁLISIS  (ThreadPoolExecutor  –  sin API)
# ════════════════════════════════════════════════════════════════════════════

class AnalysisWorker:
    """
    Envuelve un ThreadPoolExecutor de 1 hilo para análisis secuencial.
    Deposita los resultados en out_q para que el hilo principal los consuma.
    """

    def __init__(self, out_q: queue.Queue):
        self._q     = out_q
        self._pool  = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="AnalWorker")
        self._motor = AnalizadorCNN()
        self._lock  = threading.Lock()

    def submit(self, frame_bgr, filename: str):
        """Envía trabajo al pool. No bloquea el hilo que llama."""
        with self._lock:
            self._q.put(Message(MsgType.ANAL_START))
            self._pool.submit(self._run, frame_bgr.copy(), filename)

    def _run(self, frame_bgr, filename: str):
        try:
            resultado = self._motor.analizar(frame_bgr)
            resultado["filename"] = filename
            self._q.put(Message(MsgType.ANAL_RESULT, resultado))
        except Exception as exc:
            self._q.put(Message(MsgType.ANAL_ERROR, str(exc)))

    def shutdown(self):
        self._pool.shutdown(wait=False)


# ════════════════════════════════════════════════════════════════════════════
#  PALETA  (diseño oscuro original)
# ════════════════════════════════════════════════════════════════════════════

C = {
    "bg":       "#0D1B2A",
    "panel":    "#1B2A3B",
    "btn":      "#1E3A4F",
    "btn_hov":  "#2A5270",
    "accent":   "#00B4D8",
    "accent2":  "#48CAE4",
    "success":  "#52B788",
    "warning":  "#F4A261",
    "danger":   "#E63946",
    "text":     "#E0F4FF",
    "subtext":  "#90C4D8",
    "border":   "#2D4A5E",
    "cam_live": "#1A6B3C",
    "cam_cap":  "#7B2D8B",
}


# ════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class CCNAguaApp(tk.Tk):

    POLL_MS = 30

    def __init__(self):
        super().__init__()

        self.title("CCN_AGUA  │  Detección de Contaminación Hídrica")
        self.geometry("1150x740")
        self.minsize(960, 620)
        self.configure(bg=C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Cola compartida (única vía hilo → UI)
        self._q: queue.Queue = queue.Queue()

        # Workers
        self._cam_thread: Optional[CameraThread] = None
        self._anal_worker = AnalysisWorker(self._q)

        # Estado
        self._frame_actual  = None
        self._cam_running   = False
        self._history: list = []
        self._tk_img        = None

        # UI
        self._build_header()
        self._build_main()
        self._build_statusbar()

        # Atajo de teclado [Q] / [q]  →  captura desde cámara
        self.bind("<q>", lambda _e: self._capture_frame())
        self.bind("<Q>", lambda _e: self._capture_frame())

        # Arrancar polling y reloj
        self._poll_queue()
        self._tick_clock()
        self._status("Listo. Cargue una imagen o active la cámara.  [Q] = capturar")

    # ════════════════════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE LA UI
    # ════════════════════════════════════════════════════════════════════════

    def _build_header(self):
        h = tk.Frame(self, bg=C["panel"], pady=10)
        h.pack(fill="x")
        tk.Label(h, text="💧 CCN_AGUA",
                 font=("Segoe UI", 22, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(side="left", padx=20)
        tk.Label(h, text="Detección Inteligente de Contaminación Hídrica",
                 font=("Segoe UI", 11),
                 fg=C["subtext"], bg=C["panel"]).pack(side="left")
        self._lbl_clock = tk.Label(h, text="",
                                   font=("Segoe UI", 10),
                                   fg=C["subtext"], bg=C["panel"])
        self._lbl_clock.pack(side="right", padx=20)

    def _build_main(self):
        pane = tk.PanedWindow(self, orient="horizontal", bg=C["bg"],
                              sashwidth=6, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(pane, bg=C["bg"])
        pane.add(left, minsize=440, width=500)
        self._build_buttons(left)
        self._build_preview(left)

        right = tk.Frame(pane, bg=C["bg"])
        pane.add(right, minsize=380)
        self._build_results(right)

    def _build_buttons(self, parent):
        bar = tk.Frame(parent, bg=C["panel"], pady=10, padx=10)
        bar.pack(fill="x", pady=(0, 8))

        tk.Label(bar, text="ACCIONES",
                 font=("Segoe UI", 9, "bold"),
                 fg=C["subtext"], bg=C["panel"]).grid(
                     row=0, column=0, columnspan=6,
                     sticky="w", pady=(0, 8))

        defs = [
            ("📷  Cámara",        self._toggle_camera,  C["accent"],   "_btn_cam"),
            ("📸  Capturar [Q]",  self._capture_frame,  C["cam_cap"],  "_btn_cap"),
            ("📂  Subir Archivo", self._upload_file,     C["btn"],      None),
            ("💾  Guardar",       self._save_result,     C["btn"],      None),
            ("🗑  Limpiar",       self._clear_all,       C["btn"],      None),
            ("ℹ️  Info",          self._show_about,      C["btn"],      None),
        ]

        for col, (label, cmd, color, attr) in enumerate(defs):
            btn = tk.Button(bar, text=label, command=cmd,
                            font=("Segoe UI", 10),
                            bg=color, fg=C["text"],
                            activebackground=C["btn_hov"],
                            activeforeground=C["text"],
                            relief="flat", bd=0,
                            padx=10, pady=7, cursor="hand2")
            btn.grid(row=1, column=col, padx=3)
            self._hover(btn, color)
            if attr:
                setattr(self, attr, btn)

        self._btn_cap.configure(state="disabled")

    def _build_preview(self, parent):
        frm = tk.LabelFrame(parent, text=" Vista Previa ",
                            font=("Segoe UI", 10, "bold"),
                            fg=C["accent"], bg=C["panel"],
                            bd=1, relief="flat", labelanchor="nw")
        frm.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(frm, bg="#0A1520",
                                 highlightthickness=1,
                                 highlightbackground=C["border"])
        self._canvas.pack(fill="both", expand=True, padx=6, pady=6)

        bot = tk.Frame(frm, bg=C["panel"])
        bot.pack(fill="x", padx=6, pady=(0, 4))

        self._lbl_live = tk.Label(bot, text="",
                                  font=("Segoe UI", 9, "bold"),
                                  fg=C["cam_live"], bg=C["panel"])
        self._lbl_live.pack(side="left")

        self._lbl_img_info = tk.Label(bot, text="Sin imagen cargada",
                                      font=("Segoe UI", 9),
                                      fg=C["subtext"], bg=C["panel"])
        self._lbl_img_info.pack(side="right")

        self._placeholder()

    def _build_results(self, parent):
        top = tk.Frame(parent, bg=C["bg"])
        top.pack(fill="x", pady=(0, 6))
        tk.Label(top, text="RESULTADO DEL ANÁLISIS",
                 font=("Segoe UI", 11, "bold"),
                 fg=C["accent"], bg=C["bg"]).pack(side="left")

        self._lbl_badge = tk.Label(top, text="  SIN ANALIZAR  ",
                                   font=("Segoe UI", 9, "bold"),
                                   fg=C["text"], bg=C["subtext"],
                                   padx=6, pady=2)
        self._lbl_badge.pack(side="right")

        box = tk.Frame(parent, bg=C["panel"], bd=1, relief="flat",
                       highlightthickness=1,
                       highlightbackground=C["border"])
        box.pack(fill="both", expand=True)

        hdr = tk.Frame(box, bg=C["btn"], pady=8, padx=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🧪 LA IMAGEN MUESTRA:",
                 font=("Segoe UI", 12, "bold"),
                 fg=C["accent2"], bg=C["btn"]).pack(side="left")

        self._txt = scrolledtext.ScrolledText(
            box, font=("Consolas", 11),
            bg="#0A1520", fg=C["text"],
            insertbackground=C["accent"],
            relief="flat", bd=0, wrap="word",
            padx=14, pady=14, state="disabled"
        )
        self._txt.pack(fill="both", expand=True)

        for tag, fg, style in [
            ("titulo",  C["accent2"], ("Segoe UI", 13, "bold")),
            ("seccion", C["subtext"], ("Segoe UI", 10, "bold")),
            ("ok",      C["success"], ("Segoe UI", 11, "bold")),
            ("alerta",  C["warning"], ("Segoe UI", 11, "bold")),
            ("peligro", C["danger"],  ("Segoe UI", 11, "bold")),
            ("normal",  C["text"],    ("Segoe UI", 11)),
            ("fecha",   C["subtext"], ("Consolas", 9)),
            ("spin",    C["accent"],  ("Segoe UI", 11)),
        ]:
            self._txt.tag_configure(tag, foreground=fg, font=style)

        # Historial
        hf = tk.LabelFrame(parent, text=" Historial ",
                           font=("Segoe UI", 9, "bold"),
                           fg=C["subtext"], bg=C["bg"],
                           bd=1, relief="flat", labelanchor="nw")
        hf.pack(fill="x", pady=(8, 0))
        self._lst_hist = tk.Listbox(hf, height=4,
                                    bg=C["panel"], fg=C["subtext"],
                                    selectbackground=C["accent"],
                                    font=("Consolas", 9),
                                    relief="flat", bd=0, activestyle="none")
        self._lst_hist.pack(fill="x", padx=4, pady=4)
        self._lst_hist.bind("<<ListboxSelect>>", self._on_hist_select)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["panel"], pady=4)
        bar.pack(fill="x", side="bottom")
        self._lbl_status = tk.Label(bar, text="",
                                    font=("Segoe UI", 9),
                                    fg=C["subtext"], bg=C["panel"], anchor="w")
        self._lbl_status.pack(side="left", padx=12, fill="x", expand=True)
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self._progress.pack(side="right", padx=12)

    # ════════════════════════════════════════════════════════════════════════
    #  POLLING DE COLA  (hilo principal, cada 30 ms)
    # ════════════════════════════════════════════════════════════════════════

    def _poll_queue(self):
        try:
            while True:
                msg: Message = self._q.get_nowait()
                self._dispatch(msg)
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._poll_queue)

    def _dispatch(self, msg: Message):
        handlers: dict[MsgType, Callable] = {
            MsgType.FRAME:       self._on_frame,
            MsgType.CAM_ERROR:   self._on_cam_error,
            MsgType.CAM_STOPPED: self._on_cam_stopped,
            MsgType.ANAL_START:  self._on_anal_start,
            MsgType.ANAL_RESULT: self._on_anal_result,
            MsgType.ANAL_ERROR:  self._on_anal_error,
            MsgType.STATUS:      lambda m: self._status(m.payload),
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    # ════════════════════════════════════════════════════════════════════════
    #  HANDLERS DE MENSAJES  (todos en hilo principal → UI segura)
    # ════════════════════════════════════════════════════════════════════════

    def _on_frame(self, msg: Message):
        data       = msg.payload
        frame      = data["frame"]
        is_capture = data["capture"]

        self._frame_actual = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if is_capture:
            draw = ImageDraw.Draw(pil)
            for t in range(6):
                draw.rectangle(
                    [t, t, pil.width - t - 1, pil.height - t - 1],
                    outline=(0, 180, 216))
            draw.text((14, 14), "CAPTURADO", fill=(0, 180, 216))

        self._show_pil(pil)

        if is_capture:
            ts = datetime.now().strftime("%H%M%S")
            fname = f"frame_{ts}.jpg"
            self._lbl_img_info.configure(
                text=f"📸 {fname}  │  {pil.width}×{pil.height} px")
            self._lbl_live.configure(text="📸 CAPTURADO")
            self._status("Imagen capturada. Analizando…")
            self._anal_worker.submit(frame, fname)

    def _on_cam_error(self, msg: Message):
        self._cam_running = False
        self._btn_cam.configure(text="📷  Cámara", bg=C["accent"])
        self._btn_cap.configure(state="disabled")
        self._lbl_live.configure(text="")
        messagebox.showerror("Error de cámara", msg.payload)

    def _on_cam_stopped(self, _msg: Message):
        self._cam_running = False
        self._btn_cam.configure(text="📷  Cámara", bg=C["accent"])
        self._btn_cap.configure(state="disabled")
        self._lbl_live.configure(text="")
        self._status("Cámara detenida.")

    def _on_anal_start(self, _msg: Message):
        self._progress.start(10)
        self._lbl_badge.configure(text="  ⏳ ANALIZANDO…  ",
                                  bg=C["warning"], fg="#000")
        self._write("", clear=True)
        self._write("⏳  Analizando muestra, por favor espere…\n\n", "spin")
        self._status("Procesando imagen con motor local…")

    def _on_anal_result(self, msg: Message):
        self._progress.stop()
        res: dict  = msg.payload
        nivel = res["nivel"]
        nivel_mostrar = nivel.replace("_", " ").replace("-", " ")
        color_ui   = res["color_ui"]
        hallazgos  = res["hallazgos"]
        metricas   = res["metricas"]
        ts         = res["ts"]
        filename   = res.get("filename", "imagen")

        badge_txt = f"  {nivel.replace('_',' ')}  "
        badge_bg = color_ui
        self._lbl_badge.configure(text=badge_txt, bg=badge_bg, fg="#fff")

        self._write("", clear=True)
        self._write(f"{ts}\n", "fecha")
        self._write("━" * 54 + "\n", "seccion")
        self._write("LA IMAGEN MUESTRA:\n\n", "titulo")

        tag_nivel = {"CRÍTICO": "peligro", "ALTO": "peligro",
                     "MEDIO": "alerta", "BAJO": "ok"}.get(nivel, "ok")
        self._write(f"  CLASE DETECTADA: {nivel_mostrar}\n", tag_nivel)
        self._write("  " + "─" * 44 + "\n", "seccion")

        for h in hallazgos:
            self._write(f"  •  {h}\n", "normal")

        self._write("\n  MÉTRICAS FÍSICAS\n", "seccion")
        for k, v in metricas.items():
            self._write(f"    {k}: {v}\n", "normal")

        self._write("\n" + "━" * 54 + "\n", "seccion")

        entry = {"ts": ts, "file": filename,
                 "nivel": nivel, "payload": res}
        self._history.append(entry)
        self._lst_hist.insert(
            0, f"[{ts}]  {filename}  →  {badge_txt.strip()}")
        self._status(f"Análisis completado  ·  {ts}")

    def _on_anal_error(self, msg: Message):
        self._progress.stop()
        self._lbl_badge.configure(text="  ✘ ERROR  ", bg=C["danger"], fg="#fff")
        self._write("❌  Error durante el análisis:\n", "peligro", clear=True)
        self._write(str(msg.payload) + "\n", "normal")
        self._status("Error en el análisis.")

    # ════════════════════════════════════════════════════════════════════════
    #  ACCIONES DE BOTONES
    # ════════════════════════════════════════════════════════════════════════

    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self._cam_thread = CameraThread(self._q, device=0)
        self._cam_thread.start()
        self._cam_running = True
        self._btn_cam.configure(text="⏹  Detener Cámara", bg=C["danger"])
        self._btn_cap.configure(state="normal")
        self._lbl_live.configure(text="● EN VIVO")
        self._status("Cámara activa  —  Presiona 📸 o [Q] para capturar y analizar.")

    def _stop_camera(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None

    def _capture_frame(self):
        """Botón 📸 y tecla [Q]: ordena captura al CameraThread."""
        if self._cam_running and self._cam_thread:
            # Cámara activa: señalizar captura al hilo
            self._cam_thread.capture()
            self._lbl_live.configure(text="📸 CAPTURANDO…")
            self._status("Capturando imagen…")
        elif self._frame_actual is not None:
            # Sin cámara pero hay imagen cargada: analizar directamente
            ts = datetime.now().strftime("%H%M%S")
            self._anal_worker.submit(self._frame_actual, f"archivo_{ts}.jpg")

    def _upload_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen de muestra de agua",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError("OpenCV no pudo leer la imagen.")
            self._frame_actual = frame

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self._show_pil(pil)
            self._lbl_img_info.configure(
                text=f"{os.path.basename(path)}  │  {pil.width}×{pil.height} px")
            self._status(
                f"Imagen cargada: {os.path.basename(path)}  —  [Q] o 📸 para analizar.")
            self._btn_cap.configure(state="normal")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{exc}")

    def _save_result(self):
        content = self._txt.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Vacío", "No hay resultados para guardar.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
            initialfile=f"ccn_agua_{ts}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._status(f"Guardado: {path}")
            messagebox.showinfo("Guardado", f"Resultado guardado:\n{path}")

    def _clear_all(self):
        if not messagebox.askyesno("Limpiar", "¿Limpiar imagen y resultados?"):
            return
        self._frame_actual = None
        self._placeholder()
        self._lbl_img_info.configure(text="Sin imagen cargada")
        self._lbl_live.configure(text="")
        self._write("", clear=True)
        self._lbl_badge.configure(text="  SIN ANALIZAR  ",
                                  bg=C["subtext"], fg=C["text"])
        self._btn_cap.configure(state="disabled")
        self._status("Pantalla limpiada.")

    def _show_about(self):
        w = tk.Toplevel(self)
        w.title("Acerca de")
        w.geometry("400x260")
        w.configure(bg=C["panel"])
        w.resizable(False, False)

        tk.Label(w, text="💧 CCN_AGUA",
                 font=("Segoe UI", 20, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(pady=(20, 4))
        tk.Label(w, text="Detección de Contaminación Hídrica",
                 font=("Segoe UI", 11), fg=C["text"],
                 bg=C["panel"]).pack()
        tk.Label(w, text="Motor: análisis local HSV  │  Sin API externa",
                 font=("Segoe UI", 9), fg=C["subtext"],
                 bg=C["panel"]).pack(pady=4)
        tk.Label(w,
                 text="Arch: CameraThread + ThreadPoolExecutor + Queue",
                 font=("Segoe UI", 9), fg=C["subtext"],
                 bg=C["panel"]).pack()

        for name, ok in [("Pillow", PIL_OK), ("opencv-python", CV2_OK)]:
            tk.Label(w, text=f"  {'✔' if ok else '✘'}  {name}",
                     font=("Consolas", 10),
                     fg=C["success"] if ok else C["danger"],
                     bg=C["panel"]).pack()

        tk.Button(w, text="Cerrar", command=w.destroy,
                  bg=C["accent"], fg="#fff", relief="flat",
                  padx=20, pady=6, cursor="hand2").pack(pady=16)

    # ════════════════════════════════════════════════════════════════════════
    #  HELPERS DE UI  (siempre hilo principal)
    # ════════════════════════════════════════════════════════════════════════

    def _placeholder(self):
        self._canvas.delete("all")
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()  or 460
        h = self._canvas.winfo_height() or 300
        self._canvas.create_text(
            w // 2, h // 2,
            text="Sin imagen\n\n📂 Subir Archivo  o  📷 Cámara\n[Q] = capturar",
            font=("Segoe UI", 13), fill=C["border"], justify="center"
        )

    def _show_pil(self, pil_img: "Image.Image"):
        self._canvas.update_idletasks()
        w = max(self._canvas.winfo_width(),  10)
        h = max(self._canvas.winfo_height(), 10)
        thumb = pil_img.copy()
        thumb.thumbnail((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(thumb)
        self._canvas.delete("all")
        self._canvas.create_image(w // 2, h // 2,
                                  anchor="center", image=self._tk_img)

    def _write(self, text: str, tag: str = "normal", clear: bool = False):
        self._txt.configure(state="normal")
        if clear:
            self._txt.delete("1.0", "end")
        if text:
            self._txt.insert("end", text, tag)
        self._txt.see("end")
        self._txt.configure(state="disabled")

    def _status(self, msg: str):
        self._lbl_status.configure(text=f"  {msg}")

    def _hover(self, btn: tk.Button, base: str):
        btn.bind("<Enter>", lambda _: btn.configure(bg=C["btn_hov"]))
        btn.bind("<Leave>", lambda _: btn.configure(bg=base))

    def _tick_clock(self):
        self._lbl_clock.configure(
            text=datetime.now().strftime("%H:%M:%S   %d/%m/%Y"))
        self.after(1000, self._tick_clock)

    def _on_hist_select(self, _event):
        sel = self._lst_hist.curselection()
        if not sel:
            return
        idx = len(self._history) - 1 - sel[0]
        if 0 <= idx < len(self._history):
            self._on_anal_result(
                Message(MsgType.ANAL_RESULT, self._history[idx]["payload"]))

    # ════════════════════════════════════════════════════════════════════════
    #  CIERRE LIMPIO
    # ════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        if self._cam_thread:
            self._cam_thread.stop()
        self._anal_worker.shutdown()
        self.destroy()


# ════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = CCNAguaApp()
    app.mainloop()