#Realizamos la importanción de las librerías necesarias para la creación de la interfaz gráfica
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
# Comprobamos que Pillow esté instalado (para manejar imágenes) - mediante try-catch para no romper
try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False
    missing.append("Pillow           →  pip install Pillow")
# Comprobamos que OpenCV esté instalado (para la cámara) - mediante try-catch
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    missing.append("opencv-python    →  pip install opencv-python")
# Si falta algo, detenemos el programa y avisamos
if missing:
    print("Faltan dependencias críticas:")
    for m in missing:
        print(f"   {m}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# RUTA O DIRECCIÓN PARA MODELO
def ruta_recurso(rel_path):
    # Función para encontrar la ruta correcta, ya sea en un .py normal o compilado en .exe
    try:
        base_path = sys._MEIPASS  # cuando es .exe
    except Exception:
        base_path = os.path.abspath(".")  # cuando es .py
    return os.path.join(base_path, rel_path)

# ════════════════════════════════════════════════════════════════════════════
#  MENSAJES ENTRE HILOS (Para que no choque la cámara con la interfaz)

#Usamos la POO - Creamos una clase de tipo Enum para definir los tipos de mensajes que 
# se enviarán entre los hilos de trabajo (cámara, análisis) y la interfaz gráfica.
class MsgType(Enum):
    # Tipos de mensajes que se mandarán entre los hilos de trabajo
    FRAME        = auto()
    CAM_ERROR    = auto()
    CAM_STOPPED  = auto()
    ANAL_START   = auto()
    ANAL_RESULT  = auto()
    ANAL_ERROR   = auto()
    STATUS       = auto()
@dataclass
class Message:
    # Estructura del mensaje (tipo y contenido)
    type:    MsgType
    payload: object = None

#----
#  HILO DE LA CÁMARA (Para que no se congele el programa)
#Función que permite seleccionar las cámaras disponibles en el sistema, probando índices del 0 al max_cams-1. Devuelve una lista de índices de cámaras disponibles.
def listar_camaras(max_cams=3):
    disponibles = []
    for i in range(max_cams):
        # Usamos CAP_DSHOW para que Windows las detecte más rápido
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # permite saber si la cámara está disponible
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles

#Clase de Hilo de camaras que captura frames de OpenCV y los publica en out_q. Tiene métodos para detener el hilo y para señalizar que el próximo frame es una captura.
class CameraThread(threading.Thread):
    #
    #Hilo que se encarga de leer el video de la cámara en segundo plano para que la ventana principal no se trabe.
    #stop()    → termina el ciclo y capture() → indixa que el próximo frame es una captura

    FRAME_INTERVAL = 0.033   # Más o menos 30 fotogramas por segundo
    def __init__(self, out_q: queue.Queue, device: int = 0):
        super().__init__(daemon=True, name="CameraThread")
        self._q        = out_q # Aquí mandamos los fotogramas
        self._device   = device # Qué cámara elegimos (0, 1, etc.)
        self._stop_evt = threading.Event()
        self._cap_evt  = threading.Event()
    def stop(self):
        #Este método o función permite detener la cámara/evento
        self._stop_evt.set()
    def capture(self):
        # Avisa que el usuario apretó el botón de capturar foto
        self._cap_evt.set()

    def run(self):
        # Aquí empieza a leer la cámara
        cap = cv2.VideoCapture(self._device, cv2.CAP_DSHOW) #Permite que OpenCV use DirectShow(mejorar la detección de camaraS)
        if not cap.isOpened():
            # Si no hay cámara, manda error a la interfaz
            self._q.put(Message(MsgType.CAM_ERROR, "No se pudo abrir la cámara."))
            return
        try:
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read() # Lee un fotograma
                if not ret:
                    time.sleep(0.05)
                    continue

                is_capture = self._cap_evt.is_set()
                if is_capture:
                    self._cap_evt.clear() #Limpiamos la cámara para tomar mejores 
                # Mandamos la imagen a la cola principal
                self._q.put(Message(
                    MsgType.FRAME,
                    {"frame": frame, "capture": is_capture}
                ))
                # Controlamos el tiempo para ir a 30 fps
                elapsed = time.monotonic() - t0
                sleep_t = max(0.0, self.FRAME_INTERVAL - elapsed)
                if sleep_t:
                    time.sleep(sleep_t)
        finally:
            # Controlamos el tiempo para ir a 30 fps
            cap.release()
            self._q.put(Message(MsgType.CAM_STOPPED))
#---
# Analizamos las imagenes mediante el CNN diseñado en .ipynb
class AnalizadorCNN:
    def __init__(self):
        #MEJORES RESULTADOS CON 600x600
        self.altura = 600 #ESTO ES LA MODIFICACIÓN - 20260418
        self.anchura = 600 #ESTO ES LA MODIFICACIÓN - 20260418

        #ESTA ES LA RUTA QUE VERIFICAREMOS SI SIRVE y se cargará el modelo y pesos UNA VEZ nomas
        self.modelo = load_model(ruta_recurso("CNN_Imagenes/Modelo/cnn.h5"))
        self.modelo.load_weights(ruta_recurso("CNN_Imagenes/Modelo/cnn_pesos.weights.h5"))

        # Orden exacto de las clases con las que entrenamos
        self.clases = [
            "AGUA_CON_PARTICULAS_ESTANCADA-Suspended_Stanqued_Matter",
            "AGUA_HIDROCARBURO-Hydrocarbons_Pollution",
            "AGUA_LIMPIA-Clean_Water",
            "AGUA_PLASTICO-Surface_Water_Pollution",
            "AGUA_SIN_OXIGENO-Oxygen_Depletion_Pollution",
            "AGUA_SUBMARINA_CONTAMINADA-UNDERWATER_PLASTIC_POLLUTION",
            "AGUA_SUCIA-GROUNDWATER_POLLUTION",
            "NO_RELACIONADO-Not_Related", 
        ]

    def analizar(self, frame_bgr):
        import cv2
        from datetime import datetime

        # Convertimos colores de OpenCV a los normales (RGB)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Cambiamos el tamaño de la imagen para que la acepte el modelo y  mejore analizando        
        #frame = cv2.resize(frame, (self.anchura, self.altura))
        frame = cv2.resize(frame, (self.anchura, self.altura), interpolation=cv2.INTER_AREA) #INTER_AREA mejora redimensionando más grande
        #Realizamos la normalización para que en la imágen lo detecte bien
        frame = frame / 255.0
        # Expandimos la dimensión porque Keras pide un lote de imágenes (aunque sea una sola)
        frame = np.expand_dims(frame, axis=0)
        #  Predicción
        pred = self.modelo.predict(frame)
        idx = np.argmax(pred[0]) # Vemos cuál clase ganó - Mayor probabilidad
        confianza = float(pred[0][idx]) # Porcentaje de probabilidad
        clase = self.clases[idx]
        # Colores personalizados para cada tipo de agua/clase
        colores = {
            "AGUA_LIMPIA-Clean_Water": "#52B788",
            "AGUA_HIDROCARBURO-Hydrocarbons_Pollution": "#E63946",
            "AGUA_PLASTICO-Surface_Water_Pollution": "#F4A261",
            "AGUA_SIN_OXIGENO-Oxygen_Depletion_Pollution": "#E63946",
            "AGUA_CON_PARTICULAS_ESTANCADA-Suspended_Stanqued_Matter": "#FFD166",
            "AGUA_SUBMARINA_CONTAMINADA-UNDERWATER_PLASTIC_POLLUTION": "#F77F00",
            "AGUA_SUCIA-GROUNDWATER_POLLUTION": "#9D4EDD",
            "NO_RELACIONADO-Not_Related": "#6C757D",
        }

        # Armamos el resultado final que se va a mostrar - lo que se visualizará
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
# ════════════════════════════════════════════════════════════════════════════
#  WORKER PARA ANALIZAR (Otro hilo para que no se trabe al pensar) - ThreadPoolExceutor

class AnalysisWorker:
    #Este hilo toma la foto y se la manda al AnalizadorCNN mientrás nuestra red neuronal se encarga de analizarla
    
    def __init__(self, out_q: queue.Queue):
        self._q     = out_q
        self._pool  = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="AnalWorker")
        self._motor = AnalizadorCNN() # Instanciamos nuestra red neuronal
        self._lock  = threading.Lock()

    def submit(self, frame_bgr, filename: str):
        #Manda la imagen al hilo de análisis
        with self._lock:
            self._q.put(Message(MsgType.ANAL_START))
            self._pool.submit(self._run, frame_bgr.copy(), filename)

    def _run(self, frame_bgr, filename: str):
        #Ejecuta el modelo y devuelve los resultados a la interfaz
        try:
            resultado = self._motor.analizar(frame_bgr)
            resultado["filename"] = filename
            self._q.put(Message(MsgType.ANAL_RESULT, resultado))
        except Exception as exc:
            self._q.put(Message(MsgType.ANAL_ERROR, str(exc)))

    def shutdown(self):
        # Apaga el hilo de análisis
        self._pool.shutdown(wait=False)
#  PALETA de colores (diseño oscuro original)
C = {
    "bg":       "#0D1B2A", # Fondo principal
    "panel":    "#1B2A3B", #Laterales
    "btn":      "#1E3A4F", #Botones
    "btn_hov":  "#2A5270", #Hover del bóton
    "accent":   "#00B4D8", #Ciertos detallitos mas bónitos
    "accent2":  "#48CAE4",
    "success":  "#52B788", #Verde  -Agua Limpio
    "warning":  "#F4A261", #Naranja
    "danger":   "#E63946", #Tojo (muy contmainada)
    "text":     "#E0F4FF",
    "subtext":  "#90C4D8",
    "border":   "#2D4A5E",
    "cam_live": "#1A6B3C",
    "cam_cap":  "#7B2D8B",
}
#  VENTANA PRINCIPAL (La interfaz tal cual)
class CCNAguaApp(tk.Tk):
    POLL_MS = 30 # Cada cuántos milisegundos revisa mensajes nuevos - cola
    # Configuración básica de la ventana
    def __init__(self):
        super().__init__()
        self.title("CCN_AGUA  │  Detección de Contaminación Hídrica")
        self.geometry("1150x740")
        self.minsize(960, 620)
        self.configure(bg=C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close) #Cierra todo

        #Comunicación entre hilos y la UI
        self._q: queue.Queue = queue.Queue()

        #Variables para controlar los hilos
        self._cam_thread: Optional[CameraThread] = None
        self._anal_worker = AnalysisWorker(self._q)

        #Variables de estado de la aplicación
        self._frame_actual  = None
        self._cam_running   = False
        self._history: list = []
        self._tk_img        = None

        #Construimos la interfaz visual
        self._setup_styles() # Estilos de combobox y botones
        self._build_header() # Arriba (Logos y títulos)
        self._build_main() # Medio (Botones y cámara)
        self._build_statusbar() #Barra de estatus abajo

        # Si el usuario presiona la tecla 'Q,q', toma una foto
        self.bind("<q>", lambda _e: self._capture_frame())
        self.bind("<Q>", lambda _e: self._capture_frame())

        # Arranca el bucle para escuchar mensajes y el reloj
        self._poll_queue()
        self._tick_clock()
        self._status("Listo. Cargue una imagen o active la cámara.  [Q] = capturar")

    #  CONSTRUYENDO LOS BLOQUES DE LA VENTANA

    def _build_header(self):
        #Barra superior con el logo y el nombre de la app
        h = tk.Frame(self, bg=C["panel"], pady=10)
        h.pack(fill="x")
        # Intentamos cargar el logo (logo_Hydro_PNG.png)
        if PIL_OK:
            try:
                # Busca logo.png en el directorio actual - RUTA ABSOLUTA
                ruta_logo = r"C:\Users\alexi\OneDrive\Desktop\SII_CNN\CNN_Agua\logo_Hydro_PNG.png"
                # Le cambiamos el tamaño a 45x45 píxeles
                img = Image.open(ruta_logo)
                img = img.resize((45, 45), Image.Resampling.LANCZOS)
                # Crea el objeto compatible con Tkinter
                self._logo_img = ImageTk.PhotoImage(img)
                
                # Coloca el logo en un Label y lo agrega a la barra superior
                lbl_logo = tk.Label(h, image=self._logo_img, bg=C["panel"])
                lbl_logo.pack(side="left", padx=(20, 10))
                
            except Exception as e:
                # Si la imagen no existe, muestra un bloque de error en su lugar
                print(f"No se pudo cargar {ruta_logo}: {e}")
                lbl_logo = tk.Label(h, text="[LOGO]", font=("Segoe UI", 12, "bold"), fg=C["danger"], bg=C["panel"])
                lbl_logo.pack(side="left", padx=(20, 10))
        
        #Textos de título
        tk.Label(h, text="HydroScan",
                font=("Segoe UI", 22, "bold"),
                fg=C["accent"], bg=C["panel"]).pack(side="left", padx=20)
        tk.Label(h, text="Detección de Contaminación Hídrica",
                font=("Segoe UI", 11),
                fg=C["subtext"], bg=C["panel"]).pack(side="left")
        # Etiqueta vacía para el reloj (se actualiza sola después)
        self._lbl_clock = tk.Label(h, text="",
                                font=("Segoe UI", 10),
                                fg=C["subtext"], bg=C["panel"])
        self._lbl_clock.pack(side="right", padx=20)

    def _build_main(self):
        #Se creará el cuerpo principal dividido en panel izquierdo (cámara) y derecho (resultados)
        pane = tk.PanedWindow(self, orient="horizontal", bg=C["bg"],
                            sashwidth=6, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=10, pady=8)

        # Lado izquierdo
        left = tk.Frame(pane, bg=C["bg"])
        pane.add(left, minsize=440, width=500)
        self._build_buttons(left) #Botón de acción
        self._build_preview(left) #Vista previa de la cámara o imagen cargada

        # Lado derecho
        right = tk.Frame(pane, bg=C["bg"])
        pane.add(right, minsize=360)
        self._build_results(right) #Análisis y resultados

    def _build_buttons(self, parent):
        #Contenedor para todos los botones y selectores
        bar = tk.Frame(parent, bg=C["panel"], pady=10, padx=10)
        bar.pack(fill="x", pady=(0, 8))

        # FILA 0: TÍTULO y al lado derecho el selector de cámara
        tk.Label(bar, text="ACCIONES",
                font=("Segoe UI", 9, "bold"),
                fg=C["subtext"], bg=C["panel"]).grid(
                    row=0, column=0, columnspan=6,
                    sticky="w", pady=(0, 8))
        
        #FILA 1: BOTONES PRINCIPALES 
        defs = [
            ("📷  Cámara",        self._toggle_camera,  C["accent"],   "_btn_cam"),
            ("📸  Capturar [Q]",  self._capture_frame,  C["cam_cap"],  "_btn_cap"),
            ("📂  Subir Archivo", self._upload_file,     C["btn"],      None),
            ("💾  Guardar",       self._save_result,     C["btn"],      None),
            ("🗑  Limpiar",       self._clear_all,       C["btn"],      None),
            ("ℹ️  Info",          self._show_about,      C["btn"],      None),
        ]
        #Creamos los botones con un ciclo para no repetir código
        for col, (label, cmd, color, attr) in enumerate(defs):
            btn = tk.Button(bar, text=label, command=cmd,
                            font=("Segoe UI", 10),
                            bg=color, fg=C["text"],
                            activebackground=C["btn_hov"],
                            activeforeground=C["text"],
                            relief="flat", bd=0,
                            padx=10, pady=7, cursor="hand2")
            btn.grid(row=1, column=col, padx=3)
            self._hover(btn, color) #Efecto de cierre al pasar el mouse
            if attr:
                setattr(self, attr, btn) #Guardamos referencia para usarlos luego
        #Desactivamos el botón capturar al inicio
        self._btn_cap.configure(state="disabled")

        #Seleccionador de cámara (Fila 2)
        frame_cam = tk.Frame(bar, bg=C["panel"])
        frame_cam.grid(row=2, column=0, columnspan=6, sticky="w", pady=(12, 0), padx=3)

        tk.Label(frame_cam,
                text="⚙️ Seleccionar Cámara: ",
                font=("Segoe UI", 9, "bold"),
                fg=C["accent"],
                bg=C["panel"]).pack(side="left", padx=(0,10))

        #Creamos una lista dinamica para que la app no se congele buscando cámaras
        camaras_disponibles = listar_camaras()
        self.combo_cam = ttk.Combobox(
            frame_cam,
            values=camaras_disponibles,
            width=5,
            state="readonly"
        )
        self.combo_cam.pack(side="left")

        #Solo selecciona el índice 0 si encontró al menos una cámara
        if camaras_disponibles:
            self.combo_cam.current(0)
        else:
            self.combo_cam.set("") # Se queda vacío si no hay cámaras

    def _build_preview(self, parent):
        #Pantalla prebiew para visualizar la cámara o la imagen cargada
        frm = tk.LabelFrame(parent, text=" Vista Previa ",
                            font=("Segoe UI", 10, "bold"),
                            fg=C["accent"], bg=C["panel"],
                            bd=1, relief="flat", labelanchor="nw")
        frm.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(frm, bg="#0A1520",
                                highlightthickness=1,
                                highlightbackground=C["border"])
        self._canvas.pack(fill="both", expand=True, padx=6, pady=6)
        #Barrita inferior con información
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

        self._placeholder() #Pone el texto "Sin imagen" al inicio

    def _build_results(self, parent):
        # El panel derecho con el diagnóstico - Resultados de CCN
        top = tk.Frame(parent, bg=C["bg"])
        top.pack(fill="x", pady=(0, 6))
        tk.Label(top, text="RESULTADO DEL ANÁLISIS",
                font=("Segoe UI", 11, "bold"),
                fg=C["accent"], bg=C["bg"]).pack(side="left")
        #La etiqueta que cambia de color según la gravedad
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
        tk.Label(hdr, text="LA IMAGEN MUESTRA:",
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
        #Configuramos los colores de texto (tags)
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

        # # Cuadro de abajo para el historial de análisis - 
        hf = tk.LabelFrame(parent, text=" Historial ",
                        font=("Segoe UI", 9, "bold"),
                        fg=C["subtext"], bg=C["bg"],
                        bd=1, relief="flat", labelanchor="nw")
        hf.pack(fill="x", pady=(8, 0))
        self._lst_hist = tk.Listbox(hf, height=20,
                                    bg=C["panel"], fg=C["subtext"],
                                    selectbackground=C["accent"],
                                    font=("Consolas", 9),
                                    relief="flat", bd=0, activestyle="none")
        self._lst_hist.pack(fill="x", padx=4, pady=4)
        self._lst_hist.bind("<<ListboxSelect>>", self._on_hist_select) # Cargar fotos viejas al dar clic

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["panel"], pady=4)
        bar.pack(fill="x", side="bottom")
        self._lbl_status = tk.Label(bar, text="",
                                    font=("Segoe UI", 9),
                                    fg=C["subtext"], bg=C["panel"], anchor="w")
        self._lbl_status.pack(side="left", padx=12, fill="x", expand=True)
        #Barrita de progreso infinita para cuando está analizando
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self._progress.pack(side="right", padx=12)

    #FUNCIÓN PARA ESTILIZAR LOS BOTONES CON EFECTO HOVER y CURSOR
    def _setup_styles(self):
        #Configura los estilos avanzados de TTK para adaptarlos al tema oscuro.
        style = ttk.Style(self)
        style.theme_use('clam') 
        style.configure("TCombobox",
            fieldbackground=C["bg"],       # Fondo del texto
            background=C["btn"],           # Fondo del botón de la flechita
            foreground=C["text"],          # Color del texto
            bordercolor=C["border"],       # Color del borde
            arrowcolor=C["accent"],        # Color de la flecha
            lightcolor=C["border"],
            darkcolor=C["border"],
            padding=4
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", C["bg"])],
            selectbackground=[("readonly", C["accent"])], # Fondo al clickear
            selectforeground=[("readonly", C["bg"])],     # Texto al clickear
            background=[("active", C["btn_hov"])],        # Ilumina el botón al pasar el mouse
            bordercolor=[("focus", C["accent2"])]
        )
        self.option_add('*TCombobox*Listbox.background', C["panel"])
        self.option_add('*TCombobox*Listbox.foreground', C["text"])
        self.option_add('*TCombobox*Listbox.selectBackground', C["accent"])
        self.option_add('*TCombobox*Listbox.selectForeground', C["bg"])
        self.option_add('*TCombobox*Listbox.font', ("Segoe UI", 10))

    #Análisis de mensajes
    def _poll_queue(self):
        #Cada 30ms revisa si algún hilo mandó una foto o un análisis
        try:
            while True:
                msg: Message = self._q.get_nowait()
                self._dispatch(msg)
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._poll_queue)

    def _dispatch(self, msg: Message):
        #Según el mensaje que llegó, decide qué función ejecutar
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

    #Respuestas a mensajes
    def _on_frame(self, msg: Message):
        # Qué hacer cuando llega un fotograma de la cámara
        data       = msg.payload
        frame      = data["frame"]
        is_capture = data["capture"] #Si analizo/capturo la imágen

        self._frame_actual = frame
        #Convertimos para que Pillow pueda dibujarlo en Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if is_capture:
            # Dibujamos un marco azul sobre la foto
            draw = ImageDraw.Draw(pil)
            for t in range(6):
                draw.rectangle(
                    [t, t, pil.width - t - 1, pil.height - t - 1],
                    outline=(0, 180, 216))
            draw.text((14, 14), "CAPTURADO", fill=(0, 180, 216))

        self._show_pil(pil)

        if is_capture:
            #Si tomaron foto, mandarla al análisis y mostrar info
            ts = datetime.now().strftime("%H%M%S")
            fname = f"frame_{ts}.jpg"
            self._lbl_img_info.configure(
                text=f"📸 {fname}  │  {pil.width}×{pil.height} px")
            self._lbl_live.configure(text="📸 CAPTURADO")
            self._status("Imagen capturada. Analizando…")
            self._anal_worker.submit(frame, fname)

    def _on_cam_error(self, msg: Message):
        #Si la cámara falló
        self._cam_running = False
        self._btn_cam.configure(text="📷  Cámara", bg=C["accent"])
        self._btn_cap.configure(state="disabled")
        self._lbl_live.configure(text="")
        messagebox.showerror("Error de cámara", msg.payload)

    def _on_cam_stopped(self, _msg: Message):
        #Si el usuario detuvo la cámara
        self._cam_running = False
        self._btn_cam.configure(text="📷  Cámara", bg=C["accent"])
        self._btn_cap.configure(state="disabled")
        self._lbl_live.configure(text="")
        self._status("Cámara detenida.")

    def _on_anal_start(self, _msg: Message):
        #Cuando la red neuronal / análisis empieza a generarse
        self._progress.start(10)
        self._lbl_badge.configure(text="  ⏳ ANALIZANDO…  ",
                                bg=C["warning"], fg="#000")
        self._write("", clear=True)
        self._write("⏳  Analizando muestra, por favor espere…\n\n", "spin")
        self._status("Procesando imagen con CNN")

    def _on_anal_result(self, msg: Message):
        #PARA PRESENTAR LAS CLASES QIE SON USADAS - Los resultados
        # Diccionario para que los nombres de las carpetas se lean bonitos en la app
        nombres_ui = {
            "AGUA_LIMPIA-Clean_Water": "Agua Limpia",
            "AGUA_HIDROCARBURO-Hydrocarbons_Pollution": "Agua con Hidrocarburos",
            "AGUA_PLASTICO-Surface_Water_Pollution": "Agua con Plásticos",
            "AGUA_SIN_OXIGENO-Oxygen_Depletion_Pollution": "Agua sin Oxígeno",
            "AGUA_CON_PARTICULAS_ESTANCADA-Suspended_Stanqued_Matter": "Agua con Partículas, Sedimentos o Estancada",
            "AGUA_SUBMARINA_CONTAMINADA-UNDERWATER_PLASTIC_POLLUTION": "Agua Submarina Contaminada",
            "AGUA_SUCIA-GROUNDWATER_POLLUTION": "Agua Sucia",
            "NO_RELACIONADO-Not_Related": "No Reconocido", #CLASE NUEVA PARA IMÁGENES NO RELACIONADAS CON CONTAMINACIÓN HÍDRICA
        }

        self._progress.stop() #Detenemos la barra de carga
        res: dict  = msg.payload
        nivel = res["nivel"]
        nivel_mostrar = nombres_ui.get(nivel, nivel)
        color_ui   = res["color_ui"]
        hallazgos  = res["hallazgos"]
        metricas   = res["metricas"]
        ts         = res["ts"]
        filename   = res.get("filename", "imagen")

        #Actualizamos la etiqueta de arriba
        badge_txt = f"  {nivel.replace('_',' ')}  "
        badge_bg = color_ui
        self._lbl_badge.configure(text=badge_txt, bg=badge_bg, fg="#fff")
        
        #Imprimimos el análisis
        self._write("", clear=True)
        self._write(f"{ts}\n", "fecha")
        self._write("━" * 54 + "\n", "seccion")
        self._write("LA IMAGEN MUESTRA:\n\n", "titulo")
        #Escogemos un color dependiendo de la clase para el texto
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
        #Guardamos en el historial del lado derecho
        entry = {"ts": ts, "file": filename,
                "nivel": nivel, "payload": res}
        self._history.append(entry)
        self._lst_hist.insert(
            0, f"[{ts}]  {filename}  →  {badge_txt.strip()}")
        self._status(f"Análisis completado  ·  {ts}")

    def _on_anal_error(self, msg: Message):
        #Si el análisis falló por alguna razón (modelo, formato de imagen, etc)
        self._progress.stop()
        self._lbl_badge.configure(text="  ERROR  ", bg=C["danger"], fg="#fff")
        self._write("Error durante el análisis:\n", "peligro", clear=True)
        self._write(str(msg.payload) + "\n", "normal")
        self._status("Error en el análisis.")

    # ACCIONES DE BOTONES
    def _toggle_camera(self):
        #Apaga o enciende la cámara
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        #Intentamos leer qué número seleccionó el usuario en el Combobox
        try:
            texto_combo = self.combo_cam.get()
            device = int(texto_combo) if texto_combo != "" else 0
        except ValueError:
            device = 0 # Si escribieron letras o está vacío, por seguridad usamos la cámara 0
            
        #Arrancamos el hilo secundario
        self._cam_thread = CameraThread(self._q, device=device)
        self._cam_thread.start()
        self._cam_running = True
        
        #Cambiamos botones
        self._btn_cam.configure(text="⏹  Detener Cámara", bg=C["danger"])
        self._btn_cap.configure(state="normal")
        self._lbl_live.configure(text="● EN VIVO")
        self._status("Cámara activa  —  Presiona 📸 o [Q] para capturar y analizar.")

    def _stop_camera(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None

    def _capture_frame(self):
        #Botón de tomar foto
        if self._cam_running and self._cam_thread:
            #Si hay cámara prendida, le avisa que tome la foto
            self._cam_thread.capture()
            self._lbl_live.configure(text="📸 CAPTURANDO…")
            self._status("Capturando imagen…")
        elif self._frame_actual is not None:
            #Si alguien subió un archivo y quiere volver a analizarlo
            ts = datetime.now().strftime("%H%M%S")
            self._anal_worker.submit(self._frame_actual, f"archivo_{ts}.jpg")

    def _upload_file(self):
        #Abre el buscador de archivos de Windows
        path = filedialog.askopenfilename(
            title="Seleccionar imagen de muestra de agua",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                        ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            #Leemos la foto con OpenCV
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError("OpenCV no pudo leer la imagen.")
            self._frame_actual = frame
            #La ponemos en la pantalla
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
        #Guarda el reporte en un archivo .txt
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
        #Botón para limpiar la pantalla y reiniciar el estado
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
        #Pantallita extra de créditos
        w = tk.Toplevel(self)
        w.title("Acerca de")
        w.geometry("400x260")
        w.configure(bg=C["panel"])
        w.resizable(False, False)

        tk.Label(w, text="CCN_AGUA",
                font=("Segoe UI", 20, "bold"),
                fg=C["accent"], bg=C["panel"]).pack(pady=(20, 4))
        tk.Label(w, text="Detección de contaminación hídrica por imágenes",
                font=("Segoe UI", 11), fg=C["text"],
                bg=C["panel"]).pack()
        tk.Label(w, text="Análisis local con TensorFlow/Keras + CNN personalizada",
                font=("Segoe UI", 9), fg=C["subtext"],
                bg=C["panel"]).pack(pady=4)
        tk.Label(w,
                text="Arch: CameraThread + ThreadPoolExecutor + Queue",
                font=("Segoe UI", 9), fg=C["subtext"],
                bg=C["panel"]).pack()
        #Muestra si Pillow y OpenCV cargaron bien
        for name, ok in [("Pillow", PIL_OK), ("opencv-python", CV2_OK)]:
            tk.Label(w, text=f"  {'✔' if ok else '✘'}  {name}",
                    font=("Consolas", 10),
                    fg=C["success"] if ok else C["danger"],
                    bg=C["panel"]).pack()

        tk.Button(w, text="Cerrar", command=w.destroy,
                bg=C["accent"], fg="#fff", relief="flat",
                padx=20, pady=6, cursor="hand2").pack(pady=16)

    #-----
    #FUNCIONES DE AYUDA (Para no repetir código)

    def _placeholder(self):
        #Dibuja el recuadro que dice "Sin imagen" al iniciar
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
        #Actualiza la hora cada segundo
        self._lbl_clock.configure(
            text=datetime.now().strftime("%H:%M:%S   %d/%m/%Y"))
        self.after(1000, self._tick_clock)

    def _on_hist_select(self, _event):
        #Cuando le das clic a un historial, vuelve a cargar los resultados de esa foto
        sel = self._lst_hist.curselection()
        if not sel:
            return
        idx = len(self._history) - 1 - sel[0]
        if 0 <= idx < len(self._history):
            self._on_anal_result(
                Message(MsgType.ANAL_RESULT, self._history[idx]["payload"]))

    #---
    #Al finalizar el progrmama, aseguramos cerrar los hilos correctamente para no dejar procesos colgados
    def _on_close(self):
        #Aseguramos apagar la cámara y los hilos antes de cerrar - 
        if self._cam_thread:
            self._cam_thread.stop()
        self._anal_worker.shutdown()
        self.destroy()
# AQuí es donde se ejecuta el programa
if __name__ == "__main__":
    app = CCNAguaApp()
    app.mainloop()