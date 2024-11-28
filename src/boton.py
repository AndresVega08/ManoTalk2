import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import Label, Button, PhotoImage, messagebox
from PIL import Image, ImageTk
from utils.keypoints import extract_keypoints
import time

# Cargar el modelo entrenado
model = tf.keras.models.load_model('models/gesture_model.h5')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar los nombres de los gestos
DATA_PATH = 'data'
gestures = os.listdir(DATA_PATH)

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ManoTalk")
        self.root.geometry("1200x700")

        # Variables iniciales
        self.sequence = []
        self.sequence_length = 60
        self.hands_in_frame = False
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5

        # Panel izquierdo
        self.sidebar = tk.Frame(root, width=200, bg="#0C9F0F")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Título y logotipo
        title_frame = tk.Frame(self.sidebar, bg="#0C9F0F")
        title_frame.pack(pady=(20, 20))

        title_font = ("Helvetica", 16, "bold")
        title_label = tk.Label(title_frame, text="ManoTalk", font=title_font, fg="black", bg="#0C9F0F")
        title_label.pack(side="left", padx=(10, 10))

        # Cargar icono (opcional)
        try:
            icon_image = Image.open("src/img/MT2.0.png")  # Reemplaza con la ruta a tu imagen
            icon_image = icon_image.resize((30, 30), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(title_frame, image=icon_photo, bg="#0C9F0F")
            icon_label.image = icon_photo
            icon_label.pack(side="left")

            # Cargar el ícono de apagado
            power_icon = Image.open("src/img/Botones/power.png")  # Reemplaza con la ruta a tu ícono
            power_icon = power_icon.resize((24, 24), Image.Resampling.LANCZOS)
            power_icon_photo = ImageTk.PhotoImage(power_icon)
            

        except Exception as e:
            print(f"Error al cargar la imagen del icono: {e}")

        # Crear la línea de separación
        separator = tk.Frame(self.sidebar, height=2, bd=1, relief="sunken", bg="#0C9F0F")
        separator.pack(fill="x", pady=(0, 10))

        # Botones de la barra lateral
        button_style = {
            "bg": "#0C9F0F", "fg": "white", "font": ("Helvetica", 12, "bold"),
            "relief": "flat", "bd": 0, "highlightthickness": 0, "anchor": "w", "width": 12
        }

        self.round_image_inicio = PhotoImage(file="src/img/Botones/InicioBtn.png")
        self.btn_inicio = Button(self.sidebar, image=self.round_image_inicio, command=self.show_home, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_gestos = PhotoImage(file="src/img/Botones/GestosBtn.png")
        self.btn_gestos = Button(self.sidebar, image=self.round_image_gestos, command=self.mostrar_carpetas, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_guias = PhotoImage(file="src/img/Botones/GuiasBtn.png")
        self.btn_guias = Button(self.sidebar, image=self.round_image_guias, command=self.show_guides, borderwidth=0, highlightthickness=0, bg="#0C9F0F")
        
        self.round_image_info = PhotoImage(file="src/img/Botones/infoBtn.png")
        self.btn_info = Button(self.sidebar, image=self.round_image_info, command=self.show_info, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.btn_salir = Button(self.sidebar, text=" Salir", image=power_icon_photo, command=root.quit,bg="#0C9F0F", fg="#ffffff", font=("Helvetica", 12, "bold"),borderwidth=0, compound="right",padx=10, width=150)
        self.btn_salir.image = power_icon_photo  # Guardar una referencia para evitar que se elimine
        

        self.btn_inicio.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_gestos.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_guias.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_info.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_salir.pack(pady=(10, 20), padx=20, ipadx=10, ipady=10, side="bottom")

        # Panel derecho
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Añadir texto principal
        self.text_label = Label(self.main_frame, text="Reconocimiento de Gestos", font=("Arial", 24), bg="#0C9F0F", fg="#ffffff")
        self.text_label.pack(fill=tk.X)  # Ocupa todo el ancho en la parte superior

        # Frame para video y gestos
        self.video_frame = tk.Frame(self.main_frame, bg="white")
        self.video_frame.pack()

        # Label para el video
        self.video_label = Label(self.video_frame)
        self.video_label.grid(row=0, column=0)

        # Label para el texto de los gestos, ajustado para anclarse en la parte superior
        self.gesture_label = Label(self.video_frame, text="", font=("Arial", 18), bg="white", anchor="n", justify="left")
        self.gesture_label.grid(row=0, column=1, padx=10, sticky="n")

        # Inicializar historial de gestos
        self.gesture_history = []

        # Botón para iniciar captura de video
        self.btn_start = Button(self.main_frame, text="Iniciar Captura", command=self.start_video)
        self.btn_start.pack(pady=10)

        # Inicializar la captura de video
        self.cap = None

    def show_home(self):
        messagebox.showinfo("Inicio", "Bienvenido a ManoTalk")

    def show_gestures(self):
        messagebox.showinfo("Gestos", "Gestos disponibles: " + ", ".join(gestures))

    def show_guides(self):
        messagebox.showinfo("Guías", "Guías disponibles para el uso de la aplicación.")

    def show_info(self):
        messagebox.showinfo("Más información", "Información adicional sobre ManoTalk.")
    def start_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():  # Verificar si la cámara se abrió correctamente
                messagebox.showerror("Error", "No se pudo acceder a la cámara.")
                self.cap = None  # Asegurarse de que self.cap sea None si no se puede abrir la cámara
                return
            self.update_video()  # Iniciar la actualización de video si la cámara se abrió
            self.btn_start.pack_forget()
            self.btn_stop.pack()
        else:
            messagebox.showwarning("Advertencia", "La captura de video ya está en marcha.")

    def update_video(self):
        if self.cap is None:  # Verificar que la captura no es None
            return

        ret, frame = self.cap.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Dibujar las manos en el frame
            if results.multi_hand_landmarks:
                self.hands_in_frame = True
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                keypoints = extract_keypoints(results)
                if keypoints.shape[0] == 63:
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-self.sequence_length:]
            else:
                if self.hands_in_frame and len(self.sequence) >= 60:
                    current_time = time.time()
                    if current_time - self.last_prediction_time >= self.prediction_interval:
                        sequence_np = np.array(self.sequence, dtype=np.float32)
                        if len(sequence_np) < 88:
                            padding = np.zeros((88 - len(sequence_np), 63))
                            sequence_np = np.vstack([sequence_np, padding])
                        res = model.predict(np.expand_dims(sequence_np, axis=0))[0]
                        
                        if len(res) == len(gestures):
                            gesture = gestures[np.argmax(res)]
                            print(f"Gesto reconocido: {gesture}")
                            cv2.putText(frame, f"Gesto: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            self.gesture_history.append(gesture)
                            self.gesture_label.config(text="Historial de gestos:\n" + "\n".join(self.gesture_history))

                self.hands_in_frame = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img
        self.video_label.config(image=img)

        if self.cap and self.cap.isOpened():  # Solo continuar si la captura está abierta
            self.video_label.after(10, self.update_video)

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.btn_stop.pack_forget()
            self.btn_start.pack()


def start_cam():
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_video)
    root.mainloop()

if __name__ == "__main__":
    start_cam()
