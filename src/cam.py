import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import Label, Button, messagebox
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
        self.root.geometry("800x600")
        
        # Variables iniciales
        self.sequence = []
        self.sequence_length = 60
        self.hands_in_frame = False
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5
        
        # Panel izquierdo
        self.sidebar = tk.Frame(root, width=200, bg="#E0E0E0")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Agregar el texto "ManoTalk" y la imagen
        title_frame = tk.Frame(self.sidebar, bg="#E0E0E0")
        title_frame.pack(pady=(20, 20))

        title_font = ("Helvetica", 16, "bold")
        title_label = tk.Label(title_frame, text="ManoTalk", font=title_font, fg="black", bg="#E0E0E0")
        title_label.pack(side="left", padx=(10, 10))

        try:
            icon_image = Image.open("src\\img\\MT.png")  # Reemplaza con la ruta a tu imagen de icono
            icon_image = icon_image.resize((30, 30), Image.Resampling.LANCZOS)  # Ajustar el tamaño de la imagen
            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(title_frame, image=icon_photo, bg="#E0E0E0")
            icon_label.image = icon_photo  # Guardar una referencia de la imagen
            icon_label.pack(side="left")
        except Exception as e:
            print(f"Error al cargar la imagen del icono: {e}")
        
       # Crear la línea de separación
        separator = tk.Frame(self.sidebar, height=2, bd=1, relief="sunken", bg="#E0E0E0")

        

        # Botones de la barra lateral
        button_style = {
            "bg": "#7584F2", "fg": "white", "font": ("Helvetica", 12, "bold"),
            "relief": "flat", "bd": 0, "highlightthickness": 0, "anchor": "w", "width": 12
        }

        self.btn_inicio = Button(self.sidebar, text="Inicio", command=self.show_home, **button_style)
        self.btn_gestos = Button(self.sidebar, text="Gestos", command=self.mostrar_carpetas, **button_style)
        self.btn_guias = Button(self.sidebar, text="Guías", command=self.show_guides, **button_style)
        self.btn_info = Button(self.sidebar, text="Más información", command=self.show_info, **button_style)
        self.btn_salir = Button(self.sidebar, text="Salir", command=root.quit, **button_style)

        self.btn_inicio.pack(pady=10, padx=20, ipadx=10, ipady=10)
        self.btn_gestos.pack(pady=10, padx=20, ipadx=10, ipady=10)
        self.btn_guias.pack(pady=10, padx=20, ipadx=10, ipady=10)
        self.btn_info.pack(pady=10, padx=20, ipadx=10, ipady=10)

        # Agregar el botón de salir al fondo primero
        self.btn_salir.pack(pady=(10, 20), padx=20, ipadx=10, ipady=10, side="bottom")

        # Agregar la línea de separación justo encima del botón de salir
        separator = tk.Frame(self.sidebar, height=2, bd=1, relief="sunken", bg="#E0E0E0")
        separator.pack(fill="x", pady=(0, 10), side="bottom")


        
        # Panel derecho
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas para mostrar video
        self.video_label = Label(self.main_frame)
        self.video_label.pack()
        
        # Añadir texto
        self.text_label = Label(self.main_frame, text="Reconocimiento de Gestos", font=("Arial", 24), bg="white")
        self.text_label.pack(pady=20)
        
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
            self.update_video()
            self.btn_start.pack_forget()  # Eliminar el botón después de iniciada la captura
        else:
            messagebox.showwarning("Advertencia", "La captura de video ya está en marcha.")
            
    def mostrar_carpetas(self):
        ruta_data = 'Data'
        if os.path.exists(ruta_data) and os.path.isdir(ruta_data):
            carpetas = [nombre for nombre in os.listdir(ruta_data) if os.path.isdir(os.path.join(ruta_data, nombre))]
            messagebox.showinfo("Carpetas en Data", "\n".join(carpetas))
        else:
            messagebox.showerror("Error", "La carpeta 'Data' no existe o no es un directorio.")        

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Dibujar las manos en el frame
            if results.multi_hand_landmarks:
                self.hands_in_frame = True
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extraer keypoints
                keypoints = extract_keypoints(results)
                
                # Asegurarse de que los keypoints tengan la longitud correcta antes de añadirlos
                if keypoints.shape[0] == 63:  # Verificar que hay 63 elementos
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-self.sequence_length:]  # Mantener solo los últimos 'sequence_length' elementos
            else:
                if self.hands_in_frame and len(self.sequence) >= 60:
                    current_time = time.time()
                    if current_time - self.last_prediction_time >= self.prediction_interval:  # Verificar el intervalo
                        # Rellenar con ceros hasta la longitud esperada (88)
                        sequence_np = np.array(self.sequence, dtype=np.float32)
                        if len(sequence_np) < 88:
                            padding = np.zeros((88 - len(sequence_np), 63))  # Rellenar con ceros
                            sequence_np = np.vstack([sequence_np, padding])  # Apilar la secuencia con el padding
                        res = model.predict(np.expand_dims(sequence_np, axis=0))[0]
                        
                        # Imprimir los resultados de la predicción
                        if len(res) == len(gestures):
                            gesture = gestures[np.argmax(res)]  # Mostrar el gesto reconocido
                            print(f"Gesto reconocido: {gesture}")
                            cv2.putText(frame, f"Gesto: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.last_prediction_time = current_time  # Actualizar el tiempo de la última predicción
                self.hands_in_frame = False  # No hay manos en cuadro
            
            # Mostrar la imagen en el label
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = img
            self.video_label.config(image=img)
        
        self.video_label.after(10, self.update_video)  # Llamar a la función de actualización cada 10 ms

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def start_cam():
    # Crear la ventana de la aplicación
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_video)  # Asegurarse de liberar la cámara al cerrar
    root.mainloop()


# Ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_video)  # Asegurarse de liberar la cámara al cerrar
    root.mainloop()