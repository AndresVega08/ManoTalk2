import os
import time
import tkinter as tk
import webbrowser
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tkinter import Button, Label, PhotoImage, messagebox, Scrollbar, Canvas, Frame
from PIL import Image, ImageTk
import sys


if getattr(sys, 'frozen', False):
    # Si estamos corriendo desde un ejecutable
    base_path = sys._MEIPASS  # Esta es la ruta temporal donde PyInstaller extrae los archivos
else:
    # Si estamos corriendo desde el script
    base_path = os.path.abspath(".")

# Cargar los nombres de los gestos
DATA_PATH = os.path.join(base_path, 'resources/data')
gestures = os.listdir(DATA_PATH)

def mostrar_cam():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        root.destroy()
        import cam
        cam.start_cam()

def mostrar_acb():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        root.destroy()
        import Acb
        Acb.start_acb()

def mostrar_guias():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        root.destroy()
        import guias
        guias.start_guias()

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ManoTalk")
        self.root.geometry("1200x700")
        # self.root.iconbitmap("resources\icon\MT2.0.ico")

        self.root.iconbitmap(os.path.join(base_path, "resources","icon","MT2.0.ico"))

        # Variables iniciales
        self.cap = None  # Variable para la cámara
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier(os.path.join(base_path, "resources", "models", "ABC", "keras_model.h5"), os.path.join(base_path, "resources", "models", "ABC", "labels.txt"))
        
        self.offset = 20
        self.imgSize = 300
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                       "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.camera_running = False  # Estado de la cámara

        # Panel izquierdo
        self.sidebar = tk.Frame(root, width=200, bg="#0C9F0F")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Título y logotipo
        title_frame = tk.Frame(self.sidebar, bg="#0C9F0F")
        title_frame.pack(pady=(20, 20))

        title_font = ("Helvetica", 16, "bold")
        title_label = tk.Label(title_frame, text="ManoTalk", font=title_font, fg="black", bg="#0C9F0F")
        title_label.pack(side="left", padx=(10, 10))

        icon_photo = None
        power_icon_photo = None

        # Cargar icono (opcional)
        try:
            icon_image = Image.open(os.path.join(base_path, "resources", "img", "MT2.0.png"))  # Ruta a la imagen del ícono
            icon_image = icon_image.resize((40, 40), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(title_frame, image=icon_photo, bg="#0C9F0F")
            icon_label.image = icon_photo
            icon_label.pack(side="left")

            # Cargar el ícono de apagado
            power_icon = Image.open(os.path.join(base_path, "resources/img/Botones/power.png"))  # Ruta al ícono de apagado
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

        self.round_image_inicio = PhotoImage(file=os.path.join(base_path, "resources/img/Botones/InicioBtn.png"))
        self.btn_inicio = Button(self.sidebar, image=self.round_image_inicio, command=mostrar_cam, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_gestos = PhotoImage(file=os.path.join(base_path, "resources/img/Botones/GestosBtn.png"))
        self.btn_gestos = Button(self.sidebar, image=self.round_image_gestos, command=mostrar_acb, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_guias = PhotoImage(file=os.path.join(base_path, "resources/img/Botones/GuiasBtn.png"))
        self.btn_guias = Button(self.sidebar, image=self.round_image_guias, command=mostrar_guias, borderwidth=0, highlightthickness=0, bg="#0C9F0F")
        
        self.round_image_info = PhotoImage(file=os.path.join(base_path, "resources/img/Botones/infoBtn.png"))
        self.btn_info = Button(self.sidebar, image=self.round_image_info, command=self.show_info, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.btn_salir = Button(self.sidebar, text=" Salir", image=power_icon_photo, command=self.quit, bg="#0C9F0F", fg="#ffffff", font=("Helvetica", 12, "bold"), borderwidth=0, compound="right", padx=10, width=150)
        self.btn_salir.image = power_icon_photo  # Guardar una referencia para evitar que se elimine

        # Agregar los botones al panel izquierdo
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

        # Contenedor para imágenes con desplazamiento
        self.image_canvas = Canvas(self.video_frame, width=150, height=400, bg="white")
        self.image_canvas.grid(row=0, column=1, sticky="w", padx=(0, 0))  # Ajuste en columna 1

        # Barra de desplazamiento a la izquierda del canvas
        scrollbar = Scrollbar(self.video_frame, orient="vertical", command=self.image_canvas.yview)
        scrollbar.grid(row=0, column=0, sticky="ns")  # Colocar la scrollbar en la columna 0
        self.image_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame contenedor de las imágenes
        self.image_frame = Frame(self.image_canvas, bg="white")
        self.image_canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

        # Label para mostrar el video en el main_frame
        self.video_label = Label(self.video_frame)
        self.video_label.grid(row=0, column=2)  # Mantener el video_label en la columna 2
        # Cargar las imágenes desde la carpeta
        self.load_images(os.path.join(base_path, "resources/img/Abecedario LSC"))  # Cambia "path/to/images_folder" por la ruta de la carpeta

        # Imagen adicional encima del botón "Iniciar Cámara"
        try:
            cam_icon = Image.open("")  # Cambia por la ruta de tu imagen
            cam_icon = cam_icon.resize((400, 200), Image.Resampling.LANCZOS)
            cam_photo = ImageTk.PhotoImage(cam_icon)
            self.cam_icon_label = Label(self.main_frame, image=cam_photo, bg="white")
            self.cam_icon_label.image = cam_photo  # Guarda la referencia
            self.cam_icon_label.pack(pady=(10, 0))  # Espacio superior
        except Exception as e:
            print(f"Error al cargar la imagen de la cámara: {e}")

        # Botón para iniciar la cámara (ya existente, solo relocalizado)
        self.start_button = Button(self.main_frame, text="Iniciar Cámara", font=("Helvetica", 14, "bold"), command=self.start_camera, bg="#0C9F0F", fg="white")
        self.start_button.pack(pady=(10, 20))

        # Ajusta la posición de las imágenes del abecedario en el Canvas
        self.image_canvas.grid(row=0, column=1, sticky="e", padx=(20, 0))  # Desplazado hacia la izquierda


        # Botón para iniciar la cámara en el centro y más abajo
        #self.start_button = Button(self.main_frame, text="Iniciar Cámara", font=("Helvetica", 14, "bold"), command=self.start_camera, bg="#0C9F0F", fg="white")

        # Centrando el botón en el eje horizontal y colocándolo más abajo
        self.start_button.place(relx=0.5, rely=0.8, anchor="center")
        self.start_button.pack(pady=20)

        # Botón para detener la cámara (se crea pero no se muestra aún)
        self.stop_button = Button(self.main_frame, text="Detener Cámara", font=("Helvetica", 14, "bold"), command=self.stop_camera, bg="#FF6347", fg="white")
        
    def load_images(self, folder_path):
        images = os.listdir(folder_path)
        for image_file in images:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            image = image.resize((100, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = Label(self.image_frame, image=photo, bg="white")
            label.image = photo  # Guardar una referencia
            label.pack(pady=10, padx=(10, 0))  # Ajuste de 'padx' para desplazar a la izquierda
        # Configurar el tamaño del canvas para el scroll
        self.image_frame.update_idletasks()
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

        
    def load_labels(label_file_path):
        with open(label_file_path, 'r') as file:
            labels = [line.strip() for line in file.readlines()]
        return labels
    
    classifier = Classifier(os.path.join(base_path, "resources/models/ABC/keras_model.h5"), os.path.join(base_path, "resources/models/ABC/labels.txt"))
    
    # Carga las etiquetas desde el archivo labels.txt
    labels = load_labels(os.path.join(base_path, "resources/models/ABC/labels.txt"))
    
    def start_camera(self):
        # Iniciar la cámara
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_running = True
            self.start_button.pack_forget()  # Ocultar el botón de iniciar
            self.stop_button.pack(pady=20)  # Mostrar el botón de detener
            self.update_frame()  # Inicia el bucle de actualización

    def stop_camera(self):
        # Detener la cámara
        if self.camera_running:
            self.camera_running = False
            self.cap.release()  # Liberar la cámara
            self.video_label.config(image="")  # Limpiar el Label de video
            self.stop_button.pack_forget()  # Ocultar el botón de detener
            self.start_button.pack(pady=20)  # Mostrar el botón de iniciar nuevamente
            
    def update_frame(self):
        # Obtener el frame de la cámara y mostrarlo en el Label
        if self.camera_running:
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)
                imgOutput = img.copy()
                
                # Detectar manos sin dibujo adicional
                hands, img = self.detector.findHands(img, draw=False)

                if hands:
                    for hand in hands:
                        x, y, w, h = hand['bbox']
                        imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                        imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                        # Ajuste de tamaño según la relación de aspecto
                        aspectRatio = h / w
                        if aspectRatio > 1:
                            k = self.imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                            wGap = math.ceil((self.imgSize - wCal) / 2)
                            imgWhite[:, wGap: wCal + wGap] = imgResize
                        else:
                            k = self.imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                            hGap = math.ceil((self.imgSize - hCal) / 2)
                            imgWhite[hGap: hCal + hGap, :] = imgResize

                        # Realizar la predicción del gesto
                        prediction, index = self.classifier.getPrediction(imgWhite, draw=False)

                        # Verificación del índice para mostrar el nombre del gesto
                        if 0 <= index < len(self.labels):
                            gesture_name = self.labels[index]  # Obtener el nombre del gesto según el índice
                            # Dibujar el nombre del gesto sobre la mano en imgOutput
                            cv2.putText(imgOutput, gesture_name, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                        
                        # Dibujar el cuadro delimitador alrededor de la mano
                        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (255, 0, 255), 2)

                # Mostrar el frame actualizado en el Label
                imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                imgOutput = Image.fromarray(imgOutput)
                imgOutput = ImageTk.PhotoImage(imgOutput)
                self.video_label.config(image=imgOutput)
                self.video_label.image = imgOutput

            self.video_label.after(10, self.update_frame)  # Actualiza el frame cada 10ms

    def quit(self):
        self.root.quit()

    def show_home(self):
        messagebox.showinfo("Inicio", "Bienvenido a ManoTalk")

    def show_gestures(self):
        messagebox.showinfo("Gestos", "Gestos disponibles: " + ", ".join(gestures))

    def show_guides(self):
        messagebox.showinfo("Guías", "Guías disponibles para el uso de la aplicación.")

    def show_info(self):
        url = "https://manosypensamiento.upn.edu.co/lengua-de-senas-colombiana/"  # Sustituye con el enlace que quieras
        webbrowser.open(url)

    def mostrar_carpetas(self):
        ruta_data = 'Data'
        if os.path.exists(ruta_data) and os.path.isdir(ruta_data):
            carpetas = [nombre for nombre in os.listdir(ruta_data) if os.path.isdir(os.path.join(ruta_data, nombre))]
            messagebox.showinfo("Carpetas en Data", "\n".join(carpetas))
        else:
            messagebox.showerror("Error", "La carpeta 'Data' no existe o no es un directorio.")

def start_acb():
    global root
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app)
    root.mainloop()

if __name__ == "__main__":
    start_acb()

