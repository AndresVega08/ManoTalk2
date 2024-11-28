import os
import time
import tkinter as tk
from tkinter import Label, PhotoImage, ttk, Button
from tkinter import messagebox
import webbrowser
from PIL import Image, ImageTk
import imageio
import threading

DATA_PATH = 'data'
gestures = os.listdir(DATA_PATH)

def mostrar_cam():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        app.destroy()
        import cam
        cam.start_cam()

def mostrar_acb():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        app.destroy()
        import Acb
        Acb.start_acb()

def mostrar_guias():
        # Aquí puedes agregar el código necesario para cambiar a la vista guias.py
        # Por ejemplo, podrías destruir la vista actual y crear una nueva instancia de la vista guias
        app.destroy()
        import guias
        guias.start_guias()        

class VideoPlayer(tk.Tk):
    def __init__(self, videos):
        super().__init__()
        self.videos = videos
        self.title("ManoTalk")
        self.geometry("1200x700")
        self.iconbitmap("src\icon\MT2.0.ico")

        # Variables iniciales
        self.sequence = []
        self.sequence_length = 60
        self.hands_in_frame = False
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5

        # Panel izquierdo
        self.sidebar = tk.Frame(self, width=200, bg="#0C9F0F")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Título y logotipo
        title_frame = tk.Frame(self.sidebar, bg="#0C9F0F")
        title_frame.pack(pady=(20, 20))

        title_font = ("Helvetica", 16, "bold")
        title_label = tk.Label(title_frame, text="ManoTalk", font=title_font, fg="black", bg="#0C9F0F")
        title_label.pack(side="left", padx=(10, 10))

        # Cargar icono (opcional)
        try:
            icon_image = Image.open("src\img\MT2.0.png")
            icon_image = icon_image.resize((40, 40), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(title_frame, image=icon_photo, bg="#0C9F0F")
            icon_label.image = icon_photo
            icon_label.pack(side="left")

            # Cargar el ícono de apagado
            power_icon = Image.open("src/img/Botones/power.png")
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
        self.btn_inicio = Button(self.sidebar, image=self.round_image_inicio, command=mostrar_cam, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_gestos = PhotoImage(file="src/img/Botones/GestosBtn.png")
        self.btn_gestos = Button(self.sidebar, image=self.round_image_gestos, command=mostrar_acb, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.round_image_guias = PhotoImage(file="src/img/Botones/GuiasBtn.png")
        self.btn_guias = Button(self.sidebar, image=self.round_image_guias, command=mostrar_guias, borderwidth=0, highlightthickness=0, bg="#0C9F0F")
        
        self.round_image_info = PhotoImage(file="src/img/Botones/infoBtn.png")
        self.btn_info = Button(self.sidebar, image=self.round_image_info, command=self.show_info, borderwidth=0, highlightthickness=0, bg="#0C9F0F")

        self.btn_salir = Button(self.sidebar, text=" Salir", image=power_icon_photo, command=self.destroy, bg="#0C9F0F", fg="#ffffff", font=("Helvetica", 12, "bold"), borderwidth=0, compound="right", padx=10, width=150)
        self.btn_salir.image = power_icon_photo

        self.btn_inicio.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_gestos.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_guias.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_info.pack(pady=5, padx=20, ipadx=10, ipady=10)
        self.btn_salir.pack(pady=(10, 20), padx=20, ipadx=10, ipady=10, side="bottom")

        # Panel derecho
        self.main_frame = tk.Frame(self, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Añadir texto principal
        self.text_label = Label(self.main_frame, text="Reconocimiento de Gestos", font=("Arial", 24), bg="#0C9F0F", fg="#ffffff")
        self.text_label.pack(fill=tk.X)

        # Label para mostrar el video debajo del texto principal
        self.label = tk.Label(self.main_frame)
        self.label.pack(pady=10)

        # Contenedor para los botones
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        # Crear botones de reproducción para cada video con su título y miniatura
        self.buttons = []
        self.thumbnails = []
        for i, video in enumerate(videos):
            path = video["path"]
            title = video["title"]

            # Generar miniatura para cada video
            thumbnail = self.get_thumbnail(path)
            button = ttk.Button(self.button_frame, text=title, command=lambda p=path: self.play_video(p))

            # Configurar la miniatura en el botón
            if thumbnail:
                self.thumbnails.append(thumbnail)
                button.config(image=thumbnail, compound="top")

            # Colocar los botones en una cuadrícula de tres columnas por fila
            row = i // 6
            col = i % 6
            button.grid(row=row, column=col, padx=20, pady=5)
            self.buttons.append(button)

        # Botón para detener el video
        self.stop_button = ttk.Button(self.main_frame, text="Detener", command=self.stop_video, state="disabled")
        self.stop_button.pack(pady=10)

        self.video_thread = None
        self.is_playing = False
        self.frames = []

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

    def get_thumbnail(self, video_path):
        try:
            video = imageio.get_reader(video_path, "ffmpeg")
            frame = video.get_data(0)
            img = Image.fromarray(frame)
            img = img.resize((120, 150), Image.LANCZOS)
            video.close()
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error al cargar miniatura para {video_path}: {e}")
            return None

    def play_video(self, video_path):
        # Detener cualquier video en reproducción
        self.stop_video()

        # Activar el botón de "Detener"
        self.stop_button.config(state="normal")

        # Iniciar la reproducción del nuevo video
        self.is_playing = True
        self.video_thread = threading.Thread(target=self._play_video, args=(video_path,))
        self.video_thread.start()

    def _play_video(self, video_path):
        try:
            video = imageio.get_reader(video_path, "ffmpeg")
            fps = video.get_meta_data().get("fps", 60)
            delay = 0 / fps

            self.frames.clear()

            for frame in video.iter_data():
                if not self.is_playing:
                    break

                img = Image.fromarray(frame)
                img = img.resize((280, 400), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)

                # Mantener referencia del fotograma actual
                self.frames.append(imgtk)
                
                # Actualizar el label para mostrar el fotograma redimensionado
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

                self.update()
                self.after(int(delay * 0))

            video.close()
        except Exception as e:
            print(f"Error al reproducir video {video_path}: {e}")
        finally:
            self.is_playing = False
            self.stop_button.config(state="disabled")

            # Limpiar la imagen al finalizar y devolver la vista inicial
            self.label.config(image="")

    def stop_video(self):
        if self.is_playing:
            self.is_playing = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join()

        self.label.config(image="")  # Limpiar la imagen del label al detener

# Lista de videos con título y ruta
videos = [
    {"title": "Tu", "path": "src/guia/Tu.mp4"},
    {"title": "Ustedes", "path": "src/guia/Ustedes.mp4"},
    {"title": "Tu Nombre", "path": "src/guia/Tu nombre.mp4"},
    {"title": "Tu seña", "path": "src/guia/Tu seña.mp4"},
    {"title": "Mi nombre", "path": "src/guia/Mi nombre.mp4"},
    {"title": "Mi seña", "path": "src/guia/Mi seña.mp4"},
    {"title": "Por favor", "path": "src/guia/Por favor.mp4"},
    {"title": "Gracias", "path": "src/guia/Gracias.mp4"},
    {"title": "Mio", "path": "src/guia/Mio.mp4"},
    {"title": "El", "path": "src/guia/El.mp4"},
    {"title": "Ella", "path": "src/guia/Ella.mp4"},
    {"title": "Nosotros", "path": "src/guia/Nosotros.mp4"},
    {"title": "Nuestro", "path": "src/guia/Nuestro.mp4"},
    {"title": "Permiso", "path": "src/guia/Permiso.mp4"},
    {"title": "Suyo", "path": "src/guia/Suyo.mp4"},
    {"title": "Ellos", "path": "src/guia/Ellos.mp4"},

]

# Crear la aplicación
app = VideoPlayer(videos)
app.mainloop()
