import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import cam

# Crear la ventana principal
root = tk.Tk()
root.title("ManoTalk")
root.geometry("800x600")
root.iconbitmap("src\icon\MT2.0.ico")

# Intentar cargar la imagen de fondo
try:
    bg_image = Image.open("src\\img\\MT2.0.png")  # Reemplaza con la ruta a tu imagen
    bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)  # Ajustar el tamaño de la imagen
    bg_image = bg_image.convert("RGBA")  # Convertir la imagen a modo RGBA
    alpha = bg_image.split()[3]  # Obtener el canal alfa
    alpha = alpha.point(lambda p: p * 0.5)  # Ajustar la transparencia al 50%
    bg_image.putalpha(alpha)  # Aplicar la transparencia a la imagen
    bg_photo = ImageTk.PhotoImage(bg_image)
    # Crear un Label con la imagen de fondo
    background_label = tk.Label(root, image=bg_photo)
    background_label.image = bg_photo  # Guardar una referencia de la imagen
    background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Ajustar la imagen al tamaño de la ventana
except Exception as e:
    print(f"Error al cargar la imagen: {e}")

# Crear el texto "ManoTalk" y centrarlo
title_font = font.Font(family="Helvetica", size=24, weight="bold")
title_label = tk.Label(root, text="ManoTalk", font=title_font, fg="#ffffff",bg="#0C9F0F", bd=0)  # Sin fondo ni borde
title_label.pack(pady=(150, 0))  # Espacio arriba, no especificado abajo para centrar

# Crear el texto de descripción y centrarlo
desc_font = font.Font(family="Helvetica", size=12)
desc_label = tk.Label(root, text="Plataforma para el aprendizaje básico de lenguaje\nde señas colombiano LCS",
                      font=desc_font, fg="#ffffff",bg="#0C9F0F", bd=0)  # Sin fondo ni borde
desc_label.pack(pady=(10, 0))  # Espacio arriba, no especificado abajo para centrar


# Crear botón de "Empezar" y centrarlo
def on_start():
    print("¡Empezar!")
    root.destroy()  # Cerrar la ventana actual
    cam.start_cam()  # Llamar a la función para iniciar la vista cam

start_button = tk.Button(root, text="Empezar", font=("Helvetica", 14), fg="#ffffff",bg="#0C9F0F", command=on_start)
start_button.pack(pady=(20, 0))  # Espacio arriba, no especificado abajo para centrar

root.mainloop()
