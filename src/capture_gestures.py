import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
from utils.keypoints import extract_keypoints

# Inicializar MediaPipe y configuraciones
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def capturar_gesto(gesture_name):
    # Directorio para almacenar datos
    DATA_PATH = '../data'
    num_sequences = 30  # Número de secuencias por gesto
    sequence_length = 120  # Longitud de la secuencia (frames)

    # Crear carpeta para el nuevo gesto
    os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)

    # Capturar los gestos
    cap = cv2.VideoCapture(0)

    def capturar_secuencia(sequence):
        current_sequence = []  # Reiniciar secuencia para cada ciclo
        print(f'Capturando secuencia {sequence + 1}/{num_sequences} para el gesto: {gesture_name}')

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el frame")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Dibujar keypoints en la pantalla
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer keypoints
                keypoints = extract_keypoints(results)
                if keypoints is not None and keypoints.shape[0] == 63:  # Verificar que haya 63 elementos
                    current_sequence.append(keypoints)

            # Mostrar la ventana con la webcam
            cv2.imshow('Capture Gestures', frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        return current_sequence

    for sequence in range(num_sequences):
        while True:
            secuencia_completa = capturar_secuencia(sequence)

            # Guardar la secuencia completa solo si tiene longitud suficiente
            if len(secuencia_completa) >= 60:  # Guardar si la longitud es mayor o igual a 60
                # Guardar la secuencia en un archivo .npy con el nombre del gesto
                np.save(os.path.join(DATA_PATH, gesture_name, f'{gesture_name}_sequence_{sequence}.npy'),
                        secuencia_completa)
                print(f'Secuencia {sequence + 1} guardada exitosamente.')
                break  # Salir del bucle si la secuencia fue capturada correctamente
            else:
                # Si no se capturó la secuencia completa, preguntar si desea intentar de nuevo
                retry = simpledialog.askstring("Captura incompleta",
                                               f"La secuencia {sequence + 1} no tiene la longitud correcta. ¿Deseas intentar capturarla de nuevo? (sí/no)")
                if retry.lower() != 'sí':
                    print(f'Secuencia {sequence + 1} omitida.')
                    break

    cap.release()
    cv2.destroyAllWindows()


# Bucle principal para capturar múltiples gestos
def main():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    while True:
        # Crear ventana emergente para capturar el nombre del gesto
        gesture_name = simpledialog.askstring("Captura de Gesto", "Introduce el nombre del gesto que deseas capturar:")

        if not gesture_name:
            print("No se ingresó ningún nombre de gesto. Cerrando...")
            break

        # Capturar el gesto
        capturar_gesto(gesture_name)

        # Preguntar si se desea agregar otro gesto o finalizar
        another_gesture = simpledialog.askstring("Agregar otro gesto", "¿Deseas agregar otro gesto? (sí/no)")
        if another_gesture.lower() != 'sí':
            print("Finalizando el programa.")
            break


if __name__ == "__main__":
    main()
