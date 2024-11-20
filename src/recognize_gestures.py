import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from utils.keypoints import extract_keypoints
from ui_manager import UIManager
import time

# Cargar el modelo entrenado
model = tf.keras.models.load_model('../models/gesture_model.h5')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar los nombres de los gestos
DATA_PATH = '../data'
gestures = os.listdir(DATA_PATH)

# Imprimir el número de gestos
print(f"Número de gestos: {len(gestures)}")

# Inicializar la interfaz gráfica
ui = UIManager()

# Capturar video
cap = cv2.VideoCapture(0)
sequence = []
sequence_length = 116
hands_in_frame = False  # Estado para saber si hay manos en cuadro
last_prediction_time = time.time()  # Para controlar el tiempo de predicciones
prediction_interval = 0.5  # Intervalo mínimo entre predicciones en segundos

while cap.isOpened():
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dibujar las manos en el frame
    if results.multi_hand_landmarks:
        hands_in_frame = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extraer keypoints
        keypoints = extract_keypoints(results)

        # Asegurarse de que los keypoints tengan la longitud correcta antes de añadirlos
        if keypoints.shape[0] == 63:  # Verificar que hay 63 elementos
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]  # Mantener solo los últimos 'sequence_length' elementos

    else:
        # Si no hay manos en cuadro y hay una secuencia llena, hacer la predicción
        if hands_in_frame and len(sequence) == sequence_length:
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:  # Verificar el intervalo
                sequence_np = np.array(sequence, dtype=np.float32)  # Convertir a array de numpy
                res = model.predict(np.expand_dims(sequence_np, axis=0))[0]

                # Asegúrate de que la predicción tenga la misma longitud que la lista de gestos
                if len(res) == len(gestures):
                    gesture = gestures[np.argmax(res)]  # Mostrar el gesto reconocido
                    print(f"Gesto reconocido: {gesture}")

                    # Mostrar el frame con el gesto reconocido en la interfaz
                    ui.show_frame(frame, gesture_text=gesture)
                else:
                    print(f"Error: La longitud de res ({len(res)}) no coincide con la longitud de gestures ({len(gestures)}).")

                last_prediction_time = current_time  # Actualizar el tiempo de la última predicción

            # Reiniciar la secuencia
            sequence = []

        hands_in_frame = False  # No hay manos en cuadro

    # Mostrar la imagen en la interfaz (sin gesto si no hay reconocimiento)
    ui.show_frame(frame)

    if ui.wait_key(10) == ord('q'):
        break

cap.release()
ui.close()
