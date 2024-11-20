import numpy as np


# Función para extraer los keypoints de un resultado de MediaPipe
def extract_keypoints(results):
    keypoints = []

    # Verificar si hay manos en el cuadro
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer keypoints para cada mano
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])  # Almacenar las coordenadas x, y, z

    # Asegurarse de que siempre devolvemos 63 valores
    keypoints_flattened = np.array(keypoints).flatten()

    # Rellenar o recortar para asegurarse de que tengamos 63 valores
    if len(keypoints_flattened) < 63:
        # Rellenar con ceros si hay menos de 63 valores
        keypoints_flattened = np.pad(keypoints_flattened, (0, 63 - len(keypoints_flattened)), 'constant')
    elif len(keypoints_flattened) > 63:
        # Recortar si hay más de 63 valores
        keypoints_flattened = keypoints_flattened[:63]

    return keypoints_flattened  # Devolver siempre un array de 63 elementos
