import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Directorio donde están almacenadas las secuencias de gestos
DATA_PATH = '../data'
gestures = os.listdir(DATA_PATH)
expected_frame_shape = (63,)  # Forma de cada frame

sequences, labels = [], []

# Crear un diccionario para mapear nombres de gestos a índices numéricos
gesture_to_index = {gesture_name: index for index, gesture_name in enumerate(gestures)}
print("Gestos encontrados:", gestures)

# Recorrer todos los gestos y cargar las secuencias
for gesture_name in gestures:
    gesture_path = os.path.join(DATA_PATH, gesture_name)

    for sequence in os.listdir(gesture_path):
        sequence_path = os.path.join(gesture_path, sequence)

        # Cargar los archivos npy
        try:
            frames = np.load(sequence_path)
            sequences.append(frames)  # Agregar la secuencia, aunque varíe en longitud
            labels.append(gesture_to_index[gesture_name])
        except Exception as e:
            print(f"Error cargando {sequence_path}: {e}")

# Verificar si se cargaron secuencias y etiquetas
print(f"Total secuencias cargadas: {len(sequences)}")
print(f"Total etiquetas cargadas: {len(labels)}")

# Si no hay secuencias cargadas, no continuar
if not sequences or not labels:
    raise ValueError("No se encontraron secuencias o etiquetas válidas.")

# Normalizar los frames dentro de cada secuencia
for i, sequence in enumerate(sequences):
    for j, frame in enumerate(sequence):
        if np.array(frame).shape != expected_frame_shape:
            print(f"Corrigiendo frame {j} de la secuencia {i}")
            sequences[i][j] = np.zeros(expected_frame_shape)

# Encontrar la longitud máxima de secuencias para aplicar padding dinámico
max_sequence_length = max([len(seq) for seq in sequences])
print(f"Longitud máxima de secuencia: {max_sequence_length}")

# Convertir listas a arrays de NumPy aplicando padding dinámico
X = pad_sequences(sequences, maxlen=max_sequence_length, dtype='float32', padding='post')

# Verificar el número de clases
num_classes = len(gestures)
print(f"Número de clases (gestos): {num_classes}")

# Convertir etiquetas a formato categórico
y = to_categorical(labels, num_classes=num_classes).astype(int)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definir el modelo LSTM para aprender gestos en movimiento con enmascaramiento
model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_sequence_length, expected_frame_shape[0])),  # Ignorar los ceros en el padding
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Salida con tantas clases como gestos
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model_save_path = '../models/gesture_model.h5'
model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")
