import cv2

class UIManager:
    def __init__(self, window_name="Gesture Recognition"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show_frame(self, frame, gesture_text=None):
        # Si hay un gesto reconocido, lo mostramos en el frame
        if gesture_text:
            cv2.putText(frame, f"Gesto: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Mostrar el frame en la ventana
        cv2.imshow(self.window_name, frame)

    def close(self):
        cv2.destroyAllWindows()

    def wait_key(self, delay=10):
        return cv2.waitKey(delay) & 0xFF
