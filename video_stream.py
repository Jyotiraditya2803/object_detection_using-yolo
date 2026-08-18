# video_stream.py - Threaded Asynchronous Camera Reader
import cv2
import threading
import time

class VideoStream:
    """
    Background-threaded camera capture to prevent OpenCV video lag.
    Standard video capture buffers frames, causing increasing lag over time.
    This class continuously grabs the LATEST frame in a separate thread.
    """
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        """Starts the background thread."""
        thread = threading.Thread(target=self.update, args=(), daemon=True)
        thread.start()
        return self

    def update(self):
        """Continuously reads frames from camera buffer."""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005) # Prevent 100% CPU lock on background thread

    def read(self):
        """Thread-safe retrieval of the most recent frame."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        """Releases the camera and terminates the background thread."""
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
