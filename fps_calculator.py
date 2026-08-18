# fps_calculator.py - FPS and Latency Utility
import time

class FPSCalculator:
    """Calculates smoothed frames per second (FPS) and latency in milliseconds."""
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.prev_time = time.time()

    def update(self):
        """Updates frame timer and returns (fps, latency_ms)."""
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time
        
        if delta > 0:
            latency_ms = delta * 1000.0
            self.frame_times.append(delta)
            if len(self.frame_times) > self.buffer_size:
                self.frame_times.pop(0)
            avg_delta = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_delta if avg_delta > 0 else 0.0
            return fps, latency_ms
        return 0.0, 0.0
