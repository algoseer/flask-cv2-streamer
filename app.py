import cv2
import time
import threading
from flask import Flask, Response, render_template_string

# --- Configuration and Initialization ---

CAMERA_INDEX = 0
FRAME_DELAY = 1 / 10.0 # Target 30 FPS
app = Flask(__name__)

# --- Camera Streamer Class ---

class CameraStreamer(threading.Thread):
    """
    A dedicated background thread to handle camera access, frame processing,
    and frame caching for multiple web clients.
    """
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.frame = None              # Shared cache for the latest JPEG frame (bytes)
        self.is_available = False      # Flag indicating if the camera is successfully opened
        self.running = True            # Flag to control the thread's main loop
        self.lock = threading.Lock()   # Lock for thread-safe access to the shared frame cache
        self.camera = None             # OpenCV VideoCapture object
        
        # Daemonize the thread so it quits automatically when the main process exits
        self.daemon = True 

    def run(self):
        """
        The main loop executed by the background thread.
        This is the ONLY function that calls CAMERA.read().
        """
        self.camera = cv2.VideoCapture(self.camera_index)
        self.is_available = self.camera.isOpened()
        
        if not self.is_available:
            print(f"Error: Could not open camera {self.camera_index}. Streamer thread stopping.")
            self._create_error_frame()
            return

        print(f"Camera Streamer started. Capturing frames from index {self.camera_index}.")

        while self.running:
            try:
                start_time = time.time()

                # Read the frame
                success, frame = self.camera.read()

                if not success:
                    print("Warning: Failed to read frame mid-stream. Attempting to stop gracefully.")
                    self.stop()
                    break

                # Process the frame (e.g., flip it, add timestamp)
                frame = cv2.flip(frame, 1)

                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Safely update the shared cache
                with self.lock:
                    self.frame = frame_bytes

                # Frame Rate Control
                elapsed_time = time.time() - start_time
                sleep_time = FRAME_DELAY - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"Streamer thread error: {e}. Shutting down camera access.")
                self.stop()
                break

    def get_frame(self):
        """
        Safely retrieve the latest frame from the cache.
        """
        if not self.is_available:
             # Return error frame if camera failed to initialize
             return self.frame
             
        with self.lock:
            # Return the cached frame or None if not yet initialized
            return self.frame

    def stop(self):
        """
        Stop the thread and release the camera resource.
        """
        self.running = False
        if self.camera and self.is_available:
            print("Releasing camera resource...")
            self.camera.release()
            self.is_available = False # Update status

    def _create_error_frame(self):
        """Creates a static JPEG error message frame for the cache."""
        error_frame = cv2.UMat(480, 640, cv2.CV_8UC3, (0, 0, 200)).get() # Dark Red Background
        cv2.putText(error_frame, "CAMERA UNAVAILABLE", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', error_frame)
        self.frame = buffer.tobytes()

# 2. Initialize and Start the Global Streamer Thread
STREAMER = CameraStreamer(CAMERA_INDEX)

# --- HTML Template for the Frontend ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Camera Stream</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .video-frame {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 4px solid #3b82f6; 
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
        }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-2xl p-6 md:p-10">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-2 text-center">
            Multi-Client Camera Stream
        </h1>
        <p class="text-lg text-gray-500 mb-8 text-center">
            Serving frames from a single background thread to prevent segmentation faults.
        </p>

        <div class="flex justify-center">
            <img 
                src="{{ url_for('video_feed') }}" 
                class="video-frame w-full max-w-2xl h-auto rounded-lg object-contain bg-gray-100" 
                alt="Live Camera Feed"
                onerror="this.onerror=null; this.src='https://placehold.co/640x480/EF4444/FFFFFF?text=STREAM+BROKEN'"
            >
        </div>

        <div class="mt-8 text-center">
            <p class="text-sm text-gray-400">
                You can open this page in multiple browser tabs or devices simultaneously without issue.
            </p>
        </div>
    </div>
</body>
</html>
"""

# --- Frame Yielding Generator ---

def generate_frames():
    """
    Generator function for the web response. 
    It reads the pre-cached frame from the streamer thread (no CV2 call here).
    """
    while STREAMER.running:
        frame_bytes = STREAMER.get_frame()
        
        if frame_bytes is not None:
            # Yield the cached frame in the required multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Wait a small amount of time to prevent hogging the CPU 
        # while waiting for the next cached frame update.
        time.sleep(FRAME_DELAY) 


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """
    The main route that serves the continuous video stream from the cache.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# --- Execution ---

if __name__ == '__main__':
    # Start the dedicated camera thread
    STREAMER.start()
    
    try:
        # Use use_reloader=False to prevent the 'leaked semaphore' warning 
        # and simplify process management, which helps stability.
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Flask app startup error: {e}")
    finally:
        # Ensure the camera thread is stopped and resources are released
        STREAMER.stop()