import cv2
import time
import threading
import numpy as np
import os # Added for file path management
from flask import Flask, Response, render_template_string

# --- Configuration and Initialization ---

CAMERA_INDEX = 0
FRAME_DELAY = 1 / 10.0 # Target 30 FPS for live feed
app = Flask(__name__)

# --- YOLO/DNN Configuration ---
# NOTE: To enable object detection, you must download these files (e.g., YOLOv3-tiny) 
# and place them in the same directory as this script.
MODEL_CFG = 'yolov3.cfg'
MODEL_WEIGHTS = 'yolov3.weights'
NAMES_FILE = 'coco.names' # File containing class names (e.g., person, dog, car)

# --- Video Recording Configuration ---
REC_FPS = 5.0                           # Frames per second for the recording buffer
REC_FILENAME = "circular_buffer.avi"    # The file name for the continuous buffer
REC_MAX_DURATION = 3600                 # Maximum duration of the buffer in seconds (1 hour)
REC_CODEC = 'XVID'                      # Codec used for video compression

# --- Camera Streamer Class ---

class CameraStreamer(threading.Thread):
    """
    Handles camera access, YOLO processing, and caching the JPEG frame for web display.
    Also caches the raw frame for the VideoRecorder thread.
    """
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.frame = None                   # Shared cache for the latest JPEG frame (bytes)
        self.raw_frame = None               # Shared cache for the latest RAW BGR frame (numpy array)
        self.is_available = False           # Flag indicating if the camera is successfully opened
        self.is_detection_available = False # Flag indicating if YOLO model was loaded
        self.running = True                 # Flag to control the thread's main loop
        
        # Locks for thread-safe access to shared data
        self.jpeg_lock = threading.Lock()   
        self.raw_lock = threading.Lock()    

        self.camera = None                  # OpenCV VideoCapture object
        self.net = None                     
        self.ln = None                      
        self.labels = []                    
        
        self.daemon = True 
        
    def _load_detection_model(self):
        """Loads the YOLO model using OpenCV's DNN module."""
        try:
            with open(NAMES_FILE, 'r') as f:
                self.labels = f.read().splitlines()
            self.net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            ln = self.net.getLayerNames()
            self.ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.is_detection_available = True
            print("YOLO Detection Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load detection model files. Falling back to plain streaming. Error: {e}")
            self.is_detection_available = False

    def _process_frame_with_yolo(self, frame):
        """Runs the detection on the frame and draws bounding boxes (omitted for brevity)."""
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        # Check for NaN/Inf (Numerical Stability Fix)
        for output in layer_outputs:
            if np.isnan(output).any() or np.isinf(output).any():
                print("--- WARNING: YOLO layer output contained NaN/Inf. Skipping detection for this frame. ---")
                return frame

        boxes = []
        confidences = []
        classIDs = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                if classIDs[i] < len(self.labels):
                    hue = (classIDs[i] * 40) % 360 
                    color = tuple(map(int, cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]))
                else:
                    color = (255, 0, 0)

                text = f"{self.labels[classIDs[i]]}: {confidences[i]*100:.1f}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return frame

    def _create_error_frame(self):
        """Creates a static JPEG error message frame for the cache."""
        error_frame = cv2.UMat(480, 640, cv2.CV_8UC3, (0, 0, 200)).get()
        cv2.putText(error_frame, "CAMERA UNAVAILABLE", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', error_frame)
        with self.jpeg_lock:
             self.frame = buffer.tobytes()

    def run(self):
        self._load_detection_model()
        self.camera = cv2.VideoCapture(self.camera_index)
        self.is_available = self.camera.isOpened()
        
        if not self.is_available:
            self._create_error_frame()
            print(f"Error: Could not open camera {self.camera_index}. Streamer thread stopping.")
            return

        print(f"Camera Streamer started. Capturing frames from index {self.camera_index}.")

        while self.running:
            try:
                start_time = time.time()
                success, frame = self.camera.read()

                if not success:
                    self.stop()
                    break

                # 1. Flip frame and cache the RAW BGR frame for the recorder
                frame = cv2.flip(frame, 1)
                with self.raw_lock:
                    self.raw_frame = frame.copy() 
                
                # 2. OBJECT DETECTION STEP (uses a copy of the frame)
                display_frame = frame.copy()
                if self.is_detection_available:
                    display_frame = self._process_frame_with_yolo(display_frame)

                # 3. Encode the display frame as JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame)
                
                # Safely update the shared JPEG cache
                with self.jpeg_lock:
                    self.frame = buffer.tobytes()

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
        """Safely retrieve the latest JPEG frame from the cache."""
        with self.jpeg_lock:
            return self.frame

    def get_raw_frame(self):
        """Safely retrieve the latest RAW BGR frame from the cache for the recorder."""
        with self.raw_lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None

    def stop(self):
        """Stop the thread and release the camera resource."""
        self.running = False
        if self.camera and self.is_available:
            print("Releasing camera resource...")
            self.camera.release()
            self.is_available = False

# --- Video Recorder Class (New Component) ---

class VideoRecorder(threading.Thread):
    """
    Dedicated thread for writing frames to a video file for a circular buffer.
    """
    def __init__(self, streamer, fps, filename, max_duration, codec):
        super().__init__()
        self.streamer = streamer
        self.fps = fps
        self.filename = filename
        self.max_duration = max_duration
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writer = None
        self.start_time = time.time()
        self.last_write_time = 0
        self.running = True
        self.daemon = True

    def _initialize_writer(self, width, height):
        """Initializes or re-initializes the VideoWriter (overwriting the file)."""
        if self.writer:
            self.writer.release()
            print(f"Buffer reset: Closed old file {self.filename}.")

        self.writer = cv2.VideoWriter(self.filename, self.codec, self.fps, (width, height))
        self.start_time = time.time()
        print(f"Recording started on {self.filename} at {self.fps} FPS. (Buffer will reset in {self.max_duration/3600:.1f} hours)")

    def run(self):
        # Wait for the camera to open and get the first frame size
        while self.streamer.raw_frame is None and self.running:
            time.sleep(0.1)
        
        if not self.running:
            return

        h, w, _ = self.streamer.raw_frame.shape
        self._initialize_writer(w, h)

        while self.running:
            try:
                # 1. Circular Buffer Management Check
                if time.time() - self.start_time > self.max_duration:
                    print("Circular buffer duration reached (1 hour). Resetting recording file.")
                    self._initialize_writer(w, h)
                
                # 2. Frame Rate Decimation Check (ensures we write only at REC_FPS)
                if time.time() - self.last_write_time >= (1.0 / self.fps):
                    raw_frame = self.streamer.get_raw_frame()
                    if raw_frame is not None:
                        self.writer.write(raw_frame)
                        self.last_write_time = time.time()

                # Sleep to yield control, waiting for the next write interval
                time.sleep(0.005) 

            except Exception as e:
                print(f"Recorder thread error: {e}. Stopping recorder.")
                self.stop()
                break

    def stop(self):
        self.running = False
        if self.writer:
            self.writer.release()
            print(f"VideoRecorder stopped and file {self.filename} closed.")

# 2. Initialize and Start the Global Streamer and Recorder Threads
STREAMER = CameraStreamer(CAMERA_INDEX)
RECORDER = VideoRecorder(
    streamer=STREAMER, 
    fps=REC_FPS, 
    filename=REC_FILENAME, 
    max_duration=REC_MAX_DURATION, 
    codec=REC_CODEC
)

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
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .recording-indicator {
            animation: pulse-red 2s infinite;
        }
        @keyframes pulse-red {
            0%, 100% { background-color: #f87171; }
            50% { background-color: #b91c1c; }
        }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-2xl p-6 md:p-10">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-2 text-center">
            Object Detection Live Stream
        </h1>
        <p class="text-lg text-gray-500 mb-8 text-center">
            Serving frames processed by YOLO in a single background thread.
        </p>

        <div class="flex justify-center mb-6">
            <span class="status-pill bg-green-100 text-green-800">
                <svg class="w-2.5 h-2.5 mr-1.5" viewBox="0 0 6 6" fill="currentColor"><circle cx="3" cy="3" r="3"></circle></svg>
                Live Stream (30 FPS)
            </span>
            <span class="status-pill recording-indicator ml-4 text-white">
                <svg class="w-2.5 h-2.5 mr-1.5" viewBox="0 0 6 6" fill="currentColor"><circle cx="3" cy="3" r="3"></circle></svg>
                Recording Buffer (5 FPS)
            </span>
        </div>

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
                A video file named **circular_buffer.avi** is continuously overwritten to store the last 1 hour of footage.
            </p>
        </div>
    </div>
</body>
</html>
"""

# --- Frame Yielding Generator and Flask Routes (unchanged) ---

def generate_frames():
    """Generator function for the web response."""
    while STREAMER.running:
        frame_bytes = STREAMER.get_frame()
        
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(FRAME_DELAY) 

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """The main route that serves the continuous video stream from the cache."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# --- Execution ---

if __name__ == '__main__':
    # Start the dedicated camera and recorder threads
    STREAMER.start()
    RECORDER.start()
    
    try:
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Flask app startup error: {e}")
    finally:
        # Ensure all threads are stopped and resources are released
        STREAMER.stop()
        RECORDER.stop()