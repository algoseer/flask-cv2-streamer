### Flask based camera streamer

Intended as a within LAN streamer. This uses opencv to grab camera frames and serves them using a cache in flask. The format for serving is motion-jpeg and fps can be set in the code.

To run 

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Code was written using gemini-flash-2.5

### Object Detection (optional)

There is a version `app_yolo.py` that uses `yolov3` to detect objects real-time. Follow instructions below to install the models and run this verrsion.

```
curl -O https://raw.githubusercontent.com/pjreddie/darknet/refs/heads/master/cfg/yolov3.cfg
curl -O https://raw.githubusercontent.com/pjreddie/darknet/refs/heads/master/data/coco.names
curl -O https://data.pjreddie.com/files/yolov3.weights
python app_yolo.py
```