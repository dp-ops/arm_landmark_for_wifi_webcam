import cv2
import requests
import numpy as np

stream_url = 'http://192.168.1.3/stream' 

def watch_stream():
    """
    Connects to the ESP32 MJPEG stream, decodes each frame, 
    and displays it. Press 'q' to quit.
    This is where you would add your computer vision logic.
    """

    try:
        r = requests.get(stream_url, stream=True, timeout=10)
        r.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to stream: {e}")
        return

    print("Connected to stream. Press 'q' in the window to quit.")
    
    bytes_data = bytes()
    
    for chunk in r.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8') 
        b = bytes_data.find(b'\xff\xd9') 
        
        if a != -1 and b != -1:
            jpg_raw = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            img = cv2.imdecode(np.frombuffer(jpg_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is not None:
                # =========================================================
           
                # =========================================================

                cv2.imshow('ESP32-CAM Stream', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting stream.")
                    break
    

    cv2.destroyAllWindows()
    r.close()

if __name__ == '__main__':
    watch_stream()