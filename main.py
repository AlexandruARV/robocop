import cv2
import torch
import numpy as np
import socket
import threading


client_data = None
data_lock = threading.Lock()


server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('127.0.0.1', 5000))  
client_address = ('127.0.0.1', 12345)    

print(torch.__version__)  
print(torch.version.cuda)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Folosesc dispozitivul: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_skip = 2 
frame_count = 0

if not cap.isOpened():
    print("Eroare la deschiderea stream-ului video.")
    exit()


def receive_from_client():
    global client_data
    while True:
        data, address = server_socket.recvfrom(1024)
        print(f"Primit de la {address}: {data.decode('utf-8')}")
        
        with data_lock:
            client_data = data.decode('utf-8')


def detect_and_send():
    global frame_count, client_data
    while True:
        frame_count += 1

        if frame_count % frame_skip != 0:
            ret, _ = cap.read()  
            continue

        ret, frame = cap.read()  
        if not ret:
            print("Nu s-a mai putut citi frame-ul.")
            break

        height, width, _ = frame.shape
        center_x_screen = width / 2  

        results = model(frame)

        detections = results.pandas().xyxy[0]  
        bottle_detections = detections[detections['name'] == 'bottle']  

        if not bottle_detections.empty:
            max_area = 0
            max_bottle = None

            for index, row in bottle_detections.iterrows():
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])
                area = (xmax - xmin) * (ymax - ymin)  

                if area > max_area:
                    max_area = area
                    max_bottle = row  

            if max_bottle is not None:
                xmin = int(max_bottle['xmin'])
                ymin = int(max_bottle['ymin'])
                xmax = int(max_bottle['xmax'])
                ymax = int(max_bottle['ymax'])
                confidence = max_bottle['confidence']
                label = max_bottle['name']
                
                rectangleBottleLength = xmax - xmin

                center_x_bottle = (xmin + xmax) / 2
                if center_x_bottle < center_x_screen - rectangleBottleLength / 2:
                    position = "Left"
                elif center_x_bottle > center_x_screen + rectangleBottleLength / 2:
                    position = "Right"
                else:
                    position = "Center"

                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f} ({position})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Obiect detectat: {label}, Încredere: {confidence:.2f}")
                print(f"Suprafață chenar: {max_area} pixeli pătrați")
                print(f"Coordonate cutie delimitare: (xmin: {xmin}, ymin: {ymin}), (xmax: {xmax}, ymax: {ymax})")

                with data_lock:
                    if client_data == "True":
                        message = f"{position}, xmin: {xmin},xmax: {xmax}, rectangleBottleLength : {rectangleBottleLength}"
                        server_socket.sendto(message.encode('utf-8'), client_address)
                        print(f"Mesaj trimis către client {client_address}: {message}")
                        print(f"Mesaj de la client utilizat: {client_data}")
                
        
        cv2.imshow("Detectii in Timp Real - Doar Sticla cu Suprafata Maxima", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


receive_thread = threading.Thread(target=receive_from_client)
send_thread = threading.Thread(target=detect_and_send)

receive_thread.start()
send_thread.start()


receive_thread.join()
send_thread.join()
