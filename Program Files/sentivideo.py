import cv2
import pyttsx3
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from deep_emotion import Deep_Emotion  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Deep_Emotion()
net.load_state_dict(torch.load('deep_emotion-100-128-0.005.pt'))  
net.to(device)

assist=pyttsx3.init('sapi5')
voices=assist.getProperty('voices')
assist.setProperty('voices',voices[0].id)
assist.setProperty('rate',170)

def speak(audio):
    print(' ')
    assist.say(audio)
    assist.runAndWait()

f_size = 1.5
f = cv2.FONT_HERSHEY_PLAIN
rec_frme_clr = (255, 255, 255)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Webcam Unavailable !!")

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            r_gray = gray[y:y + h, x:x + w]
            r_clr = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_roi = r_clr
            graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            final_img = cv2.resize(graytemp, (48, 48))
            final_img = np.expand_dims(final_img, axis=0)
            final_img = np.expand_dims(final_img, axis=0)
            final_img = final_img / 255.0

            data = torch.from_numpy(final_img).type(torch.FloatTensor).to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            predict = torch.argmax(pred)
            print(predict)

            status = emotions[predict]

            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                captured_frame = frame.copy()  
                cv2.imshow("Captured Frame", captured_frame)
                speak(f"Detected Emotion is {status}")

                gray_captured = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
                resized_captured = cv2.resize(gray_captured, (48, 48))
                final_img_captured = np.expand_dims(resized_captured, axis=0)
                final_img_captured = np.expand_dims(final_img_captured, axis=0)
                final_img_captured = final_img_captured / 255.0

                captured_data = torch.from_numpy(final_img_captured).type(torch.FloatTensor).to(device)
                captured_outputs = net(captured_data)
                captured_pred = F.softmax(captured_outputs, dim=1)
                captured_predict = torch.argmax(captured_pred)
                emotion = emotions[captured_predict]

                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                file_name = f"{status}_{current_time}.jpg"

                save_path = f"E:\\Mini Project\\Program files\\Captured frames\\{file_name}"
                cv2.imwrite(save_path, captured_frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    
    else:
        print("Face Not Found")

    cv2.imshow("Video input", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()