import cv2
import requests
import threading
import time
import argparse

latest_frame = None


def resize_frame(frame, max_size=640):
    h, w = frame.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def capture_and_send(server_url, frame_interval):
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Cannot capture image")
                break

            frame_resized = resize_frame(frame, max_size=640)
            latest_frame = frame_resized  # Update shared frame

            # Upload once per second
            _, img_encoded = cv2.imencode('.jpg', frame_resized)
            files = {'image': ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            try:
                resp = requests.post(f"{server_url}/add_image", files=files, timeout=5)
                if resp.status_code == 200 and resp.json().get("response", False):
                    res = resp.json()
                    print(f"[{res['time']}s] {res['content']}")
            except Exception as e:
                print("Failed to send image:", e)

            time.sleep(frame_interval)
    finally:
        cap.release()


def send_text(server_url):
    while True:
        text = input("Please enter text (enter RESET to reset): ").strip()
        try:
            if text.upper() == "RESET":
                resp = requests.post(f"{server_url}/reset", timeout=5)
                print("RESET result:", resp.json())
            else:
                resp = requests.post(f"{server_url}/add_text", json={"text": text}, timeout=5)
                print("Text sending result:", resp.json())
        except Exception as e:
            print("Failed to send text:", e)


def main():
    parser = argparse.ArgumentParser(description="Frontend client for video streaming and text input")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--frame_interval", type=float, default=2, help="Time interval between frames (default: 2 seconds)")
    args = parser.parse_args()

    server_url = args.server_url
    print(f"Using server URL: {server_url}")

    # Start thread for capturing and sending video
    t1 = threading.Thread(target=capture_and_send, args=(server_url, args.frame_interval), daemon=True)
    t1.start()

    # Start thread for processing text input
    t2 = threading.Thread(target=send_text, args=(server_url,), daemon=True)
    t2.start()

    # Main thread is responsible for displaying the window
    while True:
        if latest_frame is not None:
            cv2.imshow("Camera", latest_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera window")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
