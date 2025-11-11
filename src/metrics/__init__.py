import pyautogui
import time

interval = 5

def keep_awake():
    while True:
        print("keeping awake")
        pyautogui.move(10, 10)
        pyautogui.scroll(10)
        time.sleep(0.5)
        pyautogui.scroll(-10)
        pyautogui.move(-10, -10)
        time.sleep(interval)

if __name__ == "__main__":
    time.sleep(1)
    keep_awake()
