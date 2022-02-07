import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy import signal

frame_cnt = 0
event_cnt = 0
PIX_ON_THRESH = 20
PIX_OFF_THRESH = -20

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 40 # Interval in ms to get the latest frame
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)
        self.canvas.pack()
        self.v = StringVar()
        frame = Frame(root)
        frame.pack()

        # Dictionary to create multiple buttons
        self.values = {"Original": "O",
                  "Grayscale": "G",
                  "Temporal Contrast": "T",
                  "Spatial Contrast": "S",
                  "DxDt": "Dx",
                  "DyDt": "Dy",
                  "DxDy": "Dxy",
                  "Pause": "P"}

        # Loop is used to create multiple Radiobuttons
        # rather than creating each button separately
        for (text, value) in self.values.items():
            Radiobutton(frame, text=text, variable=self.v,
                        value=value, command = self.update_image).pack(side=TOP, ipadx=5)
        # Update image on canvas
        # self.update_image()

    def update_image(self):
        # Get the latest frame and convert image format
        imgType = self.v.get()
        global frame_cnt
        global prev_frame
        global events
        # print(imgType)
        if imgType == "O":
            self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        elif imgType == "G":
            self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)  # to Grayscale
        elif imgType == "T":
            gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
            if frame_cnt == 0:
                prev_frame = gray
                events = np.zeros(np.shape(gray))
            diff_frame = gray.astype(np.int) - prev_frame
            events[...] = 128
            events[diff_frame >= PIX_ON_THRESH] = 255
            events[diff_frame <= PIX_OFF_THRESH] = 0
            self.image = events
            prev_frame = gray
            frame_cnt += 1
        elif imgType == "S":
            gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
            # global frame_cnt
            # global prev_frame
            # global events
            if frame_cnt == 0:
                # prev_frame = gray
                events = np.zeros(np.shape(gray))
            gray_int = gray.astype(np.int)
            spat_diff = (9 / 8) * gray_int - signal.convolve2d(gray_int, np.ones((3, 3)), mode='same') / 8
            events[...] = 128
            events[spat_diff >= PIX_ON_THRESH/10] = 255
            events[spat_diff <= PIX_OFF_THRESH/10] = 0
            self.image = events
            # prev_frame = gray
            frame_cnt += 1
        elif imgType == "Dy":
            gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
            if frame_cnt == 0:
                prev_frame = gray
                events = np.zeros(np.shape(gray))
            diff_frame = gray.astype(np.int) - prev_frame
            dydt_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # fAveraged = 1 / 25 * np.ones(5, 5)
            #
            # lastImage = squeeze(mean(snapshot(camera), 3));
            # temp1 = conv2(lastImage, fAveraged);
            # temp2 = conv2(currentImage, fAveraged);
            #
            # diffImage = temp1 - temp2;

            DyDtImage = signal.convolve2d(diff_frame, dydt_kernel, mode='same')
            # print(np.max(self.image),np.min(self.image))
            events[...] = 128
            events[DyDtImage >= PIX_ON_THRESH] = 255
            events[DyDtImage <= PIX_OFF_THRESH] = 0
            self.image = events
            prev_frame = gray
            frame_cnt += 1
            # DxDtImage = temp(1:W, 1: H)
        elif imgType == "Dx":
            gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
            if frame_cnt == 0:
                prev_frame = gray
                events = np.zeros(np.shape(gray))
            diff_frame = gray.astype(np.int) - prev_frame
            dxdt_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            DxDtImage = signal.convolve2d(diff_frame, dxdt_kernel, mode='same')
            # print(np.max(self.image),np.min(self.image))
            events[...] = 128
            events[DxDtImage >= PIX_ON_THRESH] = 255
            events[DxDtImage <= PIX_OFF_THRESH] = 0
            self.image = events
            prev_frame = gray
            frame_cnt += 1
        elif imgType == "Dxy":
            gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
            if frame_cnt == 0:
                prev_frame = gray
                events = np.zeros(np.shape(gray))
            diff_frame = gray.astype(np.int) - prev_frame
            dxdy_kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

            DxDyImage = signal.convolve2d(diff_frame, dxdy_kernel, mode='same')
            # print(np.max(self.image),np.min(self.image))
            events[...] = 128
            events[DxDyImage >= PIX_ON_THRESH] = 255
            events[DxDyImage <= PIX_OFF_THRESH] = 0
            self.image = events
            prev_frame = gray
            frame_cnt += 1
        else:
            self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # to RGB

        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        if imgType != "P":
            self.window.after(self.interval, self.update_image)

if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()
