import tkinter as tk
import recorder
import main
import tracker

WIDTH = 0
HEIGHT = 0

class Window:

    # Main window
    def __init__(self, root, resolution):
        global WIDTH, HEIGHT
        WIDTH = resolution[0]
        HEIGHT = resolution[1]

        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.disableExit)
        self.root.title("Main Window")
        self.frame = tk.Frame(self.root)
        self.calibration_button = tk.Button(self.frame, text = 'Calibration', width = 25, command = self.newCalibration)
        self.tracking_button = tk.Button(self.frame, text = 'Enable tracking', width = 25, command = self.toggleTracking)
        self.close_button = tk.Button(self.frame, text = 'Close', width = 25, command = self.close)

        self.calibration_button.pack()
        self.tracking_button.pack()
        self.close_button.pack()
        self.frame.pack()

        self.face_tracker = tracker.Tracker(self)
        self.tracking_enabled = False


    # Open calibration window
    def newCalibration(self):
        self.calibration_button.config(state = "disabled")
        self.tracking_button.config(state = "disabled")

        self.new_window = tk.Toplevel(self.root)
        self.app = CalibrateWindow(self, self.new_window, self.face_tracker)


    # Enable/Disable eye tracking
    def toggleTracking(self):
        self.tracking_enabled = not self.tracking_enabled
        if self.tracking_enabled:
            self.calibration_button.config(state = "disabled")
            self.tracking_button.config(text = "Disable tracking (close eyes for 1s)")

            while self.tracking_enabled:
                self.root.update_idletasks()
                self.root.update()
                self.face_tracker.predict(False)
        else:
            self.calibration_button.config(state = "normal")
            self.tracking_button.config(text = "Enable tracking")


    def close(self):
        self.tracking_enabled = False
        self.root.destroy()

    def disableExit(e):
        pass




class CalibrateWindow:
    def __init__(self, mainWindow, master, face_tracker):
        global WIDTH, HEIGHT
        self.main_window = mainWindow
        self.face_tracker = face_tracker
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.disableExit)
        self.master.title("Calibration")

        self.frame = tk.Frame(self.master)
        self.frame.pack()

        self.label = tk.Label(self.frame, text="Click each of the circles once while looking directly at it.\nGrey = Not yet clicked. Red = Hovered before but didn't clicked. Yellow = Hovering on. Green = Already clicked on.")
        self.label.pack()

        self.quit_button = tk.Button(
                            self.frame,
                            text = 'Quit',
                            width = 25,
                            command = self.close_windows)
        self.quit_button.pack()

        self.canvas = tk.Canvas(self.frame, bg='white', width=WIDTH, height=HEIGHT)
        self.canvas.pack()
        self.master.update_idletasks()
        self.master.update()


        # Create the calibration buttons in a 3x3 arrangement
        self.num_rows_cols = 3
        self.size = 25
        self.offset = 50

        self.canvas_width = self.canvas.winfo_width() - self.offset * 3
        self.canvas_height = self.canvas.winfo_height() - self.offset * 3

        self.clicked_buttons = {}
        for x in range(self.num_rows_cols):
            for y in range(self.num_rows_cols):
                self.x_coord = x * self.canvas_width / (self.num_rows_cols-1) + self.offset
                self.y_coord = y * self.canvas_height / (self.num_rows_cols-1) + self.offset
                self.button = self.canvas.create_oval(self.x_coord, self.y_coord, self.x_coord + self.size, self.y_coord + self.size,  fill="gray")

                self.canvas.tag_bind(self.button, '<Enter>', self.onEnter)
                self.canvas.tag_bind(self.button, '<Leave>', self.onLeave)
                self.canvas.tag_bind(self.button, '<Button-1>', self.onClick)

                self.clicked_buttons[self.button] = False


        self.master.geometry("+0+0")
        self.master.resizable(False, False)


    def disableExit(e):
        pass


    # Turn the button currently hovering on YELLOW
    def onEnter(self, event):
        self.button = self.canvas.find_closest(event.x, event.y)
        if self.clicked_buttons[self.button[0]]:
            return
        self.canvas.itemconfig(self.button, fill="yellow")

    # Turn the button just left RED
    def onLeave(self, event):
        self.button = self.canvas.find_closest(event.x, event.y)
        if self.clicked_buttons[self.button[0]]:
            return
        self.canvas.itemconfig(self.button, fill="red")


    # Turn the button just clicked on GREEN and record entries
    def onClick(self, event):
        self.button = self.canvas.find_closest(event.x, event.y)
        if self.clicked_buttons[self.button[0]]:
            return

        # Record calibration entries
        self.face_tracker.recordCalibration()
        self.canvas.itemconfig(self.button, fill="green")
        self.clicked_buttons[self.button[0]] = True


        # Only calibrate once all buttons are clicked
        if False in self.clicked_buttons.values():
            return

        self.master.resizable(True, True)
        self.label.config(text = "Calibrating...")
        self.quit_button.config(state = "disabled")
        self.canvas.destroy()
        self.master.wm_geometry("")
        self.master.update_idletasks()
        self.master.update()

        # Calibrate models
        self.face_tracker.beginCalibration()

        self.label.config(text = "Calibration complete.")
        self.quit_button.config(state = "normal")



    def close_windows(self):
        self.main_window.calibration_button.config(state = "normal")
        self.main_window.tracking_button.config(state = "normal")
        self.master.destroy()
