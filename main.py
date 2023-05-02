import tkinter   as tk
import window

def main():
    root = tk.Tk()
    resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
    mainWindow = window.Window(root, resolution)
    root.mainloop()

if __name__ == "__main__":
    main()
