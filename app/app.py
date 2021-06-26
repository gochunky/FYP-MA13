import platform
import tkinter as tk

from PIL import Image, ImageTk
from collections import OrderedDict
from tkinter import filedialog


class App:
    MODELS = OrderedDict((
        ("Baseline (Unperturbed) Model", "unperturbed_stats.png"),
        ("Model with Makeup Added", "perturbed_stats.png")
    ))

    __slots__ = (
        "root",
        "model",
        "frame",
        "img",
        "img_name",
        "img_tk",
        "stats_img",
    )

    def __init__(self):
        self.root = tk.Tk()
        self._constant_widgets()
        self._options_menu()
        self._mainframe()
        self._buttons()
    
    def _constant_widgets(self):
        self.root.title("Gender Classifier")

        if platform.system() == "Windows":
            self.root.state("zoomed")
        else:
            self.root.attributes("-zoomed", True)
        tk.Label(self.root, text="Gender Classifier", padx=25, pady=6, font=("Helvetica", 24, "bold")).pack()
    
    def _options_menu(self):
        menu_frame = tk.Frame(self.root)
        menu_frame.pack()
        self.model = tk.StringVar(self.root)
        self.model.set(next(iter(self.MODELS)))
        tk.OptionMenu(menu_frame, self.model, *self.MODELS).pack()
    
    def _mainframe(self):
        self.frame = tk.Frame(self.root, bg="white")
        self.frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    
    def _buttons(self):
        fixed_options = {
            "master": self.root,
            "padx": 35,
            "pady": 10,
            "fg": "white",
            "bg": "grey"
        }
        tk.Button(text="Choose Image", command=self.load, **fixed_options).pack(side=tk.LEFT)
        tk.Button(text="Classify Image", command=self.classify, **fixed_options).pack(side=tk.RIGHT)

    def update_img(self, img_path, basewidth):
        with Image.open(img_path) as img:
            wpercent = basewidth / float(img.size[0])
            hsize = int(float(img.size[1]) * wpercent)
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)

    def load(self):
        for img_display in self.frame.winfo_children():
            img_display.destroy()

        self.img_name = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=(("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*"))
        )
        self.img = self.update_img(self.img_name, 450)

        img_name = tk.Label(self.frame, text=self.img_name.split("/")[-1], font="Helvetica 18 bold", bg="white")
        img_name.place(relx=0.05, rely=0.05)

        self.img_tk = tk.Label(self.frame, image=self.img)
        self.img_tk.place(relx=0.05, rely=0.1)

    def classify(self):
        label = tk.Label(self.frame, text="Prediction: FEMALE", font="Helvetica 18 bold", bg="white")
        label_y = self.img_tk.winfo_y() + self.img_tk.winfo_height() + 50
        label.place(relx=0.05, y=label_y)

        self.stats_img = self.update_img(self.MODELS[self.model.get()], 850)

        stats_tk = tk.Label(self.frame, image=self.stats_img, bg="white")
        stats_tk.place(relx=0.4, rely=0.1)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
