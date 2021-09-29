import platform
import tkinter as tk
import imp

from PIL import Image, ImageTk
from collections import OrderedDict
from tkinter import filedialog

# Load in glasses filter module
# @TODO: Replace file paths with something more dynamic
pred_mod = imp.load_source('make_pred', '/home/monash/Desktop/fyp-work/fyp-ma-13/fyp-models/gen_results.py')
temp_fp = '/home/monash/Desktop/fyp-work/fyp-ma-13/app/tmp/'

class App:
    MODELS = OrderedDict((
        ("Baseline (Unperturbed) Model", "unperturbed_stats.png"),
        ("Debiased Model", "perturbed_stats.png")
    ))

    PERTURBS = OrderedDict((
        ("No Filter", "none"),
        ("Glasses Filter", "glasses"),
        ("Make Up Filter", "makeup"),
        ("N95 Mask Filter", "mask")
    ))
    
    MODEL_TYPES = OrderedDict((
        ("ResNet50", "res"),
        ("DenseNet", "dense"),
        ("MobileNet", "mobile")
    ))

    __slots__ = (
        "root",
        "model",
        "perturb",
        "model_type",
        "prediction",
        "confidence",
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
        tk.OptionMenu(menu_frame, self.model, *self.MODELS).pack(side=tk.LEFT)

        self.perturb = tk.StringVar(self.root)
        self.perturb.set(next(iter(self.PERTURBS)))
        tk.OptionMenu(menu_frame, self.perturb, *self.PERTURBS).pack(side=tk.RIGHT)
        
        self.model_type = tk.StringVar(self.root)
        self.model_type.set(next(iter(self.MODEL_TYPES)))
        tk.OptionMenu(menu_frame, self.model_type, *self.MODEL_TYPES).pack(side=tk.RIGHT)
    
    def _mainframe(self):
        self.frame = tk.Frame(self.root, bg="white")
        self.frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
        self.prediction = self.confidence = None
    
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

        # Updates 
        self.img = self.update_img(self.img_name, 250)      

        img_name = tk.Label(self.frame, text=self.img_name.split("/")[-1], font="Helvetica 18 bold", bg="white")
        img_name.place(relx=0.05, rely=0.05)
        self.img_tk = tk.Label(self.frame, image=self.img)
        self.img_tk.place(relx=0.05, rely=0.1)

    def classify(self):
        # Get image with filter on first
        # @TODO: Fix overlapping filters
        if self.perturb.get() != "No Filter":
            filter_type = self.PERTURBS[self.perturb.get()]
            target_fp = temp_fp + filter_type + "/" 
            if filter_type == 'glasses':
                pred_mod.apply_filter(self.img_name, target_fp, "glasses")
            elif filter_type == 'makeup':
                pred_mod.apply_filter(self.img_name, target_fp, "makeup")
            self.img_name = target_fp + self.img_name.split("/")[-1]      # Update 

        print("Chosen model:", self.model_type.get())
        res = pred_mod.make_pred(self.img_name, self.MODEL_TYPES[self.model_type.get()])  
        # @TODO: Fix overlapping text issue

        # label = tk.Label(self.frame, text="Prediction: {}".format(res[0]), font="Helvetica 18 bold", bg="white")
        if self.prediction is not None and self.confidence is not None:
            self.prediction.destroy()
            self.confidence.destroy()
        self.prediction = tk.Label(self.frame, text="Prediction: {}".format(res[0]), font="Helvetica 18 bold", bg="white")
        self.confidence = tk.Label(self.frame, text="Confidence: {}".format(res[1][0][0].round(2)), font="Helvetica 18 bold", bg="white")
        

        label_y = self.img_tk.winfo_y() + self.img_tk.winfo_height() + 50

        self.prediction.place(relx=0.05, y=label_y)
        self.confidence.place(relx=0.05, y=label_y+100)


        # Reload image
        if self.perturb.get() != "No Filter":
            self.img = self.update_img(self.img_name, 250)  
            self.img_tk = tk.Label(self.frame, image=self.img)
            self.img_tk.place(relx=0.05, rely=0.1)

        self.stats_img = self.update_img(self.MODELS[self.model.get()], 850)
        stats_tk = tk.Label(self.frame, image=self.stats_img, bg="white")
        stats_tk.place(relx=0.4, rely=0.1)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
