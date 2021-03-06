#!/usr/bin/env python
# coding: utf-8

"""
This script contains the full implementation of our classifier to demonstrate
the effects of perturbations on gender classifiers.
"""

import imp
import platform
import tkinter as tk

from PIL import Image, ImageTk
from collections import OrderedDict
from tkinter import filedialog

# Load glasses filter module
pred_mod = imp.load_source("make_pred", "../fyp-models/gen_results.py")
temp_fp = "tmp/"


class App:
    """
    This is the class for our application, whicch is used to store the state of
    the application.
    """

    # Application constants
    MODELS = OrderedDict(
        (("Baseline (Unperturbed) Model", False), ("Debiased Model", True))
    )

    PERTURBS = OrderedDict(
        (
            ("No Filter", "none"),
            ("Glasses Filter", "glasses"),
            ("Make Up Filter", "makeup"),
            ("N95 Mask Filter", "mask"),
        )
    )

    MODEL_TYPES = OrderedDict(
        (("ResNet50", "res"), ("DenseNet", "dense"), ("MobileNet", "mobile"))
    )

    # Variable slots, specified for performance optimisation
    __slots__ = (
        "root",
        "model",
        "perturb",
        "model_type",
        "prediction",
        "confidence",
        "original_fp",
        "frame",
        "img",
        "img_name",
        "img_tk",
        "stats_img",
    )

    def __init__(self):
        """
        Initialises the UI elements
        """
        self.root = tk.Tk()
        self._constant_widgets()
        self._options_menu()
        self._mainframe()
        self._buttons()

    def _constant_widgets(self):
        """
        Initialises the widgets
        """
        self.root.title("Gender Classifier")

        if platform.system() == "Windows":
            self.root.state("zoomed")
        else:
            self.root.attributes("-zoomed", True)

        tk.Label(
            self.root,
            text="Gender Classifier",
            padx=25,
            pady=6,
            font=("Helvetica", 24, "bold"),
        ).pack()

    def _options_menu(self):
        """
        Initialises the options menu
        """
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
        tk.OptionMenu(menu_frame, self.model_type, *self.MODEL_TYPES).pack(
            side=tk.RIGHT
        )

    def _mainframe(self):
        """
        Initialises the window frame
        """
        self.frame = tk.Frame(self.root, bg="white")
        self.frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
        self.prediction = self.confidence = None

    def _buttons(self):
        """
        Initialises the buttons
        """
        fixed_options = {
            "master": self.root,
            "padx": 35,
            "pady": 10,
            "fg": "white",
            "bg": "grey",
        }
        tk.Button(text="Choose Image", command=self.load, **fixed_options).pack(
            side=tk.LEFT
        )
        tk.Button(text="Classify Image", command=self.classify, **fixed_options).pack(
            side=tk.RIGHT
        )

    def load(self):
        """
        Loads a new image to the application
        """
        # Remove the current image
        for img_display in self.frame.winfo_children():
            img_display.destroy()

        # Select image
        self.img_name = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=(("Images", "*.png *.jpg "), ("All files", "*.*")),
        )

        # Update image
        self.img = self.update_img(self.img_name, 250)

        img_name = tk.Label(
            self.frame,
            text=self.img_name.split("/")[-1],
            font="Helvetica 18 bold",
            bg="white",
        )
        img_name.place(relx=0.05, rely=0.05)

        self.img_tk = tk.Label(self.frame, image=self.img)
        self.img_tk.place(relx=0.05, rely=0.1)
        self.original_fp = self.img_name

    def classify(self):
        """
        Classifies the current image in the application.
        """
        # Get the image with no filter
        if self.perturb.get() != "No Filter":
            filter_type = self.PERTURBS[self.perturb.get()]
            target_fp = temp_fp + filter_type + "/"

            self.img_name = target_fp + self.original_fp.split("/")[-1]

            # Apply filter to original image
            if (
                filter_type == "glasses"
                or filter_type == "makeup"
                or filter_type == "mask"
            ):
                pred_mod.apply_filter(self.original_fp, target_fp, filter_type)

        else:
            self.img_name = self.original_fp

        # Parameter customisation
        print("Chosen model:", self.model_type.get())
        isDebiased = self.MODELS[self.model.get()]
        modelType = self.MODEL_TYPES[self.model_type.get()]
        res = pred_mod.make_pred(self.img_name, modelType, debiased=isDebiased)

        # Update prediction and confidence labels
        if self.prediction is not None and self.confidence is not None:
            # Reset labels
            self.prediction.destroy()
            self.confidence.destroy()
        self.prediction = tk.Label(
            self.frame,
            text="Prediction: {}".format(res[0]),
            font="Helvetica 18 bold",
            bg="white",
        )
        self.confidence = tk.Label(
            self.frame,
            text="Confidence: {}".format(str(round(float(res[1][0][0]), 2))),
            font="Helvetica 18 bold",
            bg="white",
        )

        # Write prediction and confidence values
        label_y = self.img_tk.winfo_y() + self.img_tk.winfo_height() + 50
        self.prediction.place(relx=0.05, y=label_y)
        self.confidence.place(relx=0.05, y=label_y + 100)

        # Reload image
        self.img = self.update_img(self.img_name, 250)
        self.img_tk = tk.Label(self.frame, image=self.img)
        self.img_tk.place(relx=0.05, rely=0.1)

        stats_fp = "stats_diagrams/{}{}_stats_graph.png".format(
            modelType, "_debiased" if isDebiased else ""
        )
        self.stats_img = self.update_img(stats_fp, 850)
        stats_tk = tk.Label(self.frame, image=self.stats_img, bg="white")
        stats_tk.place(relx=0.4, rely=0.1)
    
    def update_img(self, img_path, basewidth):
        """
        Processes and returns a given image to be displayed in the application.

        img_path : str
            Path to the image to be displayed
        basewidth : float
            Width to scale the given image to
        """
        with Image.open(img_path) as img:
            wpercent = basewidth / float(img.size[0])
            hsize = int(float(img.size[1]) * wpercent)
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)

    def run(self):
        """
        Runs the entire program
        """
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
