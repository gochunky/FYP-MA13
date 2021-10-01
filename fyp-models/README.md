# Key Components

## poc_tl_all_cv.ipynb

This notebook is focused on model creation, model training and model evaluation.

### Outline

- Import libraries and initialise global variables
- Load data
- Data augmentation
- Load base models
- Model creation using transfer learning
    - Base models (from step 4) are used here
- Model training
- Build models for all sets
    - Call gen_models_all_sets()
- Model Analysis
    - Get model statistics
- Findings and results

## gen_results.py

This python file stores useful functions that allows us to perform useful actions. 

Such actions include:
- Adding a desired filter to a specific image
- Making a single prediction with a confidence score for a specific image

| Function | Description |
| --- | ----------- |
| ```gen_metrics``` | Generates classification report and confusion matrix (sklearn.metrics). Used by ```gen_save_cr_cm``` function.
| ```gen_save_cr_cm``` | Generates, saves and returns classification reports and confusion matrix. Used in ```poc_tl_all_cv.ipynb```.|
| ```apply_filter``` | Applies a desired filter to specific image. Used by *GUI* to apply filter to specific image and store in a target file path.|
| ```make_pred``` | Returns predicted class and confidence for a single image. Used by *GUI* to get prediction for a specified image.|
