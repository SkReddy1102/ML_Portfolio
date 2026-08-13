# Farm Insect YOLOv8

This project contains a YOLOv8-based farm insect classification application and a separate 3-class YOLO detection dataset experiment. The deployed Streamlit application uses a custom 15-class classification model (`insects.pt`) and returns the top predicted insect class from uploaded images.

## Project Structure

```text
farm-insect-yolov8/
├── classifier_app/
│   ├── app.py
│   ├── settings.py
│   ├── images/
│   └── weights/
│       └── insects.pt
├── classifier_training/
│   └── train_yolov8_classifier.ipynb
└── detection_dataset_3class/
    ├── data.yaml
    ├── README.dataset.txt
    └── README.roboflow.txt
```

## Streamlit Classifier App

The main application is in `classifier_app/`. The Streamlit app accepts an insect image, runs inference using the custom YOLOv8 classification model, and displays the predicted insect class, confidence score, and top-5 predictions.

Run the app:

```bash
pip install -r requirements.txt
streamlit run classifier_app/app.py
```

## Classifier Training

The training notebook is included in `classifier_training/train_yolov8_classifier.ipynb`. It documents the YOLOv8 classification training workflow used for the 15-class classifier.

The Roboflow API key has been removed from the notebook. Users should provide their own API key locally and never commit API credentials to the repository.

## Separate 3-Class Detection Dataset

The repository also includes a Roboflow YOLO-format dataset metadata package containing Armyworms, Brown Marmorated Stink Bugs, and tomato hornworms.

This dataset is separate from the 15-class classifier used by the Streamlit application and is included only as an additional detection experiment.

## Notes

Generated training outputs such as `runs/`, cache files, and Python bytecode are intentionally excluded from the portfolio repo to keep it readable.
