# Forest Fire Detection Using FirenetCNN and XAI Techniques
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques)

This project implements a Convolutional Neural Network (CNN) to detect and classify forest fires from images and videos. The model leverages transfer learning with the MobileNetV2 architecture and is trained to distinguish between three classes: 'fire', 'smoke', and 'no_fire'.

To enhance model interpretability and trustworthiness, the project incorporates Explainable AI (XAI) using Grad-CAM (Gradient-weighted Class Activation Mapping). This technique generates heatmaps that visualize the specific regions in an image the model focuses on to make its predictions.

## Key Features
*   **Multi-Class Classification:** Classifies input into 'fire', 'smoke', or 'no_fire' categories.
*   **Transfer Learning:** Utilizes a pre-trained MobileNetV2 model, fine-tuned for the specific task of fire detection, ensuring efficient and effective training.
*   **Data Augmentation:** Employs various image augmentation techniques (rotation, shifting, shearing, zooming, and flipping) to create a more robust and generalized model.
*   **Versatile Prediction:** Capable of performing predictions on static images, pre-recorded videos, and live webcam feeds.
*   **Explainable AI (XAI):** Implements Grad-CAM to produce heatmaps, providing visual insight into the model's decision-making process by highlighting the most influential image features.

## Model Performance
The model was evaluated on a test set of 405 images, achieving an overall accuracy of 82%. The detailed classification report is provided below:

```
--- Classification Report ---
              precision    recall  f1-score   support

        fire       0.92      0.81      0.86       121
     no_fire       0.76      0.98      0.86       146
       smoke       0.84      0.67      0.75       138

    accuracy                           0.82       405
   macro avg       0.84      0.82      0.82       405
weighted avg       0.83      0.82      0.82       405
```
A confusion matrix was also generated to visualize the performance across the different classes, showing the number of true and false predictions for each category.

## Explainable AI with Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) is used to make the model's predictions more transparent. It generates a heatmap over the input image, highlighting the regions the model focused on to make its prediction. This is particularly useful for verifying that the model is correctly identifying features of fire or smoke rather than correlating with irrelevant background elements. The implementation in the notebook overlays this heatmap on the original image for clear visualization, especially for 'fire' and 'smoke' predictions.

## Prerequisites
*   Python 3.8+
*   A webcam for live detection (optional)

## Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques.git
    cd Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques
    ```

2.  **Install the required libraries:**
    The `requirements.txt` file in this repository is empty. You can install the necessary packages using pip:
    ```sh
    pip install tensorflow opencv-python matplotlib seaborn scikit-learn pillow
    ```

3.  **Download the Dataset:**
    This project uses the [Forest Fire Classifier Dataset](https://www.kaggle.com/datasets/google-brain/forest-fire-detection-from-satellite-images). Download the dataset and structure it as follows:
    ```
    .
    ├── data/
    │   └── forestfire-classifier-dataset/
    │       ├── train/
    │       │   ├── fire/
    │       │   ├── nofire/
    │       │   └── smoke/
    │       ├── val/
    │       │   ├── ...
    │       └── test/
    │           ├── ...
    └── Fire_PredCopy.ipynb
    ```

## Usage
All functionalities, from training to prediction, are available in the `Fire_PredCopy.ipynb` Jupyter Notebook.

1.  **Training the Model:**
    Open and run the notebook cells sequentially to perform data augmentation, build the model, and initiate the training process. The trained model will be saved as `FirenetCNN1.h5`.

2.  **Making Predictions:**
    The notebook contains separate sections for different types of inference.

    *   **On a Single Image:**
        Update the `IMAGE_PATH` variable in the corresponding cell to the path of your image and run it. The output will be the predicted class with its confidence score.

    *   **On a Video File:**
        Modify the `VIDEO_PATH` variable to your video file's path. Running the cell will process the video frame-by-frame and display the output with the prediction overlaid.

    *   **With Grad-CAM Visualization:**
        To see the model's focus, run the Grad-CAM cells. Update the `IMAGE_PATH` or `VIDEO_PATH` to generate a prediction with a heatmap highlighting the important regions.

    *   **Live Webcam Inference:**
        Run the final cell in the notebook to start real-time detection using your webcam. Press 'q' to stop the feed.

## File Descriptions
*   `Fire_PredCopy.ipynb`: The main Jupyter Notebook containing all the code for data preprocessing, model training, evaluation, and prediction.
*   `FirenetCNN.h5`, `FirenetCNN1.h5`, `firenet_model.h5`: Pre-trained model files. `FirenetCNN1.h5` is the latest version generated by the notebook.
*   `LICENSE`: The MIT License file for the project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
