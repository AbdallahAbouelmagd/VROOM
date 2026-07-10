1. Dataset

The system was evaluated using a subset of the SLiCE dataset, which is publicly available at the following link:

https://www.nes.uni-due.de/wp-content/data/slice/
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. Data Preprocessing

After downloading the dataset, selected camera videos can be preprocessed by converting them to 1 frame per second (1 FPS). This step can be performed using ffmpeg.

FFmpeg Installation

To install FFmpeg, follow the official installation guide for your operating system:
https://ffmpeg.org/download.html


After installing FFmpeg, navigate to the selected camera directory in a terminal and execute the convert_to_1_fps.ps1 script. The script recursively processes all video files within the directory and converts them to 1 frame per second.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. Evaluation

The evaluate.py script is used to produce the experimental results presented in Chapter 6. It computes the following metrics:

Accuracy
Macro F1-score
Weighted F1-score


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4. Ground Truth Generation

(The CSV files containing the ground truth for the dataset used in the evaluation are provided in the CSV_GT folder)

-To evaluate the system, ground truth annotations are required for computing evaluation metrics, including accuracy, macro F1-score, and weighted F1-score.


The script count_vehicles.py is used for this purpose.

Steps:
Open the script and set the path to the selected camera directory.
Run the script:
python count_vehicles.py
The script will generate a CSV file inside the corresponding camera folder.

The generated CSV file contains the following columns:

Video
Avg_Vehicles_Per_Frame

The evaluation also requires a training–testing data split. This is performed by opening the generated CSV file, adding a data_type column, selecting a single day from the subfolder column that fulfills the requirements described in the thesis, and assigning all rows corresponding to that day to the training set, while assigning the remaining rows to the testing set.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



5. System Setup and Execution

Python 3.13.4 was used to execute the system.

To configure the evaluation, open the evaluate.py script and specify the correct paths for the camera directory and the CSV file.

Install dependencies:

pip install -r requirements.txt

Run the system:

python evaluate.py

This generates a new CSV file containing the predictions for each video, as well as the evaluation metrics for the corresponding camera.

(The CSV files produced during the evaluation are provided in the CSV_predictions folder)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


The index folder contains scripts and the CSV files used to generate additional metrics and graphs presented in the evaluation chapter, based on the predictions produced by the evaluate.py script.


