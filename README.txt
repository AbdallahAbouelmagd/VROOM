Overview of this document:

1-System’s Goal
2-Methods & Pipeline
3-Performance
4-Use instructions
----------------------------------------------------------------------------------------------------------------------------------------------------------------

1- System’s Goal

The system was designed to detect change in traffic volume on sections of the highway, using video surveillance cameras installed on highways.

The system is able to consistently detect the traffic volume by comparing the current traffic state to the previous traffic patterns of this scene, which keeps it reliable across various cameras, angles, and environments.

The main challenge in visual traffic surveillance is the resource constrained environment, which is why the system was developed to be extremely lightweight and can be installed on low cost embedded chips or a system-on-chip (SoC).
----------------------------------------------------------------------------------------------------------------------------------------------------------------

2- Methods & Pipeline

When the system is installed on a new surveillance camera, the system first analyses the feed to extract scene specific information such as the shape and sizes of present vehicles, lighting conditions, as well as the traffic patterns in that scene.


All of which are used in the following stage in order to reliably detect current traffic and compare it to the norms of this road section.


After the calibration process, the system continues to process the incoming feed, extract the foreground, and filter noise such as shadows and headlights, using thresholds defined during the automatic calibration phase, based on the lighting conditions in the scene, due to background subtraction’s high sensitivity to illumination changes.

The vehicles in the scene are then detected, and the number of detected vehicles is then accumulated over predefined intervals in order to label the current traffic volume as either in a state of low, normal, or high traffic relative to the norms of this road.
----------------------------------------------------------------------------------------------------------------------------------------------------------------

3-Performance

The system's pipeline was tested using an extensive dataset from highway surveillance cameras in California, Los Angeles, and San Francisco. The dataset covers diversity in camera angles, feed quality, road shape, lighting conditions, and weather. 

The system’s results were compared to the results of a pre-trained YOLO model, which is one of the most popular deep learning models used in traffic surveillance, which is requires extensively more computation resources. 

The system achieved an F1 score of 0.8, showing consistency in detecting change in traffic states, especially in the extreme regions across diverse traffic environments, while requiring much less resources.
----------------------------------------------------------------------------------------------------------------------------------------------------------------

4-Use Instructions

Python 3.13.4 was used to execute the system.

To configure the pipeline, open the classification_pipeline.py script and specify the correct paths for the camera directory and the CSV file containing the metadata.

Install dependencies:

pip install -r requirements.txt

Run the system:

python classification_pipeline
----------------------------------------------------------------------------------------------------------------------------------------------------------------
