\# INA700 AI Vision Pick-and-Place System



Computer vision system developed for the INA700 course project.



The project uses a Basler industrial camera, YOLO11, OpenCV and ArUco calibration to detect physical objects and convert their image positions into real-world coordinates for robotic pick-and-place.



\## System Pipeline



```text

Basler Camera

&nbsp;     |

&nbsp;     v

YOLO11n V3

&nbsp;     |

&nbsp;     v

Shape Detection

&nbsp;     |

&nbsp;     v

OpenCV Contour Extraction

&nbsp;     |

&nbsp;     +----> Centroid

&nbsp;     |

&nbsp;     +----> Orientation

&nbsp;     |

&nbsp;     v

ArUco Calibration

&nbsp;     |

&nbsp;     v

Homography

&nbsp;     |

&nbsp;     v

Pixel Coordinates

&nbsp;     |

&nbsp;     v

Physical Coordinates (mm)

&nbsp;     |

&nbsp;     v

Robot Coordinate Transformation

&nbsp;     |

&nbsp;     v

ABB Robot

```



\## Target Classes



The current YOLO model detects:



\- Square

\- Circle

\- Triangle



Other shapes are treated as background/distractors.



\## Computer Vision



YOLO is used for object detection.



OpenCV is then used inside each YOLO region of interest to determine more precise geometric information.



This includes:



\- object contour

\- centroid

\- object position

\- triangle orientation

\- square orientation



Circle orientation is undefined because a uniform circle is rotationally symmetric.



\## Camera



The system currently uses a Basler industrial camera through the Basler Pylon SDK and the Python `pypylon` library.



\## ArUco Calibration



Calibration currently uses:



```text

Dictionary: DICT\_5X5\_100

Marker ID: 21

Physical marker size: 100 x 100 mm

```



The four detected ArUco corners are mapped to:



```text

Corner 0 -> (0, 0) mm

Corner 1 -> (100, 0) mm

Corner 2 -> (100, 100) mm

Corner 3 -> (0, 100) mm

```



A planar homography is then used to transform image pixel coordinates into physical coordinates relative to the marker.



\## Calibration Validation



Initial physical validation has shown promising results.



One experiment measured approximately:



```text

Physical object separation: 201.00 mm

Vision estimate:            201.36 mm

Difference:                   0.36 mm

```



Additional validation is being performed across the camera workspace using known physical dimensions.



These preliminary measurements should not be interpreted as the final system accuracy until multi-point validation is complete.



\## Main Scripts



\### `live\_basler\_yolo.py`



Live Basler camera object detection using the trained YOLO model.



\### `live\_basler\_geometry.py`



Combines YOLO detection with OpenCV contour processing to calculate object centroid and orientation.



\### `live\_aruco\_test.py`



Tests live detection of the ArUco calibration marker.



\### `calibration\_validation.py`



Validates the pixel-to-millimetre coordinate transformation using the ArUco marker and known physical measurements.



\### `live\_basler\_vision.py`



Experimental integrated vision pipeline combining object detection, geometry, ArUco calibration and physical coordinate conversion.



\### `capture\_images.py`



Captures Basler images for dataset generation and testing.



\### Annotation / Dataset Utilities



Additional scripts are included for annotation, dataset preparation, splitting and verification.



\## Installation



Python 3.11+ is recommended.



Create a virtual environment:



```bash

python -m venv venv

```



Activate it on Windows:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



The Basler Pylon SDK must also be installed for Basler camera access.



\## Model Weights



Trained YOLO weights and datasets are intentionally excluded from the Git repository because of their size.



Place the trained model locally and configure the model path in the relevant script.



\## Current Development Status



Completed:



\- Basler camera acquisition

\- Three-class YOLO object detection

\- Distractor/hard-negative training

\- OpenCV contour extraction

\- Object centroid calculation

\- Triangle orientation

\- Square orientation

\- ArUco ID 21 detection

\- Pixel-to-millimetre homography

\- Initial physical calibration validation



In progress:



\- Multi-point calibration validation

\- Camera-to-robot coordinate transformation

\- Robot communication

\- Automated ABB pick-and-place

\- Final Raspberry Pi deployment

\- GUI and continuous operation



\## Collaboration



Contributions should preferably be developed on separate branches and merged through pull requests.



Example:



```bash

git checkout -b feature/robot-coordinate-transform

```



After making changes:



```bash

git add .

git commit -m "Add robot coordinate transformation"

git push -u origin feature/robot-coordinate-transform

```



A pull request can then be opened for review before merging into `main`.

