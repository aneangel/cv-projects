# Real-Time Face Detection Project

## Overview

This project implements a real-time face detection system using Python and OpenCV. It captures video from a webcam, processes each frame to detect faces, and displays the results live with bounding boxes around detected faces.

## Current Features

- Real-time video capture from webcam
- Face detection using Haar Cascade Classifier
- Live display of video feed with bounding boxes around detected faces
- Simple user interface (press 'q' to exit)

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Usage

1. Ensure you have the required libraries installed:
   ```
   pip install opencv-python numpy
   ```

2. Download the Haar Cascade XML file for frontal face detection and place it in the `opencvXML` directory.

3. Run the main script:
   ```
   python main.py
   ```

4. A window will open showing the live camera feed with face detection. Press 'q' to exit.

## Project Structure

- `main.py`: Contains the main script with the `real_time_face_detection()` function and program execution.
- `opencvXML/haarcascade_frontalface_default.xml`: Haar Cascade Classifier XML file for face detection.

## Future Improvements

This project is under continuous development. Planned improvements include:

1. **Stable Face Detection During Head Rotation**: 
   Implement more robust face detection algorithms that can maintain tracking as the subject's head rotates.

2. **Improved Bounding Boxes**: 
   Enhance the accuracy and stability of bounding boxes, possibly using more advanced techniques like facial landmarks.

3. **Object Segmentation**: 
   Extend the project to include general object detection and segmentation, allowing for the identification and outlining of multiple object types beyond just faces.

4. **Performance Optimization**: 
   Improve processing speed to handle higher resolution video streams in real-time.

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

MIT License

## Acknowledgments

- OpenCV community for providing robust computer vision tools