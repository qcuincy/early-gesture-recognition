# Hand Gesture Recognition

This repository contains two main Python scripts for hand gesture recognition using the Leap Motion controller:

1. `data_collect.py`: Collects hand gesture data using the Leap Motion controller.
2. `demo.py`: Demonstrates real-time hand gesture recognition using the collected data and a trained model.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Gesture Recognition Demo](#gesture-recognition-demo)
- [Customization](#customization)
  - [Data Collection Script](#data-collection-script)
  - [Demo Script](#demo-script)
- [Walkthrough Notebook](#walkthrough-notebook)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running the scripts, make sure you have the following:

- Leap Motion controller
- Python 3.x installed
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository or download the script files.
2. Install the required Python packages by running the following command:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Collection

1. Connect the Leap Motion controller to your computer.
2. Open a terminal or command prompt and navigate to the directory where the `data_collect.py` script is located.
3. Run the script using the following command:

   ```
   python data_collect.py
   ```

4. The script will open a graphical interface showing the live hand tracking data.
5. Use the following keys to interact with the script:
   - `q`: Exit the script.
   - `1`: Select "Swipe Right" gesture to record.
   - `2`: Select "Swipe Left" gesture to record.
   - `3`: Select "Swipe Up" gesture to record.
   - `4`: Select "Swipe Down" gesture to record.
   - `r`: Start recording the selected gesture.
   - `s`: Stop recording the gesture.
6. The recorded gesture data will be saved in the `hand_gesture_data` directory.

### Gesture Recognition Demo

1. Connect the Leap Motion controller to your computer.
2. Open a terminal or command prompt and navigate to the directory where the `demo.py` script is located.
3. Run the script using the following command:

   ```
   python demo.py
   ```

4. The script will open a graphical interface showing the live hand tracking data and recognised gesture.
5. Use the following keys to interact with the script:
   - `h`: Head-mounted display tracking mode.
   - `s`: Screen Top tracking mode.
   - `d`: Desktop tracking mode.
   - `q`: Exit the script.

## Customization

### Data Collection Script

You can customize the `data_collect.py` script's parameters by using command-line arguments. Here are the available options:

- `-dp, --data_path`: Default path to store gesture sequences. (Default: current working directory)
- `-td, --top_dirs`: Top number of directions to consider when determining the moving direction over the sliding window. (Default: 3)
- `-ws, --window_size`: Size of the sliding window for calculating the moving direction. (Default: 16)
- `-str, --stationary_threshold_ratio`: Stationary threshold to determine if the hand is moving or stationary. (Default: 1.5)
- `-sl, --similarity_lookback`: How many frames to look back to when calculating the similarity between the current frame and the chosen frame. (Default: 2)
- `-st, --similarity_threshold`: Similarity threshold to determine if the current frame is similar to the chosen frame. (Default: 0.9)
- `-seq, --sequence_length`: The length of the sequence to decide the Transformer model. (Default: 32)
- `-mf, --max_frames`: Maximum number of frames to store in Frames object. (Default: 100)

Example usage:

```
python data_collect.py -dp "C:\path\to\data_folder" -td 3 -ws 16 -str 1.5 -sl 2 -st 0.9 -seq 32 -mf 100
```

### Demo Script

You can customize the `demo.py` script's parameters by using command-line arguments. Here are the available options:

- `-td, --top_dirs`: Top number of directions to consider when determining the moving direction over the sliding window. (Default: 3)
- `-ws, --window_size`: Size of the sliding window for calculating the moving direction. (Default: 16)
- `-str, --stationary_threshold_ratio`: Stationary threshold to determine if the hand is moving or stationary. (Default: 1.5)
- `-sl, --similarity_lookback`: How many frames to look back to when calculating the similarity between the current frame and the chosen frame. (Default: 2)
- `-st, --similarity_threshold`: Similarity threshold to determine if the current frame is similar to the chosen frame. (Default: 0.9)
- `-seq, --sequence_length`: The length of the sequence to decide the Transformer model. (Default: 32)
- `-mf, --max_frames`: Maximum number of frames to store in Frames object. (Default: 100)

Example usage:

```
python demo.py -td 3 -ws 16 -str 1.5 -sl 2 -st 0.9 -seq 32 -mf 100
```

## Walkthrough Notebook

The `walkthrough.ipynb` Jupyter Notebook provides a detailed guide on:

1.  **Extracting Features:** How to derive discrete features (palm orientation, moving direction, hand pose) from the raw 3D hand landmark coordinates.
2.  **Feature Engineering:** Explanation and code examples for calculating and visualizing these features.
3.  **Gesture Classification (Optional):** A demonstration of how to use the extracted features to classify dynamic hand gestures using a lookup table and Transformer model.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
