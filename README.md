# Hand Gesture Recognition

This repository contains two main Python scripts for hand gesture recognition using the Ultraleap controller:

1. `data_collect.py`: Collects hand gesture data using the Ultraleap controller.
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
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running the scripts, make sure you have the following:

- Ultraleap controller
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

1. Connect the Ultraleap controller to your computer.
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

1. Connect the Ultraleap controller to your computer.
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
`-td, --top_dirs` (int): Top number of directions to consider when determining the moving direction over the sliding window. (Default: 3)
`-ws, --window_size` (int): Size of the sliding window for calculating the moving direction. (Default: 16)
`-str, --stationary_threshold_ratio` (float): Stationary threshold to determine if the hand is moving or stationary. (Default: 1.5)
`-sl, --similarity_lookback` (int): How many frames to look back to when calculating the similarity between the current frame and the chosen frame. (Default: 2)
`-st, --similarity_threshold` (float): Similarity threshold to determine if the current frame is similar to the chosen frame. (Default: 0.9)
`-seq, --sequence_length` (int): The length of the sequence to decide the Transformer model. (Default: 32)
`-ow, --output_window` (int): The output window to decide the Transformer model. (Default: 1)
`-tl, --target_length` (int): Target length for sequence normalization. (Default: 32)
`-mf, --max_frames` (int): Maximum number of frames to store in the Frames object. (Default: 100)
`-ts, --test_size` (float): Test size for the lookup table. (Default: 0.4)
`-ct, --confidence_threshold` (float): Gesture classification confidence threshold; if the confidence is below this threshold, the gesture will be classified as Unknown. (Default: 0.01)
`-d, --device` (str): Device to run the model on (requires a CUDA-enabled device for GPU). (Default: 'cpu')
`-T, --classification_timeout` (int): The time interval for gesture classification. (Default: 2000)
Example usage:

```
python demo.py -td 2 -ws 20 -ct 0.05 -d cuda
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
