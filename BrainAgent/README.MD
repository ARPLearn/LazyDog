# WAVEGO Robot Dog AI Controller

An AI-powered controller that gives your WAVEGO robot dog a lifelike personality! Using computer vision and the Qwen-VL language model, it allows your robot to see and react to its environment like a real puppy.

## Prerequisites

* Python 3.10 or higher
* NVIDIA GPU with CUDA support (recommended)
* WAVEGO robot assembled and running
* Robot's IP address on your network

## Installation

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment
py -3.10 -m venv qwen_robot_env

# Activate it
# On Windows:
qwen_robot_env\Scripts\activate
# On Linux/Mac:
source qwen_robot_env/bin/activate
```

### 2. Install Required Dependencies

```bash

pip install numpy
pip install opencv-python
pip install websockets
pip install torch torchvision
pip install accelerate
pip install qwen-vl-utils
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
```

### 3. Download Model Files

1. Create a `models` directory in your project folder:
```bash
mkdir models
```

2. Download the Qwen-VL model files:
   * Visit [Qwen-VL Model Page](https://huggingface.co/Qwen/Qwen-VL)
   * Qwen2-VL-2B-Instruct [Download](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)
   * Download and place the following files in your `models` directory:
     * `config.json`
     * `generation_config.json`
     * `model.safetensors`
     * `tokenizer_config.json`
     * `tokenizer.json`
     * `special_tokens_map.json`

### 4. Project Structure

```
WavegoAgent/
├── models/                    # Model files directory 
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── special_tokens_map.json
├── vision_controller.py       # Main robot control script
└── test.py          # Test file
```

### 5. Robot Network Setup

1. Power on your WAVEGO robot
2. Connect to the robot's WiFi:
   * SSID: `WAVESHARE Robot`
   * Password: `1234567890`
3. Note the robot's IP address (default is `192.168.4.1`)
4. Update `ROBOT_IP` in `vision_controller.py` with your robot's IP

## Running the Controller

1. Ensure your virtual environment is activated:
```bash
# Windows
qwen_robot_env\Scripts\activate

# Linux/Mac
source qwen_robot_env/bin/activate
```

2. Launch the controller:
```bash
python vision_controller.py
```

3. The robot's camera view will appear in a window
4. The robot will begin moving and reacting to its environment
5. Press 'q' in the camera window to stop

## Troubleshooting

### Common Issues

#### Model Loading Errors
* Verify all model files are downloaded correctly
* Check model path matches your directory structure
* Ensure you have enough GPU memory (or try CPU mode)

#### Camera Connection
* Confirm WiFi connection to robot
* Test camera feed: `http://<ROBOT_IP>:5000/video_feed`
* Check if robot's web interface is accessible

#### WebSocket Issues
* Verify robot IP address
* Ensure robot is powered on and responsive
* Check network connection strength

#### CUDA/GPU Problems
* Update NVIDIA drivers
* Verify CUDA toolkit installation
* Check PyTorch CUDA compatibility

### Memory Requirements

* GPU: Minimum 8GB VRAM recommended
* RAM: Minimum 8GB system memory
* Storage: ~5GB for model files

## Safety Guidelines

* Keep operation area clear of obstacles
* Monitor robot during operation
* Avoid elevated surfaces
* Keep robot within sight
* Press 'q' to stop if behavior is unexpected
* Remove any fragile items from the robot's path

## Credits

* WAVEGO Robot by WaveShare
* Qwen-VL Vision Language Model by Alibaba
* OpenCV for computer vision
* PyTorch for AI processing

## License

This project is for educational and personal use. Please refer to WAVEGO and Qwen-VL licenses for commercial use.