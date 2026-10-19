# Self-Driving Car Agent

This is my coursework project for the **University of Brighton RO501 Intelligent Agents** module. Students were given free range to pick their own project, and I chose to build a self-driving car agent for the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim).<br />
The idea is simple in concept: train a Convolutional Neural Network (CNN) to look at the camera feed coming out of the simulator and predict the **steering angle** needed to keep the car on the track. For speed I've added a small **PID controller** so the car can hold a target throttle on its own. Once the model is trained, the whole thing runs as a Flask-SocketIO server that the simulator talks to over a socket — you just launch the sim and watch the agent drive.

I collected the training data myself by driving around the simulator with the mouse, which produces a raw `driving_log.csv`. The compiled dataset (as `.npy` files) is hosted on [Hugging Face](https://huggingface.co/datasets/JakeWatson/UdacityCarSimDataset/tree/main) so you don't have to drive it all yourself if you just want to train and test. As a rule of thumb try to collect at least **~50k frames** for good results.

| Center | Left | Right |
|:---:|:---:|:---:|
| ![](https://github.com/Jake1402/University-of-Brighton-RO501-Coursework/blob/main/media/Test_Center.jpg) | ![](https://github.com/Jake1402/University-of-Brighton-RO501-Coursework/blob/main/media/Test_Left.jpg) | ![](https://github.com/Jake1402/University-of-Brighton-RO501-Coursework/blob/main/media/Test_Right.jpg) |

The three cameras are the centre, left and right views. During training the left and right frames are fed in with a constant offset added to their steering label, which effectively acts as data augmentation and teaches the model to correct back towards the middle of the road.

## How it works

The project is broken down into a small pipeline of scripts that each do one job:

1. **`DatasetBuilder.py`** — reads the raw `driving_log.csv`, smooths the steering with a moving average, crops/resizes/converts every frame and saves it out as `.npy` arrays (`center`, `left`, `right`, `steer`).
2. **`LoadTrainingSet.py`** — loads those `.npy` files, balances the dataset by steering frequency, shuffles it and splits it into train/test `TensorDataset`s.
3. **`EffecientRes.py`** — the model itself (my custom CNN, which I've been calling **UBAC**). It takes one image in and outputs a single steering angle.
4. **`TrainAgent.py`** — the PyTorch training loop. Handles the loss, optimiser, data augmentation and saves a model checkpoint every epoch.
5. **`UdacitySocket.py`** — the inference server. Connects to the simulator, runs the model for steering and the PID for throttle, and sends the control commands back.
6. **`PID.py`** — a small reusable PID controller used for throttle/speed control.

## Install

To run this you'll need all the requirements, which are contained in [requirements.txt](requirements.txt). I used an Anaconda environment for my install so I'd recommend you do the same or something similar. To begin install [PyTorch](https://pytorch.org/get-started/locally/) on your machine (it's strongly recommended you have a GPU, though the model is small enough to train on a CPU if you have to). Once PyTorch is in place simply run:

```
pip install -r requirements.txt
```

You'll also need the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) if you want to actually watch the agent drive. Grab it from the link, then download and extract the `simulator_sdc` folder. The simulator needs to be run from **inside** that folder (it expects the `controller` folder to sit next to it).

## Building the dataset

If you're using my dataset you can skip this step and just download the four `.npy` files from [Hugging Face](https://huggingface.co/datasets/JakeWatson/UdacityCarSimDataset/tree/main) into the `npysets` folder. Otherwise, drive the simulator with the mouse to generate a `driving_log.csv`, then run `DatasetBuilder.py`.

The script expects the log to be in the default Udacity location, so edit the `PATH` variable at the bottom of [DatasetBuilder.py](DatasetBuilder.py) to point at your `controller` folder:

```python
PATH = "D:\\Udacity Simulator\\controller\\"
```

The log columns are `C_img, L_img, R_img, Steer, Accel, Deccel, Velocity`. The script applies a moving average to the steering to remove sudden jerks, crops out the top of each frame (rows `60:135` — no useful info up there), resizes to `160x80`, converts to HSV and normalises to `0-1`. It saves the results out to the `npysets` folder. There are two image-loading functions in the file, `returnImageArray` and the much faster `returnImageArrayV2` (which leans on NumPy's multithreading) — the V2 version is the one actually used.

## Loading & balancing the dataset

[LoadTrainingSet.py](LoadTrainingSet.py) exposes a single function, `ReturnDataset`, which does the heavy lifting:

```python
ReturnDataset(
    PATH     : str,   # Folder holding the .npy files (e.g. "npysets")
    split    : float, # Train/test split (e.g. 0.95)
    shuffle  : bool,  # Randomly permute before splitting
    balance  : tuple, # (num_bins, max_per_bin) for frequency balancing
    figure   : str,   # If set, saves a before/after distribution plot
)
```

The raw data is heavily imbalanced — most frames are going fairly straight, so the model would otherwise learn to just hold the wheel. `ClipByFrequency` fixes this by binning the steering values and capping how many samples are kept per bin, which spreads the distribution out evenly. Passing a `figure` name saves the before/after histograms to the `media` folder:

![Dataset distribution before and after balancing](https://github.com/Jake1402/University-of-Brighton-RO501-Coursework/blob/main/media/Dists.png)

## The model (UBAC)

The agent lives in [EffecientRes.py](EffecientRes.py) as the `ConvolutionalAgent` class. It's a CNN built around **residual blocks** and **layer/batch normalisation**. A few design notes:

- **ELU** activations are used throughout for their self-normalising properties and because they can naturally produce `-1` outputs, which is handy for a steering range of `-1` to `1`.
- The first block (`conv_b0`) is just a channel-scaling head to bump the channel count from `3` up to `48`.
- Four `ResidualBlocks` follow, each with a skip connection so gradients flow uninterrupted, combined with BatchNorm + Dropout for regularisation.
- The output is a single value passed through `Tanh`, giving a steering angle in `[-1, 1]`.

```python
from EffecientRes import ConvolutionalAgent

agent = ConvolutionalAgent(action_space=1)   # 1 output = steering angle
```

## Training the agent

[TrainAgent.py](TrainAgent.py) is a standard PyTorch training loop. The key hyperparameters are set as variables at the top of the `__main__` block, so I'd recommend tuning them (a grid search works well):

| Parameter | Default | Notes |
|---|---|---|
| `lr` | `0.0007` | Learning rate for AdamW |
| `epochs` | `20` | Number of training epochs |
| `split` | `0.95` | Train/test split |
| `batch_size` | `32` | Batch size |
| `BINS` / `MAX_PER_BIN` | `31` / `550` | Dataset balancing |

A few extra things going on under the hood:

- **Loss** is `SmoothL1Loss` with a small **L1 regularisation** term added on top, plus gradient clipping at `1.5` to keep things stable.
- **Data augmentation** is applied on the fly: normalisation, a `RandomAffine` jitter, and a custom `RegressiveHFlip` that flips the image horizontally **and** negates the steering label (flipping the image should indeed flip the steering, so this is a valid augmentation for a continuous action).
- A model checkpoint is saved every epoch to the `Models` folder as `Model at epoch - {n}.pt`, and the train/test loss is plotted to `media/loss.png`.

![Training loss over epochs](https://github.com/Jake1402/University-of-Brighton-RO501-Coursework/blob/main/media/loss.png)

A few example trained models (trained on different dataset sizes) are included in the [Models](Models) folder if you want to jump straight to testing.

## Running the agent in the simulator

[UdacitySocket.py](UdacitySocket.py) is the inference server. It spins up a Flask-SocketIO server on port `4567`, receives telemetry (a base64 image + the current speed) from the simulator, runs the model for steering and the PID for throttle, then sends the control commands back. It also opens a small OpenCV window so you can watch the scene exactly as the model sees it.

Before running it, make sure the `PATH` variable points at the model you want to use:

```python
PATH = os.path.abspath("Models/Model at epoch - 20.pt")  # Model to use
```

The throttle target and PID gains are also set at the top of the file:

```python
PGain = 1.2     # Proportional Gain
IGain = 0.15    # Integral Gain
DGain = 0.1     # Derivative Gain
MaxI  = +5      # Upper limit of Integral (prevents blow ups)
LowI  = -5      # Lower limit of Integral (prevents blow ups)
```

To run it, launch the simulator from inside its own folder and then run the agent:

```
python UdacitySocket.py
```

The server will print `Simulator connected: ...` once the sim has hooked in, and from there the agent takes over the steering and throttle.

## Future plans

Future plans for this project include a few things:
- Adding a proper end-to-end dataset collection script so the CSV-to-`.npy` step is fully automated.
- Exposing the PID gains and throttle target as command line arguments rather than editing variables.
- Adding a small GUI/plot to visualise the live steering prediction and PID error while driving.
- Trying out a couple of different architectures and losses to see how far I can push the top speed.
- Cleaning up the code and removing the remaining 'spaghetti'.

## References

[Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) The simulator the agent was built and tested against.<br />
[Building a Self-Driving Car (Udacity course notes)](https://medium.com/self-driving-cars/self-driving-cars-building-a-self-driving-car-477745368d91) A solid walkthrough of the end-to-end approach used here.<br />
[Residual Networks](https://arxiv.org/abs/1512.03385) The paper behind the residual connections used in the UBAC model.<br />
[ELU activation](https://arxiv.org/abs/1502.03167) The paper proposing ELU, used for its self-normalising properties.<br />
