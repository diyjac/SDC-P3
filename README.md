SDC-P3
======
Udacity Self-Driving Car Project 3: Behavioral Cloning

Use Behavioral Cloning to train a CNN regression model to drive a car in a simulator.

SDC-P3 is a program written in Python to train a regression Keras CNN model with a Tensorflow backend to steer a simulated car in Udacity's Open Sourced Self-Driving Car Simulator: [https://github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim).  More details about the model in this repository can be found [here](./README-DETAILED.md)

*NOTE: The model in this repository was trained in the older simulator*

## Installation

This project uses python 3.5.2.  Clone the GitHub repository, and use Udacity [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) to get the rest of the dependencies.

```
$ get clone https://github.com/diyjac/SDC-P3.git
```

## Usage

Once you have recorded a set of images and steering angles from track 1 of the simulator, you can start training our model by executing this line:

```
python model.py
```

Once the `model.json` and `model.h5` have been generated by the model trainer, you can start up the simulator in autonomous mode and see how will it performs.

## Contributing

No futher updates nor contributions are requested.  This project is static.

## License

SDC-P3 results are released under [MIT License](./LICENSE)
