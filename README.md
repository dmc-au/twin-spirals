# Twin Spirals Problem

This project visually explores how the framing of a problem can dramatically affect the training performance of neural networks.

The project starts with the creation of some data points by according to the following formulae for two classes, A and B:

Class A

$$ x = r * math.cos(φ) $$

$$ y = r * math.sin(φ) $$

Class B

$$ x = -r * math.cos(φ) $$

$$ y = -r * math.sin(φ) $$

One hundred data points are created for both classes by iteratively updating both the 'r' and 'φ' parameters.
Points belonging to class A are labeled with a 1, and points belonging to class B are labeled with a 0.
After the required packages are installed, the data can be created by running the `spiral_create_data.py` script.
The data is stored in the file `spiral_data.csv`.

By graphing these points, we can see that two spirals are created, one for each class of points. The image is located in the `data_images` folder.

<img src="https://github.com/dmc-au/twin-spirals/blob/main/images/spiral.gif?raw=true" width="500" height="400"/>

Let's assume that we saw this pattern in a data set naturally, and that we wish to train a neural network which will learn to classify points between the two classes.

The `RawNet` model defined in `spiral_models.py` is a simple neural network with 2 input nodes, 2 hidden layers with as many hidden nodes as the user specifies (we'll use 20), and 1 output node. An illustration of the architecture is as follows:

<img src="https://github.com/dmc-au/twin-spirals/blob/main/images/rawnet_architecture.png?raw=true" width="500" height="400"/>

The model can be run by submitting the following terminal command from the project folder's directory:
```
python3 spiral_main.py --net 'raw'
```
The `--net` flag tells the script we want to use the `RawNet` model. There are other hyper-parameters which can be passed to the script to adjust the learning process. These are visible as arguments in `spiral_main.py`.

The terminal will show the training status of the model:
```
INFO 2022-07-12 13:16:31,761 - Training RawNet model; init=0.1, hid=20, lr=0.01, epochs=20000
INFO 2022-07-12 13:16:31,870 - ep:  100 loss: 0.3369 acc: 58.00
INFO 2022-07-12 13:16:31,969 - ep:  200 loss: 0.0494 acc: 61.00
INFO 2022-07-12 13:16:32,066 - ep:  300 loss: 0.0158 acc: 60.00
...
```
After around 4000 epochs, the training accuracy will reach 100%, and 3 images will be created in the `images` folder.
These images show iteratively how each node in each hidden layer divides up the input space to assist classification.

Hidden layer 1 activations:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_1_20.gif" width="500" height="400"/>

Hidden layer 2 activitions:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_2_20.gif" width="500" height="400"/>

We can see more complex shapes arise in the second hidden layer. This is because nodes in the second layer are able to combine inputs from the first layer, allowing for further abstraction.

Final combined activation:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_out.png" width="500" height="400"/>

The image above shows how the network combines all of its nodes to create a map of the input space in order to classify between classes. Although the coverage is complete, we can see that the network hasn't grasped that there is an underlying symmetry to leverage.

Our next step is to use our knowledge of the symmetry to help remodel our task. The symmetry in the data points is _radial_, so with our new network the data will be converted to polar coordinates prior to training. The definition for this model is under `PolarNet` in the `spiral_models.py` file. Note that this network has only one hidden layer, as opposed to the 2 defined in `RawNet`.

We can train this model with the following command from the project folder's directory:
```
python3 spiral_main.py --net 'polar'
```

The terminal will show the training status of the model:
```
INFO 2022-07-13 12:07:22,949 - Training PolarNet model; init=0.1, hid=20, lr=0.01, epochs=20000
INFO 2022-07-13 12:07:23,087 - ep:  100 loss: 0.1294 acc: 52.50
INFO 2022-07-13 12:07:23,169 - ep:  200 loss: 0.0399 acc: 62.50
INFO 2022-07-13 12:07:23,248 - ep:  300 loss: 0.0125 acc: 67.50
```
This time the model trains much faster, and reaches 100% accuracy within around 2000 epochs. That's half the amount of time as the first model, `RawNet`, with half as many hidden nodes. This is the power of having a more suitable representation of the task.

By looking at the hidden node mappings for `PolarNet`, we can see a big difference in how the insput space is mapped:

<img src="https://github.com/dmc-au/twin-spirals/blob/main/images/polar_1_20.gif?raw=true" width="500" height="400"/>

The difference is even more noticable in the graph of the final mapping:

<img src="https://github.com/dmc-au/twin-spirals/blob/main/images/polar_out.png?raw=true" width="500" height="400"/>
