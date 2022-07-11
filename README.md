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

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/data_images/spiral.gif" width="500" height="400"/>

Let's assume that we saw this pattern in a data set naturally, and that we wish to train a neural network which will learn to classify points between the two classes.

The `RawNet` model defined in `spiral_models.py` is a simple neural network with 2 input nodes, 2 hidden layers with as many hidden nodes as the user specifies (we'll use 20), and 1 output node. An illustration of the architecture is as follows:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/rawnet.png" width="500" height="400"/>

The model can be run by submitting the following terminal command from the project folder's directory:
```
python3 spiral_main.py --net 'raw' --hid 20
```
The `--net` flag tells the script we want to use the `RawNet` model, and the `--hid` flag tells the script to create the model with 20 nodes in each of the 2 hidden layers.

The terminal will show the training status of the model:
```
ep:  100 loss: 0.1597 acc: 54.00
ep:  200 loss: 0.0815 acc: 54.50
ep:  300 loss: 0.0316 acc: 53.00
ep:  400 loss: 0.0153 acc: 53.50
...
```
After several thousand epochs, the training accuracy will reach 100%, and 3 images will be created in the `images` folder.
These images show iteratively how each node in each hidden layer divides up the input space to assist classification.

Hidden layer 1 activations:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_1_20.gif" width="500" height="400"/>

Hidden layer 2 activitions:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_2_20.gif" width="500" height="400"/>

We can see more complex shapes arise in the second hidden layer. This is because nodes in the second layer are able to combine inputs from the first layer, allowing for further abstraction.

Final combined activation:

<img src="https://raw.githubusercontent.com/dmc-au/twin-spirals/main/images/raw_out.png" width="500" height="400"/>

The image above shows how the network combines all of its nodes to create a map of the input space in order to classify between classes. Although the coverage is complete, we can see that the network hasn't grasped that there is an underlying symmetry to leverage.


