# Twin Spirals

This project visually explores how the framing of a problem can dramatically affect the training performance of neural networks.

The project starts with the creation of some data points according to the following formulae for two classes, A and B:

Class A
$$ x = r * math.cos(φ) $$
$$ y = r * math.sin(φ) $$

Class B
$$ x = -r * math.cos(φ) $$
$$ y = -r * math.sin(φ) $$

One hundred data points are created for both classes by iteratively updating both the 'r' and 'φ' parameters.
Points beloning to class A are labeled with a 1, and points belonging to class B are labeled with a 0.

By graphing these points, we can see that two spirals are created, one for each class of points.
![Original data](data_images/spiral.gif)
