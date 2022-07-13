"""
Creates the cartesian co-ordinate data used to train the neural networks.
The original data points are plotted and saved in a .gif format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import imageio
import csv
from math import pi, sin, cos


def main():
    # Define the data functions
    def spiral_xy(i, spiral_num):
        φ = i/16 * pi
        r = 6.5 * ((104 - i)/104)
        x = (r * cos(φ) * spiral_num)
        y = (r * sin(φ) * spiral_num)
        label = 0 if spiral_num < 0 else 1
        return [x, y, int(label)]

    NUM_POINTS = 100
    a = pd.DataFrame([spiral_xy(i, 1) for i in range(NUM_POINTS)])
    b = pd.DataFrame([spiral_xy(i, -1) for i in range(NUM_POINTS)])


    # Write the data points to a CSV file
    filename = 'spiral_data.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        header = ['x_coord', 'y_coord',	'class']
        writer.writerow(header)
        for i in range(len(a)):
            writer.writerow(a.iloc[i,:])
            writer.writerow(b.iloc[i,:])
    print(f'Data points written to {filename}')


    # Plot the data points as a gif
    image_dir = 'data_images/'
    if not os.path.exists(image_dir): os.makedirs(image_dir)

    filenames = []
    for i in range(0,100):
        filename = f'{image_dir}plot_{i}_.png'

        plt.figure(figsize=(7,5), dpi=100)
        plt.scatter(a.iloc[:i,0], a.iloc[:i,1], label='A')
        plt.scatter(b.iloc[:i,0], b.iloc[:i,1], label='B')
        plt.xlim(-7,7); plt.ylim(-7,7)
        plt.savefig(filename)
        plt.clf()
        plt.close()

        filenames.append(imageio.imread(filename))

    imageio.mimsave(f'{image_dir}spiral.gif', filenames, duration=0.1)
    for file in glob.glob(f'{image_dir}*_.png'): os.remove(file)
    print(f'Saved {image_dir}spiral.gif')


if __name__ == '__main__':
    main()