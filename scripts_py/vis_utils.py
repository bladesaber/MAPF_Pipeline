import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np


def plot_Arrow2D(x, y, yaw, arrow_length=1.0, head_width=0.1, ax=None):
    if ax is not None:
        ax.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw), head_width=head_width, fc='r',
                 ec='k')
        ax.plot(x, y, 'xr')
    else:
        plt.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw), head_width=head_width, fc='r',
                  ec='k')
        plt.plot(x, y, 'xr')


def plot_Arc2D(
        ax, center, radius, start_angel, end_angel, right_dire, shift_angel=0.0
):
    start_angel = np.rad2deg(start_angel)
    end_angel = np.rad2deg(end_angel)
    shift_angel = np.rad2deg(shift_angel)

    if right_dire:
        tem = start_angel
        start_angel = end_angel
        end_angel = tem

    ax.add_patch(mpatches.Arc(
        center,
        radius * 2.0, radius * 2.0,
        angle=shift_angel,
        theta1=start_angel, theta2=end_angel,
    ))


def plot_Path2D(ax, wayPoints):
    ax.scatter(wayPoints[:, 0], wayPoints[:, 1])


def create_Graph3D(
        xmax, ymax, zmax, xmin=0., ymin=0., zmin=0.
):
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    ax.grid(True)

    return ax


def plot_Arrow3D(ax, xyz, vec, length=1.0):
    ax.quiver(
        xyz[0], xyz[1], xyz[2],
        vec[0], vec[1], vec[2],
        length=length, normalize=True, color='r'
    )


def plot_Path3D(ax, xyzs, color):
    # ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])
    ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], '*-', c=color)
    # ax.scatter(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])
