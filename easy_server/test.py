import numpy as np

def xyz2flag(x: int, y: int, z: int, x_size: int, y_size: int, z_size: int):
    return x * y_size * z_size + y * z_size + z

def flag2xyz(flag: int, x_size: int, y_size: int, z_size: int):
    z = flag % z_size
    y = (flag // z_size) % y_size
    x = flag // (x_size * y_size)
    return x, y, z

x_size = 10
y_size = 10
z_size = 10

for _ in range(100):
    x = np.random.randint(0, x_size)
    y = np.random.randint(0, y_size)
    z = np.random.randint(0, z_size)
    flag = xyz2flag(x, y, z, x_size, y_size, z_size)
    x2, y2, z2 = flag2xyz(flag, x_size, y_size, z_size)
    assert x == x2 and y == y2 and z == z2

print('Test passed')