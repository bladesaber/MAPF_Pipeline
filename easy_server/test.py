import numpy as np

def xyz2flag(x: int, y: int, z: int, x_size: int, y_size: int, z_size: int):
    return z * x_size * y_size + y * x_size + x

def flag2xyz(flag: int, x_size: int, y_size: int, z_size: int):
    z = flag // (x_size * y_size)
    y = (flag - z * x_size * y_size) // x_size
    x = flag - z * x_size * y_size - y * x_size
    return x, y, z

a = [65, 16, 0]
flage = xyz2flag(a[0], a[1], a[2], 131, 107, 95)
x, y, z = flag2xyz(flage, 131, 107, 95)
print(x, y, z)