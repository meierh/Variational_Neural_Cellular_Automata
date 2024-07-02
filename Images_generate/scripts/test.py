def binary_x(x, threshold=0.5):
    return (x > threshold).float()


binary_x(0.3)