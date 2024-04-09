import matplotlib.pyplot as plt

def set_y_axis_limits(ymin, ymax, percentage):
    # Calculate the range of y
    y_range = ymax - ymin

    # Calculate the offset based on the percentage
    offset = y_range * percentage / 100
    
    # Set the y-axis limits
    plt.ylim(ymin - offset, ymax + offset)
