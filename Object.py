import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt

# Create a GUI window
root = Tk()
root.withdraw()

# Ask the user to select an image file
file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Check if a file was selected
if file_path:
    # Load the image
    img = cv2.imread(file_path)

    # OpenCV opens images as BRG
    # but we want it as RGB We'll
    # also need a grayscale version
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use minSize because for not
    # bothering with extra-small
    # dots that would look like STOP signs
    stop_data = cv2.CascadeClassifier('stop_data.xml')

    found = stop_data.detectMultiScale(img_gray,
                                       minSize=(20, 20))

    # Don't do anything if there's
    # no sign
    amount_found = len(found)

    if amount_found != 0:
        # There may be more than one
        # sign in the image
        for (x, y, width, height) in found:
            # We draw a green rectangle around
            # every recognized sign
            cv2.rectangle(img_rgb, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 5)

    # Creates the environment of
    # the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
