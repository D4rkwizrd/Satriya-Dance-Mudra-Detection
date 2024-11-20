import cv2
import numpy as np

# Function to create a collage of images with names and white borders
def create_collage(images, names):
    # Create a blank canvas for the collage
    collage_width = 1900
    collage_height = 1267
    collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255  # White background

    # Set border size
    border_size = 10

    # Iterate through the images and paste them on the canvas with borders
    for i in range(len(images)):
        row = i // 4
        col = i % 4

        # Calculate the position to paste the image
        start_row = row * (collage_height // 2)
        end_row = start_row + (collage_height // 2)
        start_col = col * (collage_width // 4)
        end_col = start_col + (collage_width // 4)

        # Resize the image to fit the collage
        resized_img = cv2.resize(images[i], (collage_width // 4 - 2 * border_size, collage_height // 2 - 2 * border_size))

        # Create a white border around the image
        bordered_img = cv2.copyMakeBorder(resized_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Paste the bordered image on the canvas
        collage[start_row:end_row, start_col:end_col, :] = bordered_img

        # Display the name of the image at the bottom in bold red
        text_position = (start_col + border_size + 10, end_row - border_size - 10)
        cv2.putText(collage, names[i], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return collage

# Load 8 sample images
image_paths = ['/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Bardhaman.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Dol.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Gajadanta.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Kaput.png',
               '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Karkat.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Makar1.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Makar2.png', '/media/dishanka/6232A6F432A6CC7B/02 Project/Detection_WEB_APP/Image/Samput.png']

# Read images
images = [cv2.imread(img_path) for img_path in image_paths]

# Names of the images
image_names = ['Bardhaman', 'Dol', 'Gajadanta', 'Kaput',
               'Karkat', 'Makar1', 'Makar2', 'Samput']

# Create the collage with white borders and bold red names
collage = create_collage(images, image_names)

# Display the collage
cv2.imshow('Image Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the collage
cv2.imwrite('collage_output.png', collage)
