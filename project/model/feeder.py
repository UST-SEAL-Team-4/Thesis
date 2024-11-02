import torch
import torch.nn as nn

class Feeder(nn.Module):
    def __init__(self, resize):
        super().__init__()
        self.resize = resize

    def forward(self, img, bbox, default_size):
        """
        forward feed the the interpolated specified location across all slices of the MRI.

        :param bbox: List of top left and bottom right point of the bounding box
        :param bbox[0]: Minimum x coordinate of the location.
        :param bbox[1]: Minimum y coordinate of the location.
        :param bbox[2]: Maximum x coordinate of the location.
        :param bbox[3]: Maximum y coordinate of the location.

        """
        if any(x < 0 for x in bbox):
            raise Exception("Bounding box contains invalid values.")
        if all(x == 0 for x in bbox):
            return torch.zeros(img.shape[0], 1, default_size, default_size)
        try:
            # x_min, y_min, x_max, y_max = bbox
            x, y, X, Y = bbox

            if X >= x:
                x_min = x
                x_max = X
            else:
                x_min = X
                x_max = x

            if Y >= y:
                y_min = y
                y_max = Y
            else:
                y_min = Y
                y_max = y

            cropped_slices = []
            shape = img.shape

            # Reshape the image into 4 domensions
            img = img.view(shape[0], -1, shape[-2], shape[-1])

            for i in range(img.shape[0]):
                img_slice = img[i, 0]  # Extract the 2D slice (assuming single channel)
                cropped_slice = img_slice[y_min:y_max, x_min:x_max]
                augmented_img = self.resize(image=cropped_slice.numpy()) # Resize the image
                cropped_slices.append(torch.Tensor(augmented_img['image']))

            return torch.stack(cropped_slices, dim=0)


        except Exception as e:
            print(f"Error processing: {e}")
            raise
