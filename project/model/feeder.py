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
            x, y, w, h = bbox
            cropped_slices = []
            shape = img.shape

            x_min = min(x, w)
            x_max = max(x, w)
            y_min = min(y, h)
            y_max = max(y, h)

            # Reshape the image into 4 domensions
            img = img.view(shape[0], -1, shape[-2], shape[-1])

            for i in range(img.shape[0]):
                img_slice = img[i, 0]  # Extract the 2D slice (assuming single channel)
                cropped_slice = img_slice[y_min:y_max, x_min:x_max]
                height, width = cropped_slice.shape[:2]

                # Apply perfect square changes 
                if height > width:
                    diff = height - width
                    x_min = max(0, x_min - diff // 2) # Center the cmb
                    x_max = min(img_slice.shape[1], x_max + diff // 2)
                elif width > height:
                    diff = width - height
                    y_min = max(0, y_min - diff // 2)  # Center the cmb
                    y_max = min(img_slice.shape[0], y_max + diff // 2)

                # Re-crop to get the square
                cropped_slice = img_slice[y_min:y_max, x_min:x_max]
                augmented_img = self.resize(image=cropped_slice.numpy(force=True)) # Resize the image
                cropped_slices.append(torch.Tensor(augmented_img['image']))

            return torch.stack(cropped_slices, dim=0)


        except Exception as e:
            print(f"Error processing: {e}")
            raise

class AnchorFeeder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = int(patch_size)
        self.i2p = nn.Unfold(kernel_size = int(patch_size), stride = int(patch_size))

    def forward(self, img, confidence_scores):
        '''
        Find section with the highest confidence_score and return as patch
        '''

        assert len(img.shape) == 4, 'Image must be of dim 3'
        assert len(confidence_scores.shape) == 2, 'confidence_scores must be of dim 2'

        # patch_index = 
        patches = self.i2p(img)
        patches = patches.permute(0, 2, 1)
        out = patches.view(patches.shape[0], -1, self.patch_size, self.patch_size)
        return out
