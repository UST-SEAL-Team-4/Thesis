import torch
import torch.nn as nn

class RPN_to_GCVIT():
    def get_cropped_locations(self, img, x_min, y_min, x_max, y_max):
        """
        Display the specified location across all slices of the MRI.

        :param idx: Index of the MRI scan in the dataset.
        :param x_min: Minimum x coordinate of the location.
        :param y_min: Minimum y coordinate of the location.
        :param x_max: Maximum x coordinate of the location.
        :param y_max: Maximum y coordinate of the location.
        """
        try:
            cropped_slices = []

            # Iterate through each slice and crop to the specified region
            if img.dim() == 5:
                # If 5D, assume shape is [num_slices, batch, channels, height, width]
                for i in range(img.__len__()):
                    img_slice = img[i]
                    cropped_slice = img_slice[0, 0, y_min:y_max, x_min:x_max]
                    cropped_slices.append(torch.Tensor(cropped_slice))
                
                return torch.stack(cropped_slices, dim=0).unsqueeze(0).unsqueeze(0)

            elif img.dim() == 4:
                # If 4D, assume shape is [num_slices, channels, height, width]
                for i in range(img.shape[0]):
                    img_slice = img[i, 0]  # Extract the 2D slice (assuming single channel)
                    cropped_slice = img_slice[y_min:y_max, x_min:x_max]
                    cropped_slices.append(torch.Tensor(cropped_slice))

                return torch.stack(cropped_slices, dim=0).unsqueeze(0)

            else:
                raise ValueError("Unsupported tensor dimension. Expected 3D or 4D tensor.")


        except Exception as e:
            print(f"Error processing: {e}")
            raise