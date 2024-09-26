import torch
import torch.nn as nn

class RPN_to_GCVIT(nn.Module):
    def __init__(self, b_boxes, patch_size):
        self.patch_size = patch_size
        self.all_bbox = self.get_all_bbox(b_boxes=b_boxes)

    # Return all the  bbox predictions
    def get_all_bbox(self, b_boxes):
        all_bbox = []
        for mask_bbox in b_boxes: 
            for i in range(len(mask_bbox['bbox'])):
                for box in mask_bbox['bbox'][i]:
                    # Append each bounding box with additional informatio
                    all_bbox.append({
                        "case": mask_bbox['case'],
                        "slice_num": i,
                        "bbox": self.adjust_to_patch_size(box),
                    })
        return all_bbox # Return the list

    def __len__(self):
        return len(self.all_bbox)
    
    def __getitem__(self, idx):
        return self.all_bbox[idx]

    def adjust_to_patch_size(self, b_box):
        """
        Adjust the bounding box paramters to match the patch size

        :param bbox: List of top left and bottom right point of the bounding box
        :param bbox[0]: Minimum x coordinate of the location.
        :param bbox[1]: Minimum y coordinate of the location.
        :param bbox[2]: Maximum x coordinate of the location.
        :param bbox[3]: Maximum y coordinate of the location.

        """
        x_min, y_min, x_max, y_max = b_box # Unpack bbox

        # Current height and width of the b_box
        prev_width = abs(x_min - x_max)
        prev_height = abs(y_min - y_max)

        # Height and width difference of the current b_box and the patch_size
        width_diff = self.patch_size[0] - prev_width
        height_diff = self.patch_size[1] - prev_height

        # Determine the padding that will be added or subtracted to the b_box coordinates
        horizontal_padding = width_diff // 2
        vertical_padding = height_diff // 2

        # Adjust the current b_box coordinates with the padding
        new_x_min = x_min - horizontal_padding
        new_y_min = y_min - vertical_padding
        new_x_max = x_max + (width_diff - horizontal_padding)
        new_y_max = y_max + (height_diff - vertical_padding)

        return [new_x_min, new_y_min, new_x_max, new_y_max] # Return the new coordinates


    def get_cropped_locations(self, img, bbox):
        """
        Crop the specified location across all slices of the MRI.

        :param idx: Index of the MRI scan in the dataset.
        :param bbox: List of top left and bottom right point of the bounding box
        :param bbox[0]: Minimum x coordinate of the location.
        :param bbox[1]: Minimum y coordinate of the location.
        :param bbox[2]: Maximum x coordinate of the location.
        :param bbox[3]: Maximum y coordinate of the location.

        """
        try:
            x_min, y_min, x_max, y_max = bbox
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
                raise ValueError("Unsupported tensor dimension. Expected 5D or 4D tensor.")


        except Exception as e:
            print(f"Error processing: {e}")
            raise