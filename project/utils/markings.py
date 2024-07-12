import pandas as pd

def get_predicted_marking_validation(prediction_list, slice_num, id, score_threshold):
    predicted_cmbs = {
        'image_id': [],
        'slice_num': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }

    for box in prediction_list:
        if box[4].item() > score_threshold:
            predicted_cmbs['image_id'].append(id)
            predicted_cmbs['slice_num'].append(slice_num)
            predicted_cmbs['x'].append(box[0].item())
            predicted_cmbs['y'].append(box[1].item())
            predicted_cmbs['w'].append(box[2].item())
            predicted_cmbs['h'].append(box[3].item())

    # Convert to DataFrame once at the end
    predicted_cmbs_df = pd.DataFrame(predicted_cmbs)
    return predicted_cmbs_df

def get_all_marking(dataset):
    all_cmbs = {
        'image_id': [],
        'slice_num': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }
    for i in range(len(dataset)):
        slices, targets, id, _ = dataset[i]
        for j in range(len(slices)):
            for target in targets[j]['boxes']:
                all_cmbs['image_id'].append(id)
                all_cmbs['slice_num'].append(j)
                all_cmbs['x'].append(target[0].item())
                all_cmbs['y'].append(target[1].item())
                all_cmbs['w'].append(target[2].item())
                all_cmbs['h'].append(target[3].item())

    # Convert to DataFrame once at the end
    all_cmbs = pd.DataFrame(all_cmbs)
    return all_cmbs