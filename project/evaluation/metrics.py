import pandas as pd
from project.utils.euclid_dist import get_euclid_dist

def count_metrics(all_marking, predicted_marking):
    fp = pd.DataFrame(columns=['image_id', 'slice_num', 'x', 'y', 'w', 'h'])
    fp_count = 0
    tp = pd.DataFrame(columns=['image_id', 'slice_num', 'x', 'y', 'w', 'h'])
    tp_count = 0
    fn = pd.DataFrame(columns=['image_id', 'slice_num', 'x', 'y', 'w', 'h'])
    fn_count = 0

    merged_df = pd.merge(predicted_marking, all_marking, on=[
                         'image_id', 'slice_num'], suffixes=('_pred', '_true'))

    predicted_marking['key'] = predicted_marking['image_id'] + \
        '_' + predicted_marking['slice_num'].astype(str)
    merged_df['key'] = merged_df['image_id'] + \
        '_' + merged_df['slice_num'].astype(str)

    fp = predicted_marking[~predicted_marking['key'].isin(merged_df['key'])]
    fp = fp.drop(columns=['key'])
    fp_count += len(fp)

    grouped_dict = {}

    grouped = merged_df.groupby(['image_id', 'slice_num'])

    for (image_id, slice_num), group in grouped:
        key = (image_id, slice_num)
        grouped_dict[key] = group

    for key, df in grouped_dict.items():
        x_pred_values = df['x_pred'].values
        y_pred_values = df['y_pred'].values
        x_true_values = df['x_true'].values
        y_true_values = df['y_true'].values

        w_pred_values = df['w_pred'].values
        h_pred_values = df['h_pred'].values

        is_correct = False
        for i in range(len(x_pred_values)):
            pred_cmb = [x_pred_values[i], y_pred_values[i]]
            true_cmb = [x_true_values[i], y_true_values[i]]
            dist = get_euclid_dist(pred_cmb, true_cmb)
            if dist > 20:
                is_correct = False
            else:
                is_correct = True
                break

        new_row = {
            'image_id': key[0],
            'slice_num': key[1],
            'x': x_pred_values[i],
            'y': y_pred_values[i],
            'w': w_pred_values[i],
            'h': h_pred_values[i]
        }
        temp = pd.DataFrame(new_row, index=[0])

        if is_correct:
            tp_count += 1
            tp = pd.concat([tp, temp], ignore_index=True)
        else:
            fp_count += 1
            fp = pd.concat([fp, temp], ignore_index=True)

    all_marking['key'] = all_marking['image_id'] + \
        '_' + all_marking['slice_num'].astype(str)
    tp['key'] = tp['image_id'] + '_' + tp['slice_num'].astype(str)
    
    fn = all_marking[all_marking['key'].isin(tp['key'])]
    fn = fn.drop(columns=['key'])
    fn_count += len(fn)

    tp = tp.drop(columns=['key'])

    return fp, fp_count, tp, tp_count, fn, fn_count