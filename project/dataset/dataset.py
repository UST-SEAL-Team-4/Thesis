import os

class Dataset:
    FOLDER_DIR = 'VALDO_Dataset\Task2'
    LABEL_FNAME = '_space-T2S_CMB.nii.gz'
    ID_FNAME = '_space-T2S_desc-masked_T2S.nii.gz'
    
    def __init__(self):
        self.current_dir = os.path.abspath(os.path.join(os.getcwd(), '../', self.FOLDER_DIR))
        self.cases = {'cohort1': [], 'cohort2': [], 'cohort3': []}
        self._load_cases()
        
    def _load_cases(self):
        folders = [item for item in os.listdir(self.current_dir) if os.path.isdir(os.path.join(self.current_dir, item))]

        for folder in folders:
            if 'sub-1' in folder:
                self.cases['cohort1'].append(folder)
            elif 'sub-2' in folder:
                self.cases['cohort2'].append(folder)
            else:
                self.cases['cohort3'].append(folder)

    def generate_labels_and_ids(self, cohort_num):
        labels = []
        ids = []
        
        for case in self.cases[cohort_num]:
            label = f'{self.current_dir}\\{case}\\{case}{self.LABEL_FNAME}'
            id = f'{self.current_dir}\\{case}\\{case}{self.ID_FNAME}'
            labels.append(label)
            ids.append(id)
            
        return labels, ids

    def get_all_labels_and_ids(self):
        cohort1_labels, cohort1_ids = self.generate_labels_and_ids('cohort1')
        cohort2_labels, cohort2_ids = self.generate_labels_and_ids('cohort2')
        cohort3_labels, cohort3_ids = self.generate_labels_and_ids('cohort3')
        all_labels = cohort1_labels + cohort2_labels + cohort3_labels
        all_ids = cohort1_ids + cohort2_ids + cohort3_ids
        
        return all_labels, all_ids
