import os

class Dataset:
    def __init__(self):
        self.dataset_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset'))
        self.raw_mri_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/mri_t2s_dir'))
        self.cmb_masks_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/cmb_masks_dir'))
        self.skullstripped_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/preprocessed_dir'))
        self.cases = {'cohort1': [], 'cohort2': [], 'cohort3': []}
        self._load_cases()
        
    def _load_cases(self):
        """
        Load all cases into a dictionary, grouping them by cohorts.
        This method will check all relevant directories (raw MRI, CMB masks, and skull-stripped MRI) 
        to identify cases.
        """
        def add_case_to_cohort(case):
            """ Helper function to add a case to the appropriate cohort """
            if case.startswith('sub-1'):
                if case not in self.cases['cohort1']:
                    self.cases['cohort1'].append(case)
            elif case.startswith('sub-2'):
                if case not in self.cases['cohort2']:
                    self.cases['cohort2'].append(case)
            else:
                if case not in self.cases['cohort3']:
                    self.cases['cohort3'].append(case)

        # Check raw MRI directory
        if os.path.exists(self.raw_mri_dir):
            for file in os.listdir(self.raw_mri_dir):
                if file.startswith('sub-'):
                    case = file.split('_')[0]
                    add_case_to_cohort(case)

        # Check CMB masks directory
        if os.path.exists(self.cmb_masks_dir):
            for file in os.listdir(self.cmb_masks_dir):
                if file.startswith('sub-'):
                    case = file.split('_')[0]
                    add_case_to_cohort(case)

        # Check skull-stripped MRI directory
        if os.path.exists(self.skullstripped_dir):
            for file in os.listdir(self.skullstripped_dir):
                if file.startswith('sub-'):
                    case = file.split('_')[0]
                    add_case_to_cohort(case)
    
    def load_cmb_masks(self, cohort_num=None):
        """
        Load CMB mask files. Optionally, filter by cohort.
        """
        masks = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                mask_path = os.path.join(self.cmb_masks_dir, f"{case}_space-T2S_CMB.nii.gz")
                if os.path.exists(mask_path):
                    masks.append(mask_path)
        
        return masks
    
    def load_raw_mri(self, cohort_num=None):
        """
        Load raw MRI files. Optionally, filter by cohort.
        """
        mri_files = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                mri_path = os.path.join(self.raw_mri_dir, f"{case}_space-T2S_desc-masked_T2S.nii.gz")
                if os.path.exists(mri_path):
                    mri_files.append(mri_path)
        
        return mri_files
    
    def load_skullstripped_mri(self, cohort_num=None):
        """
        Load skull-stripped MRI files. Optionally, filter by cohort.
        """
        stripped_files = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                stripped_path = os.path.join(self.skullstripped_dir, f"{case}_space-T2S_desc-masked_T2S_stripped.nii.gz")
                if os.path.exists(stripped_path):
                    stripped_files.append(stripped_path)
        
        return stripped_files
    
    def get_cohort_lengths(self):
        """
        Get the number of cases in each cohort.
        """
        return {cohort: len(cases) for cohort, cases in self.cases.items()}

# class Dataset:
#     FOLDER_DIR = 'VALDO_Dataset\Task2'
#     PREPROCESSED_DIR = 'VALDO_Dataset\preprocessed_dataset'
#     LABEL_FNAME = '_space-T2S_CMB.nii.gz'
#     ID_FNAME = '_space-T2S_desc-masked_T2S.nii.gz'
#     ID_PREPROCESSED_FNAME = '_space-T2S_desc-masked_T2S_stripped.nii.gz'
    
#     def __init__(self):
#         self.current_dir = os.path.abspath(os.path.join(os.getcwd(), '../', self.FOLDER_DIR))
#         self.preprocessed_dir = os.path.abspath(os.path.join(os.getcwd(), '../', self.PREPROCESSED_DIR))
#         self.cases = {'cohort1': [], 'cohort2': [], 'cohort3': []}
#         self._load_cases()
        
#     def _load_cases(self):
#         folders = [item for item in os.listdir(self.current_dir) if os.path.isdir(os.path.join(self.current_dir, item))]

#         for folder in folders:
#             if 'sub-1' in folder:
#                 self.cases['cohort1'].append(folder)
#             elif 'sub-2' in folder:
#                 self.cases['cohort2'].append(folder)
#             else:
#                 self.cases['cohort3'].append(folder)
                
#     def get_cohort_lengths(self):
#          return {cohort: len(cases) for cohort, cases in self.cases.items()}

#     def generate_labels_and_ids(self, cohort_num=None):
#         labels = []
#         ids = []
        
#         if cohort_num:
#             cohorts = [f'cohort{cohort_num}']
#         else:
#             cohorts = self.cases.keys()
            
#         for cohort in cohorts:
#             for case in self.cases[cohort]:
#                 label = f'{self.current_dir}\\{case}\\{case}{self.LABEL_FNAME}'
#                 id = f'{self.current_dir}\\{case}\\{case}{self.ID_FNAME}'
#                 labels.append(label)
#                 ids.append(id)
            
#         return labels, ids

#     def get_all_labels_and_ids(self):
#         return self.generate_labels_and_ids()
    
#     def get_labels_and_ids_by_cohort(self, cohort_num):
#         return self.generate_labels_and_ids(cohort_num)

#     def get_preprocessed_labels_and_ids(self):
#         labels = []
#         ids = []
        
#         for cohort in self.cases.keys():
#             for case in self.cases[cohort]:
#                 label = f'{self.current_dir}\\{case}\\{case}{self.LABEL_FNAME}'
#                 id = f'{self.current_dir}\\{case}{self.ID_PREPROCESSED_FNAME}'
                
#                 if os.path.exists(label) and os.path.exists(id):
#                     labels.append(label)
#                     ids.append(id)
#         return labels, ids