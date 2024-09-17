import os

class Dataset:
    def __init__(self):
        self.dataset_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset'))
        self.raw_mri_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/mri_t2s_dir'))
        self.cmb_masks_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/cmb_masks_dir'))
        # self.skullstripped_dir = os.path.abspath(os.path.join(os.getcwd(), '../Dataset/VALDO_dataset/preprocessed_dir'))
        self.cases = {'cohort1': [], 'cohort2': [], 'cohort3': []}
        self._load_cases()
        
    def _load_cases(self):
        """
        Load all cases into a dictionary, grouping them by cohorts.
        It will check all directories (raw MRI, CMB masks, and skull-stripped MRI)
        and add those in the dictionary.
        """
        
        def add_case_to_cohort(case):
            """ 
            Add the case number to their corresponding cohorts.
            """
            if case.startswith('sub-1'):
                if case not in self.cases['cohort1']:
                    self.cases['cohort1'].append(case)
            elif case.startswith('sub-2'):
                if case not in self.cases['cohort2']:
                    self.cases['cohort2'].append(case)
            else:
                if case not in self.cases['cohort3']:
                    self.cases['cohort3'].append(case)

        """ 
        Check raw MRI directory 
        For example, `sub-101_space-T2S_desc-masked_T2S.nii.gz` -> `sub-101`,
        so it gets added to cohort1
        """
        for file in os.listdir(self.raw_mri_dir):
            if file.startswith('sub-'):
                case = file.split('_')[0]
                add_case_to_cohort(case)

        """ 
        Check CMB masks directory 
        For example, `sub-101_space-T2S_CMB.nii.gz` -> `sub-101`,
        so it gets added to cohort1
        """
        for file in os.listdir(self.cmb_masks_dir):
            if file.startswith('sub-'):
                case = file.split('_')[0]
                add_case_to_cohort(case)

        """ 
        Check skull-stripped MRI directory 
        For example, `sub-101_space-T2S_desc-masked_T2S_stripped.nii.gz` -> `sub-101`,
        so it gets added to cohort1
        """
        # for file in os.listdir(self.skullstripped_dir):
        #     if file.startswith('sub-'):
        #         case = file.split('_')[0]
        #         add_case_to_cohort(case)
    
    def load_cmb_masks(self, cohort_num=None):
        """
        Load CMB mask files. There is also an option
        to load them by a specific cohort.
        """
        masks = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                mask_path = os.path.join(self.cmb_masks_dir, f"{case}_space-T2S_CMB.nii.gz")
                masks.append(mask_path)
        
        return masks
    
    def load_raw_mri(self, cohort_num=None):
        """
        Load raw MRI files. There is also an option
        to load them by a specific cohort.
        """
        mri_files = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                mri_path = os.path.join(self.raw_mri_dir, f"{case}_space-T2S_desc-masked_T2S.nii.gz")
                mri_files.append(mri_path)
        
        return mri_files
    
    def load_skullstripped_mri(self, cohort_num=None):
        """
        Load skull-stripped MRI files. There is also an option
        to load them by a specific cohort.
        """
        stripped_files = []
        cohorts = [f'cohort{cohort_num}'] if cohort_num else self.cases.keys()
        
        for cohort in cohorts:
            for case in self.cases[cohort]:
                stripped_path = os.path.join(self.skullstripped_dir, f"{case}_space-T2S_desc-masked_T2S_stripped.nii.gz")
                stripped_files.append(stripped_path)
        
        return stripped_files
    
    def get_cohort_lengths(self):
        """
        Get the number of cases in each cohort.
        """
        return {cohort: len(cases) for cohort, cases in self.cases.items()}