import pandas as pd
import re, copy, os
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    def __init__(self, input_dir= r"outputs", sel_prop = 'refractive index', lb= 1.5, ub= 2.5):
        self.input_dir = input_dir
        self.df_abs_dir = os.path.join(input_dir, "output_abs_clean.csv")
        self.df_exp_dir = os.path.join(input_dir, "output_exp_clean.csv")
        
        self.sel_prop = sel_prop
        self.lb = lb
        self.ub = ub
    def props_select(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, low_memory= False)
        self.df_merged = pd.merge(self.df_abs, self.df_exp, on=['doi', 'mat'])
        
        df_clean = pd.DataFrame()
        props_columns = [col for col in self.df_merged.columns if re.match(r'^props_\d+$', col)]
        for prop_col in props_columns:
            df_tmp = self.df_merged.loc[self.df_merged[prop_col] == self.sel_prop]
            df_clean = pd.concat([df_clean, df_tmp], axis= 0, ignore_index=True)
        df_clean.drop_duplicates(inplace=True)
        df_clean.reset_index(inplace=True, drop=True)
        
        df_prop_clean = copy.deepcopy(df_clean)
        df_prop_clean.dropna(inplace=True, axis=1, how= "all")

        for prop_col in props_columns:
            col_num = re.findall(r'\d+', prop_col)[0]
            value_col = f'props_{col_num}_value'
            unit_col = f'props_{col_num}_unit'
            cols = [prop_col, value_col, unit_col]
            for idx, prop in df_clean[prop_col].items():
                if prop == self.sel_prop:
                    df_prop_clean.loc[idx, ['props_1', 'props_1_value', 'props_1_unit']] = list(df_clean.loc[idx, cols])

        df_prop_clean.drop_duplicates(inplace=True)
        df_prop_clean.dropna(inplace=True, axis= 1, how='all')
        df_prop_clean.reset_index(inplace=True, drop=True)

        return df_prop_clean
    
    def fea_map(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, low_memory= False)

        #mat label map
        mat_map = {}
        prc_columns = [col for col in self.df_exp.columns if re.match(r'^prc\d+_name$', col)]
        all_materials = pd.concat([
            self.df_abs.mat, 
            self.df_exp.mat, 
            pd.Series(self.df_exp[prc_columns].values.flatten())
        ], ignore_index=True)        
        unique_materials = all_materials.unique()
        mat_map = {k: i for i, k in enumerate(unique_materials)}

        #unit label map
        unit_map = {}
        props_unit_columns = [col for col in self.df_abs.columns if re.match(r'^props_\d+_unit$', col)]
        exp_unit_columns = [col for col in self.df_exp.columns if re.match(r'^prc\d+_unit$', col)]
        proc_unit_columns = [col for col in self.df_exp.columns if re.match(r'^post_proc\d+_unit$', col)]
        
        all_units = pd.concat([
            self.df_abs[props_unit_columns].stack(),
            self.df_exp[exp_unit_columns].stack(),
            self.df_exp[proc_unit_columns].stack()
        ]).dropna().astype(str).unique()

        unit_map = {k: i for i, k in enumerate(all_units)}


        #syns_method freq map
        syns_method_map = {}
        syns_counts = self.df_exp['syns_method'].value_counts(dropna=False)
        total = syns_counts.sum()
        syns_method_map = {
            str(k).strip(): v/total
            for k, v in syns_counts.items()
            if pd.notnull(k)
        }

        #post_proc freq map
        post_proc_map = {}

        data = pd.read_csv(r"outputs/words_count/post_procs_count.csv")
        scount = sum(data['count'])

        post_proc_map = {}
        seen = set()
        for _, row in data.iterrows():
            k = str(row['post_proc']).strip()
            if k and k not in seen:
                post_proc_map[k] = row['count'] / scount
                seen.add(k)

        return mat_map, unit_map, syns_method_map, post_proc_map
    

    def fea_transform(self, df_prop_clean, mat_map, syns_method_map, unit_map, post_proc_map):
        required_columns = {'syns_method'} | \
                      set(col for col in df_prop_clean.columns if '_name' in col or '_unit' in col)
        missing_cols = required_columns - set(df_prop_clean.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}.\n"
                             f"Available columns: {df_prop_clean.columns.tolist()}")
        
        df_prop_clean['syns_method'] = df_prop_clean['syns_method'].map(syns_method_map)

        prc_columns = [col for col in df_prop_clean.columns if re.match(r'^prc\d+_name$', col)]
        for col in prc_columns:
            df_prop_clean[col] = df_prop_clean[col].map(mat_map)

        unit_cols = [col for col in df_prop_clean.columns if col.endswith('_unit')]
        for col in unit_cols:
            df_prop_clean[col] = df_prop_clean[col].map(unit_map)

        post_proc_columns = [col for col in df_prop_clean.columns if re.match(r'^post_proc\d+_step$', col)]
        for col in post_proc_columns:
            df_prop_clean[col] = df_prop_clean[col].map(post_proc_map)

        df_prop_clean_trans = df_prop_clean.fillna(0)

        return df_prop_clean_trans
    
    def data_aug(self,
             X, y,
             data_aug_num = 1500,
             noise_factor_min=0.01,
             noise_factor_max=0.15,
             noise_distribution='normal'):

        all_X_augmented = [copy.deepcopy(X)]
        all_y_augmented = [copy.deepcopy(y)]

        num =  data_aug_num // X.shape[0]

        if num > 0:
            num_sets = num + 1
        else:
            num_sets = 1

        for _ in range(num_sets):
            X_noisy_copy = X.copy()
            y_noisy_copy = y.copy()
            current_noise_factor = np.random.uniform(noise_factor_min, noise_factor_max)
    
            for col in X_noisy_copy.columns:
                if pd.api.types.is_numeric_dtype(X_noisy_copy[col]):
                    feature_std = X[col].std()
                    if feature_std == 0:
                        continue
                    noise_magnitude = feature_std * current_noise_factor * 0.6
                    if noise_distribution == 'normal':
                        noise = np.random.normal(0, noise_magnitude, X_noisy_copy.shape[0])
                    elif noise_distribution == 'uniform':
                        limit = np.sqrt(3) * noise_magnitude
                        noise = np.random.uniform(-limit, limit, X_noisy_copy.shape[0])
                    else:
                        noise = 0
                    X_noisy_copy[col] += noise
    
            if pd.api.types.is_numeric_dtype(y_noisy_copy):
                y_std = y.std()
                if y_std > 0:
                    noise_magnitude_y = y_std * current_noise_factor * 0.2
                    if noise_distribution == 'normal':
                        y_noise = np.random.normal(0, noise_magnitude_y, y_noisy_copy.shape[0])
                    elif noise_distribution == 'uniform':
                        limit_y = np.sqrt(2) * noise_magnitude_y
                        y_noise = np.random.uniform(-limit_y, limit_y, y_noisy_copy.shape[0])
                    else:
                        y_noise = 0
                    y_noisy_copy += y_noise
    
            all_X_augmented.append(X_noisy_copy)
            all_y_augmented.append(y_noisy_copy)
    
        X_final_augmented = pd.concat(all_X_augmented, ignore_index=True)
        y_final_augmented = pd.concat(all_y_augmented, ignore_index=True)
        y_final_augmented = y_final_augmented[(y_final_augmented >= self.lb) & (y_final_augmented <= self.ub)]
            
        Q1 = y_final_augmented.quantile(0.25)
        Q3 = y_final_augmented.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        valid_indices = y_final_augmented[(y_final_augmented >= lower_bound) & 
                                          (y_final_augmented <= upper_bound)].index
        X_final_augmented_clean = X_final_augmented.loc[valid_indices]
        y_final_augmented_clean = y_final_augmented.loc[valid_indices]
    
        return X_final_augmented_clean, y_final_augmented_clean
    
    def exe(self):
        print("Starting feature extraction...")
        df_prop_clean = self.props_select()
        mat_map, unit_map, syns_method_map, post_proc_map = self.fea_map()
        df_prop_clean_trans = self.fea_transform(df_prop_clean, mat_map, syns_method_map, unit_map, post_proc_map)

        X = df_prop_clean_trans.loc[:, 'syns_method':]
        y_values = df_prop_clean_trans['props_1_value']
        X = X[X.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
        y = y_values[X.index]
        cols  = X.columns

        if X.shape[0] <= 400:
            data_aug_num = min(X.shape[0]*40, 400)
            print("Starting data augmentation......")
            X_final_augmented_clean, y_final_augmented_clean = self.data_aug(X, y, data_aug_num)
        else:
            X_final_augmented_clean, y_final_augmented_clean = X, y

        scaler = StandardScaler()
        X_final_augmented_clean = scaler.fit_transform(X_final_augmented_clean)
        X_final_augmented_clean = pd.DataFrame(X_final_augmented_clean)
        X_final_augmented_clean.columns = cols

        X_final_augmented_clean.to_csv(os.path.join(self.input_dir, "X.csv"))
        y_final_augmented_clean.to_csv(os.path.join(self.input_dir, "y.csv"))