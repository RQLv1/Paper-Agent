import pandas as pd
import re, os

class DataCount:
    def __init__(self, input_dir= r"outputs", 
                 output_dir_words=r"outputs/words_count",
                 output_dir_val = r"outputs/value_count"
                 ):
        self.df_abs_dir = os.path.join(input_dir, "output_abs_clean.csv")
        self.df_exp_dir = os.path.join(input_dir, "output_exp_clean.csv")
        self.output_dir_words = output_dir_words
        self.output_dir_val = output_dir_val
        
    # props get
    def analyze_props_from_csv(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, index_col= 0, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, index_col= 0, low_memory= False)

        prop_stats = []

        props_columns = [col for col in self.df_abs.columns if re.match(r'^props_\d+$', col)]

        for prop_col in props_columns:
            try:
                col_num_match = re.search(r'\d+', prop_col)
                if not col_num_match:
                    print(f"Could not extract number from column name: {prop_col}. Skipping.")
                    continue
                col_num = col_num_match.group(0)

                unit_col_name = f'props_{col_num}_unit'
                has_unit_column = unit_col_name in self.df_abs.columns

                prop_counts = self.df_abs[prop_col].astype(str).value_counts()

                for prop_name, count in prop_counts.items():
                    if pd.isna(prop_name) or str(prop_name).strip() == "":
                        continue

                    unit = "unk"

                    if has_unit_column:
                        unit_series = self.df_abs.loc[self.df_abs[prop_col].astype(str) == str(prop_name), unit_col_name].dropna()

                        if not unit_series.empty:
                            modes = unit_series.mode()
                            if not modes.empty:
                                unit = modes[0]

                    prop_stats.append({
                        'property': str(prop_name),
                        'count': count,
                        'unit': str(unit)
                    })
            except KeyError as e:
                print(f"KeyError while processing column {prop_col} or its associated unit column: {e}. Skipping this column.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {prop_col}: {e}. Skipping this column.")


        if not prop_stats:
            print("No property statistics were generated.")
            return

        try:
            total_counts_df = (
                pd.DataFrame(prop_stats)
                .groupby(['property', 'unit'], as_index=False)
                .agg(count=('count', 'sum'))
                .sort_values('count', ascending=False)
            )
            total_counts_df.reset_index(inplace=True, drop=True)
        except Exception as e:
            print(f"Error during aggregation of property stats: {e}")
            return

        try:
            os.makedirs(self.output_dir_words, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory '{self.output_dir_words}': {e}")
            return

        total_counts_df.to_csv(os.path.join(self.output_dir_words, "props_count.csv"), index=False)

        return total_counts_df

    #syns method get
    def get_syns_method(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, index_col= 0, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, index_col= 0, low_memory= False)

        counts = self.df_exp['syns_method'].value_counts()
        syns_method_df = pd.DataFrame({
            'syns_method': counts.index,
            'count': counts.values
        })

        syns_method_df.to_csv(os.path.join(self.output_dir_words, "syn_method_count.csv"), index=False)

    #post_proc_count_unit get
    def get_post_proc(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, index_col= 0, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, index_col= 0, low_memory= False)

        post_proc_stats = []
        post_proc_columns = [col for col in self.df_exp.columns if re.match(r'^post_proc\d+_step$', col)]

        for post_proc_col in post_proc_columns:
            col_num = re.findall(r'\d+', post_proc_col)[0]  
            unit_col = f'post_proc{col_num}_unit'

            post_proc_counts = self.df_exp[post_proc_col].value_counts()

            for post_proc_name, count in post_proc_counts.items():
                if unit_col in self.df_exp.columns:
                    unit_series = self.df_exp.loc[self.df_exp[post_proc_col] == post_proc_name, unit_col].dropna()
                else:
                    unit_series = pd.Series(["unk"])
                if not unit_series.empty:
                    unit = unit_series.mode()[0] if not unit_series.empty else "unk"
                else:
                    unit = "unk"

                post_proc_stats.append({
                    'post_proc': post_proc_name,
                    'count': count,
                    'unit': unit
                })

        total_counts = (
            pd.DataFrame(post_proc_stats)
            .groupby(['post_proc', 'unit'], as_index=False)
            .agg(count=('count', 'sum'))
            .sort_values('count', ascending=False)
        )
        total_counts.reset_index(inplace= True, drop=True)
        total_counts.to_csv(os.path.join(self.output_dir_words, "post_procs_count.csv"), index= False)

    def val_count(self):
        self.df_abs = pd.read_csv(self.df_abs_dir, index_col= 0, low_memory= False)
        self.df_exp = pd.read_csv(self.df_exp_dir, index_col= 0, low_memory= False)
        
        df_merged = pd.merge(self.df_abs, self.df_exp, on=['doi', 'mat'])
        df_merged.to_csv(r"outputs/merged.csv",index= False)

        if not os.path.exists(self.output_dir_val):
            os.makedirs(self.output_dir_val)

        #props_val
        df_props_val = pd.DataFrame()
        props_columns = [col for col in self.df_abs.columns if re.match(r'^props_\d+$', col)]
        for prop_col in props_columns:
            col_num = re.findall(r'\d+', prop_col)[0]
            value_col = f'props_{col_num}_value'
            unit_col = f'props_{col_num}_unit'

            if prop_col in self.df_abs.columns and value_col in self.df_abs.columns and unit_col in self.df_abs.columns:
                df_tmp = pd.DataFrame({
                    'Property': self.df_abs[prop_col],
                    'Value': self.df_abs[value_col],
                    'Unit': self.df_abs[unit_col]
                })

                df_props_val = pd.concat([df_props_val, df_tmp], axis=0).reset_index(drop=True)
        df_props_val.dropna(axis= 0, inplace= True)
        df_props_val.reset_index(drop= True, inplace= True)

        df_props_val.to_csv(os.path.join(self.output_dir_val, "props_val_count.csv"), index= False)

        #exp_val
        df_post_proc_val = pd.DataFrame()
        post_proc_columns = [col for col in self.df_exp.columns if re.match(r'^post_proc\d+_step$', col)]
        for post_proc_col in post_proc_columns:
            col_num = re.findall(r'\d+', post_proc_col)[0] 
            value_col = f'post_proc{col_num}_value'
            unit = f'post_proc{col_num}_unit'

            if post_proc_col in self.df_exp.columns and value_col in self.df_exp.columns:
                df_tmp = pd.DataFrame({
                    'Post_proc': self.df_exp[post_proc_col],
                    'Value': self.df_exp[value_col],
                    'Unit': self.df_exp[unit]
                })

                df_post_proc_val = pd.concat([df_post_proc_val, df_tmp], axis=0).reset_index(drop=True)
        df_post_proc_val.dropna(axis= 0, inplace= True)
        df_post_proc_val.reset_index(drop= True, inplace= True)

        df_post_proc_val.to_csv(os.path.join(self.output_dir_val, "post_proc_val_count.csv"), index= False)

    def exe(self):
        print("Analyzing properties from csv files....")
        self.analyze_props_from_csv()
        print("Analyzing synthesis method from csv files....")
        self.get_syns_method()
        print("Analyzing post processing from csv files....")
        self.get_post_proc()
        print("Analyzing val count from csv files....")
        self.val_count()








