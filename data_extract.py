import json
import re
import csv
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class DataExtractor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.output_abs_path = os.path.join(output_path, "output_abs.csv")
        self.output_abs_clean_path = os.path.join(output_path, "output_abs_clean.csv")
        self.output_exp_path = os.path.join(output_path, "output_exp.csv")
        self.output_exp_clean_path = os.path.join(output_path, "output_exp_clean.csv")

    @staticmethod
    def safe_get(data, keys, default=None):
        for key in keys:
            try:
                data = data[key]
            except (KeyError, TypeError, IndexError):
                return default
        return data

    def convert_abs_jsonl_to_csv(self):
        columns = set()
        all_rows = []

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_number} due to JSONDecodeError: {e}")
                    continue
                
                doi_value = data.get('doi', '')
                abstract_data = data.get('abstract', {})
                if not isinstance(abstract_data, dict):
                    abstract_data = {}

                for mat_key in abstract_data:
                    mat = abstract_data[mat_key]
                    if not isinstance(mat, dict):
                        continue

                    row = {
                        'doi': doi_value,
                        'mat': self.safe_get(mat, ['name']),
                        'application': self.safe_get(mat, ['application'])
                    }

                    props = self.safe_get(mat, ['props'], {})
                    if not isinstance(props, dict):
                        props = {}

                    for idx, (prop_name, prop_value_input) in enumerate(props.items(), 1):
                        

                        parsed_prop_content = None 

                        if isinstance(prop_value_input, str):
                            try:
                                
                                parsed_prop_content = json.loads(prop_value_input.replace("'", '"'))
                            except json.JSONDecodeError:

                                parsed_prop_content = None 
                        elif isinstance(prop_value_input, (int, float, list, dict, bool)) or prop_value_input is None:
                            
                            parsed_prop_content = prop_value_input

                        value = None
                        unit = None

                        if isinstance(parsed_prop_content, list):
                            for item_in_list in parsed_prop_content:
                                if isinstance(item_in_list, dict):
                                    if 'value' in item_in_list:
                                        value = item_in_list['value']
                                    if 'unit' in item_in_list:
                                        unit = item_in_list['unit']
                        elif isinstance(parsed_prop_content, dict): 
                            if 'value' in parsed_prop_content:
                                value = parsed_prop_content['value']
                            if 'unit' in parsed_prop_content:
                                unit = parsed_prop_content['unit']
                        elif isinstance(parsed_prop_content, (int, float)): 
                            value = parsed_prop_content
                            # unit remains None

                        row.update({
                            f'props_{idx}': prop_name,
                            f'props_{idx}_value': value,
                            f'props_{idx}_unit': unit
                        })

                    columns.update(row.keys())
                    all_rows.append(row)

        sorted_columns = []
        if 'mat' in columns: sorted_columns.append('mat')
        if 'application' in columns: sorted_columns.append('application')

        props_column_parts = []
        other_columns = []

        for col in columns:
            if col not in ['mat', 'application']:
                if col.startswith('props_'):
                    parts = col.split('_')
                    try:
                        num = int(parts[1])
                        suffix_order = 0
                        if len(parts) > 2:
                            if parts[2] == 'value':
                                suffix_order = 1
                            elif parts[2] == 'unit':
                                suffix_order = 2
                        props_column_parts.append((num, suffix_order, col))
                    except (ValueError, IndexError):
                        other_columns.append(col)
                else:
                    other_columns.append(col)

        props_column_parts.sort()
        sorted_columns.extend([col_name for _, _, col_name in props_column_parts])
        sorted_columns.extend(sorted(other_columns))

        final_fieldnames = sorted_columns[:]
        for col in columns:
            if col not in final_fieldnames:
                final_fieldnames.append(col)

        final_sorted_columns = ['doi', 'mat', 'application']
        temp_props_columns = []
        remaining_columns = []
        for col in columns:
            if col not in final_sorted_columns:
                if col.startswith('props_'):
                    parts = col.split('_')
                    num = int(parts[1])
                    
                    order_in_group = 0
                    if len(parts) > 2 and parts[2] == 'value':
                        order_in_group = 1
                    elif len(parts) > 2 and parts[2] == 'unit':
                        order_in_group = 2
                    temp_props_columns.append((num, order_in_group, col))
                else:
                    remaining_columns.append(col)

        temp_props_columns.sort()
        final_sorted_columns.extend([col for _, _, col in temp_props_columns])
        final_sorted_columns.extend(sorted(remaining_columns))


        with open(self.output_abs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_sorted_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)

    def convert_exp_jsonl_to_csv(self):
        columns = set()
        all_rows = []

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_number} in experimental due to JSONDecodeError: {e}")
                    continue
                
                doi_value = data.get('doi', '')
                experimental_data = data.get('experimental')
                if not isinstance(experimental_data, dict): # Ensure experimental_data is a dictionary
                    continue

                for mat_key in experimental_data:
                    mat = experimental_data[mat_key]
                    if not isinstance(mat, dict): # Ensure mat is a dictionary
                        continue

                    row = {
                        'doi': doi_value,
                        'mat': self.safe_get(mat, ['name']),
                        'syns_method': self.safe_get(mat, ['Syns_method'])
                    }

                    precursors = self.safe_get(mat, ['Syns_processing','precursors'], {})
                    if isinstance(precursors, dict):
                        for i, prc_data in enumerate(precursors.values(), 1):
                            if isinstance(prc_data, dict): # ensure prc_data is a dict
                                row.update({
                                    f'prc{i}_name': self.safe_get(prc_data, ['name'], ''),
                                    f'prc{i}_value': self.safe_get(prc_data, ['amount', 1, 'value'], ''), # Original: safe_get(prc, ['amount', 1, 'value'], '')
                                    f'prc{i}_unit': self.safe_get(prc_data, ['amount', 2, 'unit'], '')   # Original: safe_get(prc, ['amount', 2, 'unit'], '')
                                })

                    post_processing_steps = self.safe_get(mat, ['Syns_processing', 'post_processing'], {})
                    if isinstance(post_processing_steps, dict):
                        for j, proc_step_data in enumerate(post_processing_steps.values(), 1):
                            if not isinstance(proc_step_data, dict): # ensure proc_step_data is a dict
                                continue
                            try:
                                row[f'post_proc{j}_step'] = proc_step_data.get('step', '') # Use .get for safety

                                parameters = proc_step_data.get('parameters', {})
                                if not isinstance(parameters, dict): # Ensure parameters is a dict
                                    parameters = {}

                                param_idx = 0
                                for param_name, param_details in parameters.items():
                                    param_idx +=1 # To make unique keys if multiple params are stored
                                    value = ''
                                    unit = ''

                                    if isinstance(param_details, list):
                                        if len(param_details) >= 1 and isinstance(param_details[0], dict) and 'value' in param_details[0]: # Common [{"value":X, "unit":Y}]
                                             value = param_details[0].get('value', '')
                                             if len(param_details) >= 1 and isinstance(param_details[0], dict) and 'unit' in param_details[0]:
                                                 unit = param_details[0].get('unit', '')
                                        elif len(param_details) >= 2 and isinstance(param_details[1], dict): # Original logic style
                                            value = param_details[1].get('value', '') if isinstance(param_details[1], dict) else ''
                                            if len(param_details) >= 3 and isinstance(param_details[2], dict):
                                                unit = param_details[2].get('unit', '')
                                    elif isinstance(param_details, dict):
                                        value = param_details.get('value', '')
                                        unit = param_details.get('unit', '')
                                    elif isinstance(param_details, (str, int, float)): # If param_details is just a value
                                        value = param_details

                                    if param_idx == 1: # Take first parameter's value/unit
                                        row.update({
                                            f'post_proc{j}_value': value, # Or just f'post_proc{j}_value': value
                                            f'post_proc{j}_unit': unit   # Or just f'post_proc{j}_unit': unit
                                        })
                                        # For the original column naming:
                                        row[f'post_proc{j}_value'] = value
                                        row[f'post_proc{j}_unit'] = unit
                                        break # Process only the first parameter for the step's main value/unit

                            except Exception as e: # Broader exception catch for safety
                                print(f"Error processing post_proc step {j} for mat {mat_key} on line {line_number}: {e}")
                                continue
                            
                    columns.update(row.keys())
                    all_rows.append(row)

        final_sorted_columns_exp = []
        if 'doi' in columns: final_sorted_columns_exp.append('doi')
        if 'mat' in columns: final_sorted_columns_exp.append('mat')
        if 'syns_method' in columns: final_sorted_columns_exp.append('syns_method')

        prc_cols = []
        post_proc_cols = []
        other_cols_exp = []

        for col in columns:
            if col not in final_sorted_columns_exp:
                if col.startswith('prc'):
                    try:
                        num = int(col.split('_')[0][3:]) # e.g. prc1 -> 1
                        suffix = col.split('_')[1] if '_' in col else '' # name, value, unit
                        order = {'name': 0, 'value': 1, 'unit': 2}.get(suffix, 3)
                        prc_cols.append((num, order, col))
                    except: other_cols_exp.append(col)
                elif col.startswith('post_proc'):
                    try:
                        parts = col.split('_')
                        num = int(parts[1][4:])

                        suffix_group = 0
                        if parts[-1] == 'step': suffix_group = 0
                        elif parts[-1] == 'value': suffix_group = 2
                        elif parts[-1] == 'unit': suffix_group = 3
                        else: suffix_group = 1

                        post_proc_cols.append((num, "_".join(parts[2:]), col))
                    except: other_cols_exp.append(col)
                else:
                    other_cols_exp.append(col)

        prc_cols.sort()
        post_proc_cols.sort()

        final_sorted_columns_exp.extend([c for _, _, c in prc_cols])
        final_sorted_columns_exp.extend([c for _, _, c in post_proc_cols])
        final_sorted_columns_exp.extend(sorted(other_cols_exp))


        with open(self.output_exp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_sorted_columns_exp, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)
    
    @staticmethod
    def is_empty(value):
        return pd.isna(value) or str(value).strip() == ''

    def clean_dataframe_redundancy(self, df, prefix, value_suffix, other_suffixes):

        max_idx = 0
        found_indices = set()
        column_pattern = re.compile(f"^{re.escape(prefix)}(\\d+){re.escape(value_suffix)}$")

        for col in df.columns:
            match = column_pattern.match(col)
            if match:
                found_indices.add(int(match.group(1))) 

        if not found_indices:
            return df 

        max_idx = max(found_indices)

        source_columns_marked_for_removal = set()

        for row_index in df.index:
            for i in range(1, max_idx + 1): 
                target_value_col = f"{prefix}{i}{value_suffix}"

                if target_value_col not in df.columns:
                    continue

                if self.is_empty(df.loc[row_index, target_value_col]):
                    for j in range(i + 1, max_idx + 1):
                        source_value_col = f"{prefix}{j}{value_suffix}"

                        if source_value_col not in df.columns:
                            continue
                        
                        if not self.is_empty(df.loc[row_index, source_value_col]):
                            df.loc[row_index, target_value_col] = df.loc[row_index, source_value_col]
                            source_columns_marked_for_removal.add(source_value_col)

                            for suffix in other_suffixes:
                                target_other_col = f"{prefix}{i}{suffix}"
                                source_other_col = f"{prefix}{j}{suffix}"

                                if source_other_col in df.columns and target_other_col in df.columns:
                                    df.loc[row_index, target_other_col] = df.loc[row_index, source_other_col]
                                    source_columns_marked_for_removal.add(source_other_col)
                                    df.loc[row_index, source_other_col] = pd.NA
                                elif source_other_col in df.columns: 
                                    source_columns_marked_for_removal.add(source_other_col)
                                    df.loc[row_index, source_other_col] = pd.NA

                            df.loc[row_index, source_value_col] = pd.NA
                            break 
                        
        actual_cols_to_drop = [col for col in source_columns_marked_for_removal if col in df.columns]
        if actual_cols_to_drop:
            df = df.drop(columns=list(set(actual_cols_to_drop)), errors='ignore')
        else:
            pass

        return df

    def reindex_column_groups(self, df, prefix, all_suffixes: list):

        current_indices = set()
        for col_name in df.columns:
            if col_name.startswith(prefix):
                potential_num_suffix_part = col_name[len(prefix):]

                num_str = ""
                for char_idx, char_val in enumerate(potential_num_suffix_part):
                    if char_val.isdigit():
                        num_str += char_val
                    else:
                        extracted_suffix = potential_num_suffix_part[char_idx:]
                        if num_str and extracted_suffix in all_suffixes:
                            current_indices.add(int(num_str))
                        break 
                if num_str and potential_num_suffix_part[len(num_str):] in all_suffixes:
                     current_indices.add(int(num_str))


        if not current_indices:
            return df

        sorted_indices = sorted(list(current_indices))

        is_already_sequential = True
        if not sorted_indices:
            is_already_sequential = True
        else:
            for i, idx in enumerate(sorted_indices):
                if idx != i + 1:
                    is_already_sequential = False
                    break
                
        if is_already_sequential:
            return df

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices, 1)}

        columns_rename_map = {}
        for old_idx, new_idx in index_mapping.items():
            if old_idx == new_idx:
                continue
            for suffix in all_suffixes:
                old_col_name = f"{prefix}{old_idx}{suffix}"
                new_col_name = f"{prefix}{new_idx}{suffix}"
                if old_col_name in df.columns:
                    columns_rename_map[old_col_name] = new_col_name

        if not columns_rename_map:
            pass
        else:
            df = df.rename(columns=columns_rename_map)

        return df

    def clean_exp_redundancy(self, df_cleaned):
        prc_value_suffix = "_value"
        prc_other_suffixes = ["_name", "_unit"]
        prc_all_suffixes = prc_other_suffixes + [prc_value_suffix]

        df_cleaned = self.clean_dataframe_redundancy(df_cleaned, 
                                                prefix="prc", 
                                                value_suffix=prc_value_suffix, 
                                                other_suffixes=prc_other_suffixes)

        df_cleaned = self.reindex_column_groups(df_cleaned,
                                           prefix="prc",
                                           all_suffixes=list(set(prc_all_suffixes)))

        post_proc_value_suffix = "_value"
        post_proc_other_suffixes = ["_step", "_unit"]
        post_proc_all_suffixes = post_proc_other_suffixes + [post_proc_value_suffix]
        
        df_cleaned = self.clean_dataframe_redundancy(df_cleaned, 
                                                prefix="post_proc", 
                                                value_suffix=post_proc_value_suffix, 
                                                other_suffixes=post_proc_other_suffixes)

        df_cleaned = self.reindex_column_groups(df_cleaned,
                                           prefix="post_proc",
                                           all_suffixes=list(set(post_proc_all_suffixes)))
        return df_cleaned

    def clean_dataframe_redundancy_abs(self, df, prefix, value_suffix, other_suffixes):

        max_idx = 0
        found_indices = set()
    
        column_pattern = re.compile(f"^{re.escape(prefix)}_(\\d+){re.escape(value_suffix)}$")

        for col in df.columns:
            match = column_pattern.match(col)
            if match:
                found_indices.add(int(match.group(1)))

        if not found_indices:
            return df

        max_idx = max(found_indices)

        source_columns_marked_for_removal = set()

        for row_index in df.index:
            for i in range(1, max_idx + 1):
                target_value_col = f"{prefix}_{i}{value_suffix}"

                if target_value_col not in df.columns:
                    continue

                if self.is_empty(df.loc[row_index, target_value_col]):
                    for j in range(i + 1, max_idx + 1):
                        source_value_col = f"{prefix}_{j}{value_suffix}"

                        if source_value_col not in df.columns:
                            continue

                        if not self.is_empty(df.loc[row_index, source_value_col]):
                            df.loc[row_index, target_value_col] = df.loc[row_index, source_value_col]
                            source_columns_marked_for_removal.add(source_value_col)

                            for suffix in other_suffixes:
                                if suffix == "":
                                    target_other_col = f"{prefix}_{i}"
                                    source_other_col = f"{prefix}_{j}"
                                else:
                                    target_other_col = f"{prefix}_{i}{suffix}" 
                                    source_other_col = f"{prefix}_{j}{suffix}" 

                                if source_other_col in df.columns and target_other_col in df.columns:
                                    df.loc[row_index, target_other_col] = df.loc[row_index, source_other_col]
                                    source_columns_marked_for_removal.add(source_other_col)
                                    df.loc[row_index, source_other_col] = pd.NA
                                elif source_other_col in df.columns: 
                                    source_columns_marked_for_removal.add(source_other_col)
                                    df.loc[row_index, source_other_col] = pd.NA

                            df.loc[row_index, source_value_col] = pd.NA
                            break 

        actual_cols_to_drop = [col for col in source_columns_marked_for_removal if col in df.columns]
        if actual_cols_to_drop:

            df = df.drop(columns=list(set(actual_cols_to_drop)), errors='ignore')
        else:
            pass
        return df

    def reindex_column_groups_abs(self, df, prefix, all_suffixes: list):


        current_indices = set()
        expected_prefix_with_underscore = prefix + "_"
        for col_name in df.columns:
            if col_name.startswith(expected_prefix_with_underscore):
                potential_num_suffix_part = col_name[len(expected_prefix_with_underscore):]

                num_str = ""
                for char_idx, char_val in enumerate(potential_num_suffix_part):
                    if char_val.isdigit():
                        num_str += char_val
                    else:
                        extracted_suffix = potential_num_suffix_part[char_idx:]
                        if num_str and extracted_suffix in all_suffixes:
                            current_indices.add(int(num_str))
                        break
                else:
                    if num_str and potential_num_suffix_part[len(num_str):] in all_suffixes:
                        current_indices.add(int(num_str))

        if not current_indices:
            return df

        sorted_indices = sorted(list(current_indices))

        is_already_sequential = all(idx == i + 1 for i, idx in enumerate(sorted_indices))

        if is_already_sequential:
            return df

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices, 1)}

        columns_rename_map = {}
        for old_idx, new_idx in index_mapping.items():
            if old_idx == new_idx:
                continue
            for suffix in all_suffixes:
                if suffix == "":
                    old_col_name = f"{prefix}_{old_idx}"
                    new_col_name = f"{prefix}_{new_idx}"
                else:
                    old_col_name = f"{prefix}_{old_idx}{suffix}"
                    new_col_name = f"{prefix}_{new_idx}{suffix}"

                if old_col_name in df.columns:
                    columns_rename_map[old_col_name] = new_col_name

        if not columns_rename_map:
            pass
        else:
            df = df.rename(columns=columns_rename_map)

        return df

    def clean_abs_redundancy(self, df_props_cleaned):
        props_prefix = "props"
        props_value_suffix = "_value"
        props_other_suffixes = ["", "_unit"] 
        props_all_suffixes = ["", props_value_suffix, "_unit"]

        df_props_cleaned = self.clean_dataframe_redundancy_abs(df_props_cleaned, 
                                                      prefix=props_prefix, 
                                                      value_suffix=props_value_suffix, 
                                                      other_suffixes=props_other_suffixes)

        df_props_cleaned = self.reindex_column_groups_abs(df_props_cleaned,
                                                 prefix=props_prefix,
                                                 all_suffixes=list(set(props_all_suffixes)))
        return df_props_cleaned

    def exe(self):
        #abs
        print("Converting abstact jsonl to csv......")
        self.convert_abs_jsonl_to_csv()
        df_abs_raw = pd.read_csv(self.output_abs_path, low_memory= False)

        df_abs_raw.drop_duplicates(inplace=True)
        df_abs_raw.dropna(axis=1, how='all', inplace=True)
        df_abs_raw.reset_index(drop=True, inplace=True)

        df_abs_raw = self.clean_abs_redundancy(df_abs_raw.copy())
        df_abs_raw.to_csv(self.output_abs_clean_path, index=False)

        #exp
        print("Converting experiment jsonl to csv......")
        self.convert_exp_jsonl_to_csv()
        df_exp_raw = pd.read_csv(self.output_exp_path, low_memory= False)

        df_exp_raw.drop_duplicates(inplace=True)
        df_exp_raw.dropna(axis=1, how='all', inplace=True)
        df_exp_raw.reset_index(drop=True, inplace=True)

        df_exp_raw = self.clean_exp_redundancy(df_exp_raw.copy())
        df_exp_raw.to_csv(self.output_exp_clean_path, index=False)


