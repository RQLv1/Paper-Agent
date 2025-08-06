import json, os
from openai import OpenAI
from quantulum3 import parser
from pint import UnitRegistry
from tqdm import tqdm
from json_repair import repair_json

class Clean_db:

    def __init__(self, llms_model, llms_api_key, base_url, cleanedprop_prompt_file, cleanedoper_prompt_file, output_path):
        self.llms_model = llms_model
        self.llms_api_key = llms_api_key
        self.base_url = base_url

        with open(cleanedprop_prompt_file, 'r', encoding='utf-8') as f:
            cleanedprop_prompt = f.read()
        self.cleanedprop_prompt = cleanedprop_prompt

        with open(cleanedoper_prompt_file, 'r', encoding='utf-8') as f:
            cleanedoper_prompt = f.read()
        self.cleanedoper_prompt = cleanedoper_prompt

        self.output_path = output_path
        self.ext_input_file = os.path.join(self.output_path, 'ext_output.jsonl')
        self.clean_ouput_file = os.path.join(self.output_path, 'dataset.jsonl')

        self.clean_str = ["", {}, None]
        
    def llms_model_use(self, prompt, ask):
        client = OpenAI(
            api_key= self.llms_api_key, 
            base_url=self.base_url,
        )
        completion = client.chat.completions.create(
            model= self.llms_model,
            temperature= 0,
            top_p= 0, 
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': ask}],
            )
        answer_content = completion.choices[0].message.content

        return answer_content
    @staticmethod
    def json_control(answer_content):
        if "```json" in answer_content:
            answer_content = answer_content.replace("```json", "")
            answer_content = answer_content.replace("```", "")

        try:
            answer_content_js = json.loads(answer_content)
        except:
            try:
                answer_content_js = repair_json(answer_content)
            except:
                answer_content_js = answer_content

        return answer_content_js
    
    def clean_nan(self):
        if not os.path.exists(self.ext_input_file):
            return print("The file does not exist.")
        with open(self.ext_input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        dropna_data = []
        for item in data:
            abs = item['abstract']
            exp = item['experimental']

            if abs or exp:
                dropna_data.append(item)
        return dropna_data
    
    def clean_abs_prop(self, abs_prop_data):
        props = set()
        for item in abs_prop_data:
            try:
                if item['abstract'] not in self.clean_str:
                    for mat in item['abstract']:
                        property = item['abstract'][mat]['props'].keys()
                        for prop in [prop for prop in property]:
                            props.add(prop)
            except:
                continue
        
        ask = "Please help me to categorize the following material properties:" + f"{props}"
        print(f"Call LLM for field categorization...")
        answer_content = self.llms_model_use(prompt=self.cleanedprop_prompt, ask=ask)
        cleanedprop_str = self.json_control(answer_content)

        print("Field cleaning...")
        for item in tqdm(abs_prop_data):
            try:
                if item['abstract'] not in self.clean_str:
                    for mat in item['abstract']:
                        properties = item['abstract'][mat]['props'].copy()
                        for prop in properties:
                            for citem in cleanedprop_str:
                                if prop in cleanedprop_str[citem]:
                                    item['abstract'][mat]['props'][citem] = item['abstract'][mat]['props'].pop(prop)
            except:
                continue
        return abs_prop_data
    
    @staticmethod
    def transform_units(input_string):
        ureg = UnitRegistry()
        entry = []
        text_parse = input_string
        quants = parser.parse(text_parse)

        if quants:
            quant = quants[0]
            original_value = quant.value
            original_unit = quant.unit.name

            try:
                quantity = original_value * ureg(original_unit)

                converted_quantity = quantity.to_base_units()

                standard_value = converted_quantity.magnitude
                standard_unit = str(converted_quantity.units)
            except:
                standard_value = original_value
                standard_unit = original_unit

            entry = [
                {"surface": text_parse},
                {"value": standard_value},
                {"unit": standard_unit}
            ]
        return entry
    
    def clean_abs_unit(self, unit_data):

        print("Perform unit cleaning...")
        for item in tqdm(unit_data):
            try:
                if item['abstract'] not in self.clean_str:
                    for mat in item['abstract']:
                        properties = item['abstract'][mat]['props']
                        for prop in properties:
                            text_parse = properties[prop]
                            entry = self.transform_units(text_parse)
                            properties[prop] = json.dumps(entry)
            except:
                continue
        return unit_data

    def clean_exp_oper(self, exp_oper_data):
        filtered_exp_data_oper = []

        for item in exp_oper_data:
            try:
                if item['experimental'] not in self.clean_str:
                    if isinstance(item['experimental'], dict):
                        filtered_exp_data_oper.append(item)
                    else:
                        continue
            except:
                continue
            
        exp_data_oper = filtered_exp_data_oper

        operate_set = set()
        for item in exp_data_oper:
            try:
                if item['experimental'] not in self.clean_str:
                    for matidx in item['experimental']:
                        sysn_process = item['experimental'][matidx]['Syns_processing']
                        post_processing = sysn_process.get("post_processing", {})

                        if post_processing:
                            for proc in post_processing.values():
                                if "step" in proc:
                                    operate_set.add(proc['step'])
            except:
                continue

        ask = "Please help me to categorize the following operation:" + f"{operate_set}"
        print(f"Call LLM for operation categorization...")
        answer_content = self.llms_model_use(prompt=self.cleanedoper_prompt, ask=ask)
        cleanedoper_str = self.json_control(answer_content)

        print("Operation cleaning...")
        for item in tqdm(exp_data_oper):
            try:
                if item['experimental'] not in self.clean_str:
                    for matidx in item['experimental']:
                        sysn_process = item['experimental'][matidx]['Syns_processing']
                        post_processing = sysn_process.get("post_processing", {})
                        if post_processing:
                            for proc in post_processing.values():
                                for oitem in cleanedoper_str:
                                    if "step" in proc.keys():
                                        if proc['step'] in cleanedoper_str[oitem]:
                                            proc['step'] = oitem
            except:
                continue

        return exp_data_oper

    def clean_exp_unit(self, exp_unit_data):
        
        print("Perform unit cleaning...")
        for item in tqdm(exp_unit_data):
            try:
                if item['experimental'] not in self.clean_str:
                    for matidx in item['experimental']:
                        sysn_process = item['experimental'][matidx]['Syns_processing']

                        for key in sysn_process.keys(): #precursor, solvent, post_processing
                            for key2 in sysn_process[key].keys():
                            
                                if isinstance(sysn_process[key][key2], dict):
                                    for key3 in sysn_process[key][key2].keys():
                                    
                                        if "amount" in key3:
                                            entry_amt1 = self.transform_units(sysn_process[key][key2][key3])
                                            sysn_process[key][key2][key3] = entry_amt1

                                        if "parameter" in key3:
                                            for key4_p in sysn_process[key][key2][key3].keys():
                                                if "temp"  in key4_p or "time" in key4_p:
                                                    entry_tmp_tme = self.transform_units(sysn_process[key][key2][key3][key4_p])
                                                    sysn_process[key][key2][key3][key4_p] = entry_tmp_tme

                                        if "substance" in key3:
                                            for iter in sysn_process[key][key2][key3]:
                                                for key4_sub in iter.keys():
                                                    if "amount" in key4_sub:
                                                        entry_sub_amt = self.transform_units(iter[key4_sub])
                                                        iter[key4_sub] = entry_sub_amt
            except:
                continue
                                                        
        return exp_unit_data

    def clean_db(self):
        print("cleaning nan value...")
        dropna_data = self.clean_nan()
        print("cleaning abstract value...")
        abs_prop_data = self.clean_abs_prop(dropna_data)
        abs_unit_data = self.clean_abs_unit(abs_prop_data)
        print("cleaning experimental value...")
        exp_oper_data = self.clean_exp_oper(abs_unit_data)
        exp_unit_data = self.clean_exp_unit(exp_oper_data)

        with open(self.clean_ouput_file, 'w') as f:
            for item in exp_unit_data:
                json.dump(item, f)
                f.write('\n')

        return print("Database clean successfully")