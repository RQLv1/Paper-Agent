import requests, time, os
from tqdm import tqdm
import json
from json_repair import repair_json
from pdfminer.high_level import extract_text
from io import BytesIO
import re
from DrissionPage import ChromiumPage, ChromiumOptions
from crossref_commons.retrieval import get_publication_as_json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class Create_db:

    def __init__(self, 
                 spg_api_key, 
                 els_api_key, 
                 wly_api_key, 
                 llms_model, 
                 llms_api_key, 
                 base_url,
                 abs_prompt_file, 
                 exp_prompt_file, 
                 titleclean_prompt_file, 
                 wlyclean_prompt_file,
                 output_path, 
                 agent_model = "ollama_model", 
                 agent_key = "ollama", 
                 agent_url = "http://localhost:6006/v1/", 
                 is_agent = False):

        self.spg_api_key = spg_api_key
        self.els_api_key = els_api_key
        self.wly_api_key = wly_api_key

        self.llms_model = llms_model
        self.llms_api_key = llms_api_key
        self.base_url = base_url

        self.agent_model = agent_model
        self.agent_key = agent_key
        self.agent_url = agent_url
        self.is_agent = is_agent

        with open(abs_prompt_file, 'r', encoding='utf-8') as f:
            abs_prompt = f.read()
        self.abs_prompt = abs_prompt

        with open(exp_prompt_file, 'r', encoding='utf-8') as f:
            exp_prompt = f.read()
        self.exp_prompt = exp_prompt

        with open(titleclean_prompt_file, 'r') as f:
            self.titleclean_prompt = f.read()

        with open(wlyclean_prompt_file, 'r') as f:
            self.wlyclean_prompt = f.read()

        self.output_path = output_path
        print(f"database path set: {self.output_path}")
        self.doi_file = os.path.join(self.output_path, 'doi_output.txt')
        self.doi_output_file = os.path.join(self.output_path, 'dois_clean.txt')
        self.originaltext_output_file = os.path.join(self.output_path, 'originaltext_output.jsonl')
        self.ext_output_file = os.path.join(self.output_path, 'ext_output.jsonl')
        self.clean_title = os.path.join(self.output_path, 'clean_title.jsonl')
        

    def get_dois(self, query):

        api_key = self.els_api_key #scopus api
        url = f"https://api.elsevier.com/content/search/scopus?httpAccept=application/json&query={query}&apiKey={api_key}&count=200"
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}, {response.text}")
        total = int(response.json()['search-results']['opensearch:totalResults'])

        doi_list = []
        for start in tqdm(range(0, total, 200)): # Limit 5000 usage per request
            time.sleep(1) # Avoid hitting rate limits
            try:
                url = f"https://api.elsevier.com/content/search/scopus?httpAccept=application/json&query={query}&apiKey={api_key}&start={start}&count=200"
                response = requests.get(url, verify=False).json()
                entries = response['search-results']['entry']

                for entry in entries:
                    if 'prism:doi' in entry:
                        doi_list.append(entry['prism:doi'])
            except:
                continue

        existing_dois = set()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if os.path.exists(self.doi_file):
            with open(self.doi_file, 'r') as f:
                existing_dois = set(f.read().splitlines())
            unique_dois = list(existing_dois.union(doi_list))
            with open(self.doi_file, 'w') as f:
                f.write('\n'.join(unique_dois))
        else:
            with open(self.doi_file, 'w') as f:
                f.write('\n'.join(doi_list))
        
        return "Dois collected successfully"
    
    def doi_clean(self):
        with open(self.doi_file, "r") as f:
            doi_lst = f.read().split("\n")
        pub_doi = []
        for doi in tqdm(doi_lst):
            if self.journal_publisher(doi) in ['acs', 'rsc', 'wiley', 'springer', 'elsevier']:
                pub_doi.append(doi)
        with open(self.doi_output_file, "w") as f:
            f.write('\n'.join(pub_doi))

    def llms_agent(self, ask):
        client = OpenAI(
            base_url= self.agent_url,
            api_key=self.agent_key,  
        )
        response = client.chat.completions.create(
            model=self.agent_model,
            messages=[
                {"role":"user", "content": ask},
                ])
        answer_content = response.choices[0].message.content
        return answer_content

    def llms_model_use(self, prompt, ask, mode = "simple"):
        client = OpenAI(
            api_key= self.llms_api_key, 
            base_url= self.base_url,
        )
        if mode != "deep thought":
            reasoning_content = "" 
            completion = client.chat.completions.create(
                model= self.llms_model,
                temperature= 0,
                top_p= 0, 
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': ask}],
                )
            answer_content = completion.choices[0].message.content
            return reasoning_content, answer_content

        else:
            reasoning_content = "" 
            answer_content = ""    
            is_answering = False

            completion = client.chat.completions.create(
                    model= self.llms_model,
                    stream= True,
                    messages=[
                        {
                        'role': 'system', 
                        'content': prompt

                         },
                        {
                            'role': 'user', 
                            'content': ask
                         }
                    ]
                )

            for chunk in completion:
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        reasoning_content += delta.reasoning_content
                    else:
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                        answer_content += delta.content
        
        return reasoning_content, answer_content

    @staticmethod
    def json_control(answer_content):
        if "```json" in answer_content:
            answer_content = answer_content.replace("```json", "")
            answer_content = answer_content.replace("```", "")
            
        if "Note" in answer_content:
            answer_content = answer_content.split("Note")[0]

        try:
            answer_content_js = json.loads(answer_content)
        except:
            try:
                answer_content_js = json.loads(repair_json(str(answer_content)))
            except:
                answer_content_js = answer_content
        try:
            for key in answer_content_js.keys():
                if key != 'doi':
                    value = answer_content_js[key]
                    if isinstance(value, str):
                        answer_content_js[key] = json.loads(repair_json(value))
        except:
            answer_content_js = answer_content_js

        return answer_content_js
    
    @staticmethod
    def journal_publisher(doi):

        try:
            publisher = get_publication_as_json(doi)['publisher']
            if 'elsevier' in publisher.lower():
                return 'elsevier'
            elif 'wiley' in publisher.lower():
                    return 'wiley'
            elif 'springer' in publisher.lower():
                    return 'springer'
            elif 'rsc' in publisher.lower():        
                    return 'rsc'
            elif 'informa' in publisher.lower():
                    return 'taylor & francis'
            elif 'iop' in publisher.lower():
                    return 'iop'
            elif 'aaas' in publisher.lower():
                 return 'aaas'
            elif 'acs' in publisher.lower():
                    return 'acs'
            else:
                return 'out of publishers'
        except Exception as e:
            print(e)
            pass

    def llms_clean_title(self, matched_text):
        if self.is_agent:
            ask = "Now execute your field filter function and filter the fields from the following text: " + f'{matched_text}'
            answer_content = self.llms_agent(ask= ask)
            answer_content_js = self.json_control(answer_content)
            reasoning_content = ""
        
        else:
            ask = 'Filter the fields from the following text:' +  f'{matched_text}'
            reasoning_content, answer_content = self.llms_model_use(
                                                                    prompt= self.titleclean_prompt, 
                                                                    ask= ask, 
                                                                    mode= "simple"
                                                                    )
            answer_content_js = self.json_control(answer_content)


        return reasoning_content, answer_content_js

    def extract_sections(self, text):
        pattern = r'(?:1\s+Introduction|INTRODUCTION).*?(?:References|REFERENCES)'
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            matched_text = matches.group(0)
            reasoning_content, sections = self.llms_clean_title(matched_text)

            entry = {
                    "query": "Clean the following string:",
                    "origin": matched_text,
                    "reasoning": reasoning_content, 
                    "ext": sections  
                    }
            
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            with open(self.clean_title, "a") as f:
                json.dump(entry, f)
                f.write('\n')

            return sections
        return None

    @staticmethod
    def extract_experimental_section(text, match_title, match_title_1):
        first_pos = text.find(match_title)
        if first_pos != -1:
            second_pos = text.find(match_title, first_pos + 1)
            if second_pos != -1:
                text_from_second = text[second_pos:]
                end_pos = text_from_second.find(match_title_1)
                if end_pos != -1:
                    return text_from_second[:end_pos].strip()
        return None

    def get_content(self, doi):
        co = ChromiumOptions().set_browser_path('C:\Program Files\Google\Chrome\Application\chrome.exe')
        co = ChromiumOptions() \
            .set_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36") \
            .set_argument("--window-position=-32000,-32000")\
            .set_argument("--window-size=1,1")\
            .set_argument("--force-device-scale-factor=1") 
        page = ChromiumPage(addr_or_opts=co)

        experimental_keywords = [
            'theory',
            'experimental', 'experiment', 'experiments', 'experimental section', 'experimental design', 'experimental Section', 'experimental Design',
            'methods', 'methodology', 'materials and methods', 'materials and Methods', 'material and method',
            'procedure', 'procedures', 
            'synthesis', 'synthesis and characterisation', 'synthesis and Characterisation',
            ]

        if self.journal_publisher(doi) == 'springer':
            
            if self.spg_api_key is not None:

                try:
                    url = f"https://api.springernature.com/meta/v2/json?api_key={self.spg_api_key}&callback=&s=1&p=1&q=(doi: {doi})"
                    response = requests.request("GET", url)
                    abs_content = json.loads(response.text)['records'][0]['abstract']
                    
                except:
                    page.get(f'https://doi.org//{doi}')

                    #abstract
                    try:
                        abs_section = page.ele('css:[data-title="Abstract"]')
                        abs_content = abs_section.text
                    except:
                        abs_content = "can not get content"

                #experiment
                for keyword in experimental_keywords:
                    try:
                        selector = f'css:[data-title*="{keyword.capitalize()}"]'
                        exp_section = page.ele(selector)
                        if exp_section:
                            exp_content = exp_section.text
                            break  
                    except:
                        exp_content = "can not get content"
                        continue

        elif self.journal_publisher(doi) == 'acs':
            page.get(f'https://pubs.acs.org/doi/full/{doi}')

            #abstract
            try:
                abs_section = page.ele('css:.article_abstract')
                abs_content = abs_section.text
            except:
                abs_content = "can not get content"

            #experiment
            exp_content = ""
            try:
                for i in range(1, 6):  
                    exp_section = page(f'css:#sec{i}')
                    if exp_section:
                        exp_title = exp_section('css:h2').text.lower()
                        if any(keyword in exp_title.lower() for keyword in experimental_keywords):
                            exp_content = exp_section.text
                            break
                        
                if not exp_content:
                    exp_content = "can not get content"

            except Exception as e:
                exp_content = "can not get content"

        elif self.journal_publisher(doi) == 'wiley':
            if self.wly_api_key is not None:
                try:
                    headers = {
                        "Accept": "application/pdf",
                        "Wiley-TDM-Client-Token": f"{self.wly_api_key}"
                    }

                    response = requests.get(
                        f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
                        headers=headers,
                        verify=False
                    )

                    if response.status_code == 200:
                        try:
                            text = extract_text(BytesIO(response.content))
                            ask = f'Extract the contents of the abstract and experimental sections: {text}'
                            answer_content = self.llms_model_use(
                                                                 prompt= self.wlyclean_prompt,
                                                                 ask= ask,
                                                                 mode = "batch"
                                                                )
                            answer_content_js = self.json_control(answer_content)
                            
                            if isinstance(answer_content_js, dict):
                                abs_content = answer_content_js.get('abstract', "can not get content")
                                exp_content = answer_content_js.get('experimental', "can not get content")

                        except Exception as e:
                            abs_content = "can not get content"
                            exp_content = "can not get content"
                    
                except:
                    page.get(f'https://onlinelibrary.wiley.com/doi/full/{doi}')

                    #abstract
                    try:
                        abs_section = page.ele('css:.abstract-group')
                        abs_content = abs_section.text
                    except:
                        abs_content = "can not get content"

                    #experiment
                    exp_content = ""
                    try:
                        exp_sections = page.eles('css:.article-section__title.section__title')
                        for exp_section in exp_sections:
                            exp_title = exp_section.text.lower()
                            if any(keyword in exp_title.lower() for keyword in experimental_keywords):
                                exp_content_section = exp_section.next()
                                if exp_content_section:
                                    exp_content = exp_content_section.text
                                break
                    except:
                        exp_content = "can not get content"

        elif self.journal_publisher(doi) == 'elsevier':

            pdf_url = 'https://api.elsevier.com/content/article/doi/' + doi
            headers = {
                'X-ELS-APIKEY': self.els_api_key,
                'Accept': 'application/json'
            }

            try:
                response = requests.get(pdf_url, headers=headers, verify=False)
                if response.status_code == 200:
                    data = response.json()
                    #abstract
                    abs_content = data['full-text-retrieval-response']['coredata']['dc:description']

                    #experimental
                    result = self.extract_sections(data['full-text-retrieval-response']['originalText'])
                    

                    match_title = ''
                    match_title_1 = ''
                    for number, title in result.items():
                        if '.' not in number:
                            if any(keyword in title.lower() for keyword in experimental_keywords):
                                match_title = number+ ' ' + title
                                match_title_1 = str(int(number) + 1)+ ' ' + result[str(int(number) + 1)]
                                break
                            
                    if match_title and match_title_1:
                        exp_content = self.extract_experimental_section(
                            data['full-text-retrieval-response']['originalText'],
                            match_title,
                            match_title_1
                        )

                    if not exp_content:
                        exp_content = "can not get content"
                else:
                    exp_content = "can not get content"
            except Exception as e:
                exp_content = "can not get content"

        elif self.journal_publisher(doi) == 'rsc':
            page.get(f'https://doi.org//{doi}')

            #abstract
            try:
                abs_section = page.ele('css:.capsule__text')
                abs_content = abs_section.text
            except:
                abs_content = "can not get content"

            #experiment
            exp_content = ""
            try:
                headings = page.eles('css:.h--heading2')
                for i, heading in enumerate(headings):
                    if any(keyword in heading.text.lower() for keyword in experimental_keywords):
                        exp_h = heading.nexts()
                        ele_p = []
                        for ele in exp_h:
                            if ele.tag == 'p':
                                ele_p.append(ele)
                            if ele.tag == 'h2':
                                break
                        for exp_section in ele_p:
                            exp_content += exp_section.text + "\n"
                        break
                    
            except Exception as e:
                exp_content = "can not get content"

        page.quit()

        sanitized_doi = doi.replace('/', '_')

        return abs_content, exp_content, sanitized_doi
    
    def get_abs_content(self, abs):
        if self.is_agent:

            ask = "Now execute your abstract extract function and extract the following abstract text: " + f'{abs}'
            answer_content = self.llms_agent(ask= ask)
            answer_content_js = self.json_control(answer_content)
            reasoning_content = ""
            
        else:
            ask = 'Extract the following text:' +  f'{abs}'
            reasoning_content, answer_content = self.llms_model_use(
                                                                    prompt= self.abs_prompt,
                                                                    ask= ask,
                                                                    mode= "simple"
                                                                    )
            answer_content_js = self.json_control(answer_content)

        return reasoning_content, answer_content_js

    def get_exp_content(self, exp):
        if self.is_agent:
            ask = "Now execute your experimental extract function and extract the following experimental section text: " + f'{exp}'
            answer_content = self.llms_agent(ask= ask)
            answer_content_js = self.json_control(answer_content)
            reasoning_content = ""

        else:
            ask = 'Extract the following experimental section text:' +  f'{exp}'
            reasoning_content, answer_content = self.llms_model_use(
                                                                    prompt= self.exp_prompt,
                                                                    ask= ask,
                                                                    mode= "simple"
                                                                    )
            answer_content_js = self.json_control(answer_content)

        return reasoning_content, answer_content_js
    
    def get_originaltext(self):

        #read dois
        DOI_list = []
        with open(self.doi_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\ufeff' in line:
                    doi = line.strip().split('\ufeff')[-1]
                else:
                    doi = line.strip()
                DOI_list.append(doi)
        assert DOI_list != []
        
        for doi in tqdm(DOI_list):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self.get_content, doi)
                try:
                    abs_originaltext, exp_originaltext, sanitized_doi = future.result(timeout=300)

                    entry_originaltext = {
                        'doi': doi,
                        'abstract': abs_originaltext if "can not get content" not in abs_originaltext else None,
                        'experimental': exp_originaltext if "can not get content" not in exp_originaltext else None
                    }

                    # original entry wirte in
                    with open(self.originaltext_output_file, 'a', encoding='utf-8') as f:
                        json.dump(entry_originaltext, f, ensure_ascii=False)
                        f.write('\n')

                except TimeoutError:
                    print(f"TimeoutError: The operation timed out for doi: {doi}")
                    continue
                except Exception as e:
                    continue
        
        return "Originaltext get successfully"
    
    def get_ext(self):
        with open(self.originaltext_output_file, "r", encoding="utf-8") as f:
            origintexts = [json.loads(line) for line in f]

        for original_entry in tqdm(origintexts):
            
            doi = original_entry['doi']
            abs_originaltext = original_entry['abstract']
            exp_originaltext = original_entry['experimental']
            
            # try:
            if abs_originaltext is None or abs_originaltext in ["", "can not get content"]:
                abs_reasoning_content, abs_content = None, None
            else:
                abs_reasoning_content, abs_content = self.get_abs_content(abs_originaltext)
            if exp_originaltext is None or exp_originaltext in ["", "can not get content"]:
                exp_reasoning_content, exp_content = None, None
            else:
                exp_reasoning_content, exp_content = self.get_exp_content(exp_originaltext)
            
            entry_ext = {
                'doi': doi,
                'abstract': abs_content,
                'experimental': exp_content
            }
            # ext entry write in
            with open(self.ext_output_file, 'a', encoding='utf-8') as f:
                json.dump(entry_ext, f, ensure_ascii=False)
                f.write('\n')
            # except:
            #     continue
            
        return "Extract successfully"

    def stream_mode(self):

        #clean dois
        self.doi_clean()
        #read dois
        DOI_list = []
        with open(self.doi_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\ufeff' in line:
                    doi = line.strip().split('\ufeff')[-1]
                else:
                    doi = line.strip()
                DOI_list.append(doi)
        assert DOI_list != []
        
        # stream 
        for doi in tqdm(DOI_list):
            try:
                abs_originaltext, exp_originaltext, sanitized_doi = self.get_content(doi)

                entry_originaltext = {
                    'doi': doi,
                    'abstract': abs_originaltext if "can not get content" not in abs_originaltext else None,
                    'experimental': exp_originaltext if "can not get content" not in exp_originaltext else None
                }

                # original entry wirte in
                with open(self.originaltext_output_file, 'a', encoding='utf-8') as f:
                    json.dump(entry_originaltext, f, ensure_ascii=False)
                    f.write('\n')

                if abs_originaltext is None or abs_originaltext in ["can not get content"]:
                    abs_reasoning_content, abs_content = None, None
                else:
                    abs_reasoning_content, abs_content = self.get_abs_content(abs_originaltext)


                if exp_originaltext is None or exp_originaltext in ["can not get content"]:
                    exp_reasoning_content, exp_content = None, None
                else:
                    exp_reasoning_content, exp_content = self.get_exp_content(exp_originaltext)

                entry_ext = {
                    'doi': doi,
                    'abstract': abs_content,
                    'experimental': exp_content
                }
                # ext entry write in
                with open(self.ext_output_file, 'a', encoding='utf-8') as f:
                    json.dump(entry_ext, f, ensure_ascii=False)
                    f.write('\n')
            except Exception as e:
                continue
        
        return "Database create successfully"