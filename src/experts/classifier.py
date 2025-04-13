import asyncio
import json
import pandas as pd
from src.clients.openai import OpenAIClient
from src.experts.base import BaseExpert
from src.utils.utils import extract_json_from_llm_response
from src.prompts.prompts import Prompts
from src.models.errors import ClientError
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from configs.config import MODEL_NAME, TEMPERATURE
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClassifierExpert(BaseExpert):
    
    def __init__(self):
        self.client = OpenAIClient()
        self.llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
        
        self.predefined_cluster_names = set()
        self.clusters_info = {}
        
        self.prompt_template = Prompts.classifier_prompt
    
    def create_chain(self, template: str, input_variables: list):
        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=template
        )
        return prompt_template | self.llm
    
    def invoke(self, chain, inputs: dict):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            raise ClientError(f"Error invoking OpenAI client: {e}")
    
    async def process_question(self, question: str):
        chain = self.create_chain(self.prompt_template, ["question", "clusters"])
        response = await asyncio.to_thread(
            self.invoke, chain, {"question": question, "clusters": list(self.predefined_cluster_names)}
        )
        try:
            result = extract_json_from_llm_response(response.content)
            return question, result
        except Exception as e:
            print(f"Error extracting JSON for question: {question}\nError: {e}")
            print(f"Raw response: {response.content}")
            return question, None
    
    async def process_batch(self, questions: list):
        tasks = [asyncio.create_task(self.process_question(q)) for q in questions]
        return await asyncio.gather(*tasks)
    
    async def run(self, customer_questions: list, batch_size: int = 5):
        
        all_results = []
        total = len(customer_questions)
        progress_bar = tqdm(total=total, desc="Processing 533 Datapoints")
        
        for i in range(0, total, batch_size):
            batch = customer_questions[i: i + batch_size]
            batch_results = await self.process_batch(batch)
            batch_results_list = []
            for question, result in batch_results:
                if result is None:
                    continue
                cluster_name = result.get("name")
                cluster_description = result.get("description")
                if not cluster_name:
                    print(f"No cluster name found for question: {question}")
                    continue
                
                if cluster_name not in self.clusters_info:
                    self.clusters_info[cluster_name] = {"description": cluster_description}
                    self.predefined_cluster_names.add(cluster_name)
                    print(f"New cluster '{cluster_name}' added.")
                else:
                    print(f"Cluster '{cluster_name}' already exists.")
                
                res_dict = {
                    "question": question,
                    "cluster_name": cluster_name,
                    "description": cluster_description
                }
                all_results.append(res_dict)
                batch_results_list.append(res_dict)
                progress_bar.update(1)
                print(f"Processed question: {question}")
            
            batch_number = i // batch_size + 1
            intermediate_json_filename = f"intermediate_results_batch_{batch_number}.json"
            intermediate_csv_filename = f"intermediate_results_batch_{batch_number}.csv"
            with open(intermediate_json_filename, "w") as f:
                json.dump(batch_results_list, f, indent=4)
            pd.DataFrame(batch_results_list).to_csv(intermediate_csv_filename, index=False)
            print(f"Saved intermediate batch {batch_number} to files.")
            
            await asyncio.sleep(1)
        progress_bar.close()
        return all_results
