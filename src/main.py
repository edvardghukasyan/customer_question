import asyncio
import json
import logging
import pandas as pd
from src.dataloader.csv_loader import load_csv
from src.experts.classifier import ClassifierExpert
from configs.config import CSV_FILE_PATH, JSON_OUTPUT_PATH, CSV_OUTPUT_PATH, BATCH_SIZE, SAMPLE_SIZE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def main():
    df = load_csv(CSV_FILE_PATH)
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    customer_questions = df_sample.instruction.tolist()
    
    classifier = ClassifierExpert()
    all_results = []
    total = len(customer_questions)
    
    for i in range(0, total, BATCH_SIZE):
        batch = customer_questions[i: i + BATCH_SIZE]
        batch_results = await classifier.process_batch(batch)
        
        for question, result in batch_results:
            if result is None:
                continue
            cluster_name = result.get("name")
            cluster_description = result.get("description")
            if not cluster_name:
                logging.warning("No cluster name found for question: %s", question)
                continue
            
            all_results.append(
                {
                    "question": question,
                    "cluster_name": cluster_name,
                    "description": cluster_description
                }
            )
        
        with open(JSON_OUTPUT_PATH, "w") as f:
            json.dump(all_results, f, indent=4)
        pd.DataFrame(all_results).to_csv(CSV_OUTPUT_PATH, index=False)
        logging.info(
            "Saved cumulative results after processing %d out of %d questions.", min(i + BATCH_SIZE, total), total
        )
        
        await asyncio.sleep(1)
    
    logging.info("Final cumulative results saved.")


if __name__ == "__main__":
    asyncio.run(main())
