# Customer Questions Classifier

This repository contains code for classifying customer questions using language models (LLMs) via LangChain. Our solution is designed to work with the [Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset/tree/main) (downloaded as `data.csv`) and predicts clusters of questions.

## Key Components

### Prediction Pipeline:
- We use LangChain and OpenAI's GPT-4o model (with the option to add other models/clients).
- A prompt is designed to classify customer questions based on a predefined set of predicted clusters.
- A singleton pattern is used to manage the connection between our code and the LLM client (ensuring reusability without repeated initializations).

### Evaluation & Mapping:
- Since we do not use true labels for our predictions, we map the predicted clusters to the dominant true label extracted from the real dataset using a combination of the Hungarian assignment mapping and majority voting.
- Our evaluation uses overall accuracy as a metric, achieving an accuracy of 93.53% on a sample.

### Asynchronous Processing:
- OpenAI's standard client cannot send too many requests concurrently. Therefore, we process requests asynchronously in batches.
- We found that a batch size of 5 is the maximum allowable without encountering API errors. Larger batch sizes resulted in rate limiting errors and connection failures.
- Processing the full dataset with a larger batch size would not only trigger API errors but would also significantly increase API costs. 
- The batch size of 5 represents an optimal balance between processing efficiency and API reliability.
- Due to the large overall dataset size, we sample 533 datapoints to perform our evaluation. Based on statistical confidence, this sample size provides a 99% confidence level that our evaluation metrics will reflect the performance on the entire dataset.

## Output Format

The aggregated results (written to question_results.csv and question_results.json) have the following JSON format:
```json
{
  "name": "Technical Issues",
  "description": "Questions about bugs, errors, or problems using the product features.",
  "count": 31
}
```
- **name**: Corresponds to the predicted cluster label.
- **description**: The description provided from the prediction (unchanged).
- **count**: The number of questions that fall into the cluster.

## Technologies & Libraries

### Language & Frameworks:
- Python

### LLM & LangChain:
- LangChain is used to construct prompt templates and chain the model calls to OpenAI's GPT-4o.
- Our design enables easy extension to other models and clients.

### Asynchronous Processing:
- Asynchronous calls are used (with batch size = 5) to ensure efficient processing when making multiple concurrent API requests.
- Our testing showed that larger batch sizes consistently failed with API rate limit errors.
- This batch size optimization was crucial for balancing processing speed with API reliability and cost management.

### Mapping & Evaluation:
- We use Hungarian assignment mapping along with majority vote mapping to optimally align predicted clusters to true labels.
- Evaluation is done using overall accuracy (achieving 93.53% accuracy on our sample).

### Singleton Pattern:
- Our code uses a singleton class to manage connections to our LLM client, ensuring that each client connection is initialized only once.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/edvardghukasyan/customer_questions_classifier.git
cd customer_questions_classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your environment variables:
Create a file (or add to your shell profile) to set your API key securely. For example, in your terminal:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

## Usage

### Running Evaluation

1. Download the data:
   - Ensure data.csv (from bitext/customer-support-llm-chatbot-training-dataset) is in the project root.

2. Run the evaluation script:
   - This will process the data, perform Hungarian assignment mapping, compute accuracy, and update the output files:
```bash
python -m src.evaluator.evaluate
```
