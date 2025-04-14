# Customer Questions Classifier

This repository contains code for classifying customer questions using language models (LLMs) via LangChain. Our solution is designed to work with the [Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset/tree/main) (downloaded as `data.csv`) and predicts clusters of questions. The final output is an aggregated JSON format of predicted clusters, each containing:

```json
{
  "name": "Technical Issues",
  "description": "Questions about bugs, errors, or problems using the product features.",
  "count": 31
}
