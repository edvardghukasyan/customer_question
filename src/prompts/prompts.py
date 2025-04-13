class Prompts:
    classifier_prompt = """
        You are provided with:
        1. A raw customer question:
        {question}

        2. A set of predefined cluster names:
        {clusters}

        Your task is as follows:

        1. Analyze the semantic content of the question.
        2. Return the name of the cluster for the question.
        3. For the cluster:
           - If the semantic topic matches one of the predefined names, use that exact name.
           - Otherwise, generate a new descriptive name for the cluster.
        4. Create an object with the following keys:
           - "name": The cluster name.
           - "description": A clear explanation of what types of questions are included in the cluster.

        Output Format: Your final output must be a valid JSON object with two keys:
        - "name", "description".

        Example Output:
        {{{{
              "name": "Pricing Inquiries",
              "description": "Questions related to subscription plans, costs, discounts, or billing issues."
        }}}}

        Please process the question and provide the clustering result in the JSON format as described.
        """