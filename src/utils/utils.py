import json
import re


def clean_json_string(json_string: str) -> str:
    # Remove any invalid control characters.
    json_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_string)
    # Escape newlines and returns.
    json_string = json_string.replace('\n', '\\n').replace('\r', '\\r')
    return json_string


def extract_json_from_llm_response(response_str: str):
    json_pattern = re.search(r"\{.*\}", response_str, re.DOTALL)
    if json_pattern:
        extracted_json = json_pattern.group()
        extracted_json = clean_json_string(extracted_json)
        try:
            json_data = json.loads(extracted_json)
            # Optionally check for known keys:
            json_data = json_data.get("result") or json_data.get("Result") or json_data
            return json_data
        except json.JSONDecodeError:
            raise Exception(f"Failed to extract JSON from LLM response: {response_str}")
    else:
        raise Exception(f"Failed to extract JSON from LLM response: {response_str}")