class ClientError(Exception):
    """Exception raised when a client (e.g. OpenAI API) call fails."""
    pass


class ExtractionError(Exception):
    """Exception raised when JSON extraction from LLM response fails."""
    pass
