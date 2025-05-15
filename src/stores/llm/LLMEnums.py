from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    HUGGINGFACE = "HUGGINGFACE"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"

    DOCUMENT = "search_document"
    QUERY = "search_query"
class HuggingFaceEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
class DocumentTypeEnum(Enum):
    SEARCH_DOCUMENT = "search_document"
    QUERY = "search_query"