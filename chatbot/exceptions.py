class ChatbotError(Exception):
    """Base chatbot exception."""


class GeminiError(ChatbotError):
    """Raised when Gemini API fails."""