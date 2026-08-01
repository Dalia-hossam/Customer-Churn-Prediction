# دعم التوافق مع المكتبتين (الجديدة google-genai والقديمة google-generativeai)
GENAI_SDK_V2 = False
try:
    from google import genai
    GENAI_SDK_V2 = True
except ImportError:
    import google.generativeai as genai

from chatbot.actions import Action
from chatbot.config import DEFAULT_MODEL, GEMINI_API_KEY
from chatbot.context import build_customer_context
from chatbot.exceptions import GeminiError
from chatbot.memory import ChatMemory
from chatbot.prompts import (
    ACTION_PROMPTS,
    SYSTEM_PROMPT,
)


class ChurnAssistant:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        if not GEMINI_API_KEY:
            raise GeminiError("GEMINI_API_KEY is missing from environment or st.secrets")

        self.model_name = model_name
        self.memory = ChatMemory()

        # تهيئة العميل بناءً على المكتبة المثبتة في السيرفر
        if GENAI_SDK_V2:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(self.model_name)

    def ask(
        self,
        customer: dict,
        prediction: dict,
        question: str = "",
        action: Action = Action.FREE_CHAT,
    ) -> str:

        if isinstance(action, Action):
            action_key = action.value
        else:
            action_key = action

        task = ACTION_PROMPTS.get(
            action_key,
            ACTION_PROMPTS["chat"],
        )

        context = build_customer_context(
            customer,
            prediction,
        )

        history = self.memory.get_history()

        prompt = f"""
{SYSTEM_PROMPT}

===================================

TASK

{task}

===================================

CUSTOMER DATA

{context}

===================================

CONVERSATION HISTORY

{history}

===================================

USER QUESTION

{question}
"""

        try:
            # التوليد بحسب المكتبة
            if GENAI_SDK_V2:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
            else:
                response = self.client.generate_content(prompt)

            answer = response.text.strip()

            if question:
                self.memory.add_user(question)

            self.memory.add_assistant(answer)

            return answer

        except Exception as e:
            raise GeminiError(
                f"Gemini API Error: {e}"
            ) from e

    def clear_memory(self):
        self.memory.clear()