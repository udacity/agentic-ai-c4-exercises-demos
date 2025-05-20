### Uluru Cultural Center: Multi-Agent System Solution – Presentation Script

#### Goal of the System

The goal is to help visitors get culturally respectful and accurate information about Uluru in their own language, especially in underrepresented Indigenous languages like Arrernte and Pitjantjatjara. This system uses AI agents and tools that work together to provide tailored, multimodal responses.

---

#### Core System Flow (Original Design Recap)

Let’s quickly revisit the original system before diving into the extensions. In the original version:

* A **Visitor Input** component captures a question or prompt from the user.
* The system passes that input to a **Language Identification** tool, which acts as an orchestrator.
* Depending on the detected language, the system routes the input to either an **Arrernte** or **Pitjantjatjara Language Specialist Agent**.
* These agents then query a **Knowledge Base Lookup** tool to get relevant information.
* The language agents return the response, translated into the appropriate language, which is sent back to the visitor.

This formed a basic loop of user input, language routing, knowledge retrieval, and response delivery.

---

