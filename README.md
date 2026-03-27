Adaptive Personal Assistant Agent - Notes

Files:
- personal_assistant_agent.py
- README_personal_assistant_agent.txt

Install:
    pip install google-genai requests

Set API key:
Windows CMD:
    set GEMINI_API_KEY=your_api_key

PowerShell:
    $env:GEMINI_API_KEY="your_api_key"

Linux/macOS:
    export GEMINI_API_KEY="your_api_key"

Run:
    python personal_assistant_agent.py

Architecture highlights:
- BaseTool: abstract interface for all tools
- ToolRegistry: dynamic registration/execution (Factory/Registry)
- MemoryManager: session conversation memory
- PersonalAssistantAgent: ReAct loop orchestration
- EventBus + ConsoleLoggerObserver: optional Observer pattern

Included tools:
- calculator
- current_time
- weather_lookup
- read_local_file (custom)
- unit_converter (custom)

Example prompts:
- What time is it in UTC+02:00?
- Calculate (45*12)/3
- What's the weather in Riga?
- Convert 15 miles to kilometers.
- Read the file ./notes.txt
