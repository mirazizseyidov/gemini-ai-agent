"""
Seyidov Miraziz - Adaptive Personal Assistant Agent

Assignment solution: Personal Assistant AI agent using the Google Gemini API,
implemented with SOLID principles and common design patterns:
- Strategy Pattern: each tool is an interchangeable strategy
- Registry/Factory Pattern: ToolRegistry dynamically stores and resolves tools
- Observer Pattern: optional event observers for logging/state monitoring
- ReAct Loop: reason -> act -> observe -> final response

Requirements:
    pip install google-genai requests

Environment:
    set GEMINI_API_KEY=your_api_key   # Windows
    export GEMINI_API_KEY=your_api_key  # Linux/macOS

Run:
    python personal_assistant_agent.py
"""
from __future__ import annotations

import abc
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import requests
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for agent-related failures."""


class ToolExecutionError(AgentError):
    """Raised when a tool fails during execution."""


class UnknownToolError(AgentError):
    """Raised when the model requests an unregistered tool."""


class ConfigurationError(AgentError):
    """Raised when the application is misconfigured."""


# ---------------------------------------------------------------------------
# Observer Pattern (bonus)
# ---------------------------------------------------------------------------


class Observer(Protocol):
    def update(self, event: str, payload: Dict[str, Any]) -> None:
        ...


class ConsoleLoggerObserver:
    """Simple observer that logs state changes without coupling to Agent."""

    def update(self, event: str, payload: Dict[str, Any]) -> None:
        logging.info("[%s] %s", event, json.dumps(payload, ensure_ascii=False, default=str))


@dataclass
class EventBus:
    observers: List[Observer] = field(default_factory=list)

    def subscribe(self, observer: Observer) -> None:
        self.observers.append(observer)

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        for observer in self.observers:
            observer.update(event, payload)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryManager:
    """Stores the current session conversation history for Gemini."""

    system_prompt: str
    _history: List[types.Content] = field(default_factory=list)

    def reset(self) -> None:
        self._history.clear()

    def add_user_message(self, text: str) -> None:
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )

    def add_model_content(self, content: types.Content) -> None:
        self._history.append(content)

    def add_function_response(self, name: str, response: Dict[str, Any], call_id: Optional[str]) -> None:
        part = types.Part.from_function_response(
            name=name,
            response=response,
            id=call_id,
        )
        self._history.append(types.Content(role="user", parts=[part]))

    def build_contents(self) -> List[types.Content]:
        system_content = types.Content(
            role="user",
            parts=[types.Part(text=f"SYSTEM INSTRUCTION:\n{self.system_prompt}")],
        )
        return [system_content, *self._history]


# ---------------------------------------------------------------------------
# Tools - Strategy Pattern
# ---------------------------------------------------------------------------


class BaseTool(abc.ABC):
    """Abstract tool contract enforcing OCP/DIP-friendly extensibility."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_declaration(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    def get_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Evaluate a mathematical expression. Use this for arithmetic, "
                "percentages, powers, roots, and common math functions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression such as '(25*8)/2' or 'sqrt(81)+5'",
                    }
                },
                "required": ["expression"],
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        expression = str(kwargs["expression"])
        safe_globals = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "ceil": math.ceil,
            "floor": math.floor,
            "pi": math.pi,
            "e": math.e,
        }
        try:
            result = eval(expression, safe_globals, {})  # noqa: S307 - restricted globals
        except Exception as exc:
            raise ToolExecutionError(f"Invalid mathematical expression: {exc}") from exc
        return {"expression": expression, "result": result}


class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "current_time"

    def get_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Return the current date and time. Optionally convert to a UTC offset like '+02:00' or '-05:00'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_offset": {
                        "type": "string",
                        "description": "Optional UTC offset in format '+HH:MM' or '-HH:MM'",
                    }
                },
                "required": [],
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        offset = kwargs.get("utc_offset")
        if offset:
            sign = 1 if offset[0] == "+" else -1
            hours, minutes = map(int, offset[1:].split(":"))
            delta_minutes = sign * (hours * 60 + minutes)
            tz = timezone.utc
            now_utc = datetime.now(tz)
            converted = now_utc.timestamp() + delta_minutes * 60
            dt = datetime.fromtimestamp(converted, tz=timezone.utc)
            return {
                "utc_offset": offset,
                "iso_datetime": dt.isoformat(),
                "formatted": dt.strftime("%Y-%m-%d %H:%M:%S"),
            }

        now = datetime.now().astimezone()
        return {
            "timezone": str(now.tzinfo),
            "iso_datetime": now.isoformat(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        }


class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather_lookup"

    def get_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "Get the current weather for a city or place name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location, for example 'Riga' or 'London'",
                    }
                },
                "required": ["location"],
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        location = str(kwargs["location"]).strip()
        if not location:
            raise ToolExecutionError("Location cannot be empty.")

        try:
            response = requests.get(
                f"https://wttr.in/{location}",
                params={"format": "j1"},
                timeout=10,
                headers={"User-Agent": "AdaptiveAgent/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            current = data["current_condition"][0]
        except Exception as exc:
            raise ToolExecutionError(f"Unable to retrieve weather for '{location}': {exc}") from exc

        return {
            "location": location,
            "temperature_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "humidity": current.get("humidity"),
            "wind_kmph": current.get("windspeedKmph"),
            "description": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
        }


class LocalFileReaderTool(BaseTool):
    """Custom tool #1: reads small local text files safely."""

    @property
    def name(self) -> str:
        return "read_local_file"

    def get_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Read a local text file when the user asks to inspect a file on disk. "
                "Only use for text-like files such as .txt, .md, .py, .json, .csv, or .log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute file path",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to read, default 4000",
                    },
                },
                "required": ["path"],
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        path = Path(str(kwargs["path"])).expanduser().resolve()
        max_chars = int(kwargs.get("max_chars", 4000))

        allowed_suffixes = {".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml"}
        if path.suffix.lower() not in allowed_suffixes:
            raise ToolExecutionError(
                f"Unsupported file type '{path.suffix}'. Allowed: {sorted(allowed_suffixes)}"
            )
        if not path.exists() or not path.is_file():
            raise ToolExecutionError(f"File not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception as exc:
                raise ToolExecutionError(f"Unable to decode file: {exc}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"Unable to read file: {exc}") from exc

        return {
            "path": str(path),
            "characters_returned": min(len(text), max_chars),
            "truncated": len(text) > max_chars,
            "content": text[:max_chars],
        }


class UnitConverterTool(BaseTool):
    """Custom tool #2: useful non-trivial conversion tool."""

    @property
    def name(self) -> str:
        return "unit_converter"

    def get_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Convert between supported units. Categories: length (m, km, cm, mm, mi, yd, ft, in), "
                "weight (kg, g, lb), temperature (C, F, K)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        value = float(kwargs["value"])
        from_unit = str(kwargs["from_unit"]).lower()
        to_unit = str(kwargs["to_unit"]).lower()

        length = {
            "mm": 0.001,
            "cm": 0.01,
            "m": 1.0,
            "km": 1000.0,
            "in": 0.0254,
            "ft": 0.3048,
            "yd": 0.9144,
            "mi": 1609.344,
        }
        weight = {"g": 0.001, "kg": 1.0, "lb": 0.45359237}

        if from_unit in length and to_unit in length:
            result = value * length[from_unit] / length[to_unit]
        elif from_unit in weight and to_unit in weight:
            result = value * weight[from_unit] / weight[to_unit]
        elif {from_unit, to_unit}.issubset({"c", "f", "k"}):
            result = self._convert_temperature(value, from_unit, to_unit)
        else:
            raise ToolExecutionError(f"Unsupported conversion from '{from_unit}' to '{to_unit}'.")

        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": result,
        }

    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        if from_unit == to_unit:
            return value

        celsius = value
        if from_unit == "f":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "k":
            celsius = value - 273.15

        if to_unit == "c":
            return celsius
        if to_unit == "f":
            return celsius * 9 / 5 + 32
        if to_unit == "k":
            return celsius + 273.15
        raise ToolExecutionError(f"Unsupported temperature unit: {to_unit}")


# ---------------------------------------------------------------------------
# Registry / Factory Pattern
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    _tools: Dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool requested: {name}") from exc

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get(name)
        return tool.execute(**arguments)

    def declarations(self) -> List[types.Tool]:
        function_declarations = [tool.get_declaration() for tool in self._tools.values()]
        return [types.Tool(function_declarations=function_declarations)]

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    model_name: str = "gemini-2.5-flash"
    max_reasoning_steps: int = 6
    temperature: float = 0.2


class PersonalAssistantAgent:
    def __init__(
        self,
        client: genai.Client,
        memory: MemoryManager,
        registry: ToolRegistry,
        config: AgentConfig,
        events: Optional[EventBus] = None,
    ) -> None:
        self.client = client
        self.memory = memory
        self.registry = registry
        self.config = config
        self.events = events or EventBus()

    def chat(self, user_text: str) -> str:
        self.memory.add_user_message(user_text)
        self.events.emit("user_message", {"text": user_text})

        for step in range(1, self.config.max_reasoning_steps + 1):
            response = self._generate()
            self.events.emit("model_response_received", {"step": step})

            function_calls = list(response.function_calls or [])
            if not function_calls:
                model_text = response.text or "I could not generate a response."
                self._append_model_content_if_present(response)
                self.events.emit("final_answer", {"step": step, "text": model_text})
                return model_text

            self._append_model_content_if_present(response)

            for call in function_calls:
                args = dict(call.args or {})
                self.events.emit(
                    "tool_requested",
                    {"step": step, "tool": call.name, "arguments": args},
                )
                tool_result = self._safe_execute_tool(call.name, args)
                self.memory.add_function_response(call.name, tool_result, getattr(call, "id", None))
                self.events.emit(
                    "tool_completed",
                    {"step": step, "tool": call.name, "result": tool_result},
                )

        return (
            "I reached the maximum reasoning steps for this request. "
            "Please try rephrasing or asking for a simpler subtask."
        )

    def _generate(self):
        contents = self.memory.build_contents()
        generation_config = types.GenerateContentConfig(
            tools=self.registry.declarations(),
            temperature=self.config.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            return self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=generation_config,
            )
        except Exception as exc:
            raise AgentError(f"Gemini API request failed: {exc}") from exc

    def _safe_execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self.registry.execute(tool_name, arguments)
            return {"ok": True, "data": result}
        except UnknownToolError as exc:
            return {"ok": False, "error": str(exc), "error_type": "UnknownToolError"}
        except ToolExecutionError as exc:
            return {"ok": False, "error": str(exc), "error_type": "ToolExecutionError"}
        except Exception as exc:  # last safety net
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    def _append_model_content_if_present(self, response: Any) -> None:
        candidate_content = getattr(response.candidates[0], "content", None) if getattr(response, "candidates", None) else None
        if candidate_content is not None:
            self.memory.add_model_content(candidate_content)


# ---------------------------------------------------------------------------
# Bootstrap / Composition Root
# ---------------------------------------------------------------------------


def build_system_prompt(available_tools: Iterable[str]) -> str:
    return (
        "You are a modular personal assistant AI agent. "
        "Be helpful, concise, and accurate. "
        "Use tools only when needed. "
        "When a tool is needed, choose the most appropriate one and provide valid arguments. "
        "If a tool returns an error, explain the issue clearly and continue helping the user. "
        "Remember earlier turns in the same session. "
        f"Available tools: {', '.join(available_tools)}."
    )


def create_agent() -> PersonalAssistantAgent:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Create an API key in Google AI Studio and set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(WeatherTool())
    registry.register(LocalFileReaderTool())
    registry.register(UnitConverterTool())

    memory = MemoryManager(system_prompt=build_system_prompt(registry.list_tool_names()))

    events = EventBus()
    events.subscribe(ConsoleLoggerObserver())

    config = AgentConfig()
    return PersonalAssistantAgent(client=client, memory=memory, registry=registry, config=config, events=events)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("Adaptive Personal Assistant Agent")
    print("Type 'exit' to quit, 'reset' to clear session memory.\n")

    try:
        agent = create_agent()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.memory.reset()
            print("Assistant: Session memory has been cleared.")
            continue

        try:
            answer = agent.chat(user_input)
            print(f"Assistant: {answer}\n")
        except AgentError as exc:
            print(f"Assistant: Sorry, I ran into an error: {exc}\n")
        except Exception as exc:
            print(f"Assistant: Unexpected error: {exc}\n")


if __name__ == "__main__":
    main()
