# Gemini AI Agent

## Overview
This project implements an adaptive AI personal assistant using the Google Gemini API.  
The agent follows the ReAct pattern (Reason → Act → Observe) and applies SOLID principles and design patterns.

## Architecture
- Agent – controls reasoning loop
- MemoryManager – stores conversation history
- ToolRegistry – manages tools dynamically
- BaseTool – abstract tool interface

## Features
- Natural language interaction
- Contextual memory
- Automatic tool selection
- Robust error handling

## Tools
- Calculator
- Time
- Weather
- File Reader (custom)
- Unit Converter (custom)

## Installation
pip install -r requirements.txt

## Run
python personal_assistant_agent.py
