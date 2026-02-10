import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from tools import paddle_ocr_read_document

from PIL import Image
import cv2


task = """
Extract the Training Cost (FLOPs) for EN-DE for ALL methods from
the table.jpeg using the OCR tool.
Return as a list with model name and its training cost. Can you also run the code and give me the extracted data itself rather than the code to run it.
"""

# 1. Define the list of tools
tools = [paddle_ocr_read_document]

# 2. Set up the OpenAI GPT model
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=1 
)

# Create LangGraph agent (without state_modifier)
agent = create_agent(llm, tools)

config = {
    "configurable": {"thread_id": "1"}
}

response = agent.invoke(
    {
        "messages": [
            SystemMessage(content="You are a helpful assistant designed to extract and analyze information from documents. "
                        "You have access to a PaddleOCR tool that extracts text, bounding boxes, and confidence scores from images. "
                        ),
            HumanMessage(content=task)
        ]
    },
    config=config
)

# Print results
print("\n\nAgent Response:")
print("\n--- All Messages ---")
for message in response["messages"]:
    print(f"\n{message.type.upper()}:")
    if hasattr(message, 'content'):
        print(message.content)
    else:
        print(message)

# Print just the final answer
print("\n--- Final Answer ---")
print(response["messages"][-1].content)