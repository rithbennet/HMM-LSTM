"""
config.py

Utility for loading configuration and environment variables.
"""

import os
from dotenv import load_dotenv

def load_api_keys():
    load_dotenv()
    return {
        "CRYPTOQUANT_API_KEY": os.getenv("CRYPTOQUANT_API_KEY"),
        "GLASSNODE_API_KEY": os.getenv("GLASSNODE_API_KEY"),
        "COINGLASS_API_KEY": os.getenv("COINGLASS_API_KEY"),
        # Add more keys as needed
    }

def get_data_dir():
    data_dir = os.getenv("DATA_DIR", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir