# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:13:59 2023

@author: cevas
"""

import re

def regex_similarity(string1, string2):
    # Define a regex pattern to match common substrings
    pattern = re.compile(r'\b\w{3,}\b')  # Match words with at least 3 characters
    
    # Find all matches in both strings
    matches1 = set(pattern.findall(string1.lower()))  # Convert to lowercase for case insensitivity
    matches2 = set(pattern.findall(string2.lower()))
    
    # Calculate similarity as the Jaccard similarity between sets of matches
    intersection = len(matches1 & matches2)
    union = len(matches1 | matches2)
    similarity = intersection / union if union > 0 else 0.0
    
    return similarity

# Example usage:
string1 = "yudis"
string2 = "yudik"
pattern = re.compile(r'\b\w{3,}\b')  # Match words with at least 3 characters

# Find all matches in both strings
matches1 = set(pattern.findall(string1.lower()))  # Convert to lowercase for case insensitivity
matches2 = set(pattern.findall(string2.lower()))

# Calculate similarity as the Jaccard similarity between sets of matches
intersection = len(matches1 & matches2)
union = len(matches1 | matches2)
similarity = intersection / union if union > 0 else 0.0
    

similarity = regex_similarity(string1, string2)
print(f"Similarity: {similarity}")

