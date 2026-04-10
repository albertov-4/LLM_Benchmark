"""
Answer Post-processor for PDDL Plans

This module provides functionality to clean, format, and extract structured information 
from LLM-generated PDDL plans and responses.
"""

import re
from typing import List, Dict, Optional, Tuple
import json


def formatter(raw_response: str, include_reasoning: bool = False) -> Dict:
    """
    Format and clean raw LLM response to extract PDDL plan.
    
    Args:
        raw_response (str): Raw response from the language model
        include_reasoning (bool): Whether to include reasoning/explanation in output
        
    Returns:
        Dict: Formatted response with extracted plan and metadata
    """
    
    if not raw_response or not isinstance(raw_response, str):
        return {
            "plan": [],
            "raw_response": raw_response,
            "reasoning": "",
            "confidence": "unknown",
            "format_issues": ["Empty or invalid response"]
        }
    
    # Extract different components
    plan_actions = extract_plan_actions(raw_response)
    reasoning = extract_reasoning(raw_response) if include_reasoning else ""
    confidence = extract_confidence_indicators(raw_response)
    format_issues = detect_format_issues(raw_response)
    
    return {
        "plan": plan_actions,
        "raw_response": raw_response,
        "reasoning": reasoning,
        "confidence": confidence,
        "format_issues": format_issues,
        "plan_length": len(plan_actions)
    }


def extract_plan_actions(text: str) -> List[str]:
    """
    Extract PDDL action sequence from text response.
    
    Args:
        text (str): Text containing PDDL plan
        
    Returns:
        List[str]: List of PDDL actions in proper format
    """
    
    if not text:
        return []
    
    # Extract the assistant's response part to avoid picking up domain/problem content
    if "assistant\n\n" in text:
        text = text.split("assistant\n\n")[-1]
    
    # Debug: log a preview of the isolated text
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Isolated assistant text preview: {text[:500]}...")
    
    # Keep a copy of the raw text (after assistant split) for fallback
    raw_text_fallback = text
    
    # Clean the text first
    cleaned_text = clean_response_text(text)
    
    # Debug: log cleaned text
    logger.debug(f"Cleaned text preview: {cleaned_text[:500]}...")
    
    # Helper to try extraction on a given text
    def try_extract(input_text):
        acts = []
        # Strategy 1: Look for explicit plan sections
        plan_section = extract_plan_section(input_text)
        if plan_section:
            acts = parse_action_lines(plan_section)
        
        # Strategy 2: Extract parenthesized actions
        if not acts:
            acts = extract_parenthesized_actions(input_text)
        
        # Strategy 3: Extract numbered/bulleted actions  
        if not acts:
            acts = extract_numbered_actions(input_text)
        
        # Strategy 4: Extract lines starting with '('
        if not acts:
            acts = extract_lines_starting_with_paren(input_text)
        return acts

    # 1. Try extracting from cleaned text (standard behavior)
    actions = try_extract(cleaned_text)
    
    # 2. Fallback: If no actions found, try extracting from raw text
    # This handles cases where the plan is inside <think> blocks or the cleaning was too aggressive
    if not actions:
        logger.debug("No actions found in cleaned text. Attempting fallback to raw text...")
        actions = try_extract(raw_text_fallback)
        if actions:
            logger.debug(f"Fallback successful: found {len(actions)} actions in raw text.")
    
    # Clean and validate actions
    return [clean_action(action) for action in actions if is_valid_action(action)]


def clean_response_text(text: str) -> str:
    """
    Clean raw response text from common LLM artifacts.
    
    Args:
        text (str): Raw response text
        
    Returns:
        str: Cleaned text
    """
    
    # Remove <think>...</think> blocks (common in reasoning models)
    # Handle both complete blocks and unclosed blocks (if generation was truncated)
    # Also handle Kimi's style: ◁think▷...◁/think▷ or similar variants if they appear
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'◁think▷.*?(?:◁/think▷|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove common prefixes/suffixes
    text = re.sub(r'^(Here is|Here\'s|The plan is|Plan:)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(I hope this helps|Let me know if|Any questions)', '', text, flags=re.IGNORECASE)
    
    # Remove code block markers
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def extract_plan_section(text: str) -> Optional[str]:
    """
    Extract the main plan section from response.
    
    Args:
        text (str): Full response text
        
    Returns:
        Optional[str]: Plan section if found
    """
    
    # Look for explicit plan markers
    plan_patterns = [
        r'(?:plan|solution|actions?):\s*\n?(.*?)(?:\n\n|$)',
        r'(?:here (?:is|are) the actions?):\s*\n?(.*?)(?:\n\n|$)',
        r'(?:step-by-step|sequence):\s*\n?(.*?)(?:\n\n|$)'
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


def parse_action_lines(text: str) -> List[str]:
    """
    Parse individual action lines from text.
    
    Args:
        text (str): Text containing action lines
        
    Returns:
        List[str]: List of extracted actions
    """
    
    lines = text.split('\n')
    actions = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Remove common line prefixes
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line)  # Numbers
        line = re.sub(r'^[-\*\+]\s*', '', line)     # Bullets
        line = re.sub(r'^step\s*\d*[\:\.\)]\s*', '', line, flags=re.IGNORECASE)
        
        if line and not line.lower().startswith(('note:', 'explanation:', 'because:')):
            actions.append(line)
    
    return actions


def _looks_like_action(text: str) -> bool:
    """
    Heuristic to determine if text looks like a PDDL action.
    
    Args:
        text (str): Text to evaluate
        
    Returns:
        bool: True if text looks like an action
    """
    
    text = text.lower().strip()
    
    # Should contain action-like words
    action_indicators = [
        'move', 'place', 'pick', 'put', 'go', 'drive', 'load', 'unload',
        'drop', 'lift', 'push', 'pull', 'rotate', 'turn', 'shift',
        'car', 'arrived', 'start', 'build', 'destroy', 'straight', 'diagonal'
    ]
    
    # Should not contain explanation words
    explanation_words = [
        'because', 'since', 'therefore', 'this will', 'we need', 'explanation',
        'note that', 'remember', 'important', 'first we', 'then we'
    ]
    
    has_action_word = any(word in text for word in action_indicators)
    has_explanation = any(word in text for word in explanation_words)
    
    # Basic format check - should have multiple words
    words = text.split()
    has_good_length = 2 <= len(words) <= 10
    
    return has_action_word and not has_explanation and has_good_length


def extract_parenthesized_actions(text: str) -> List[str]:
    """
    Extract actions in parentheses format.
    
    Args:
        text (str): Text to search
        
    Returns:
        List[str]: List of actions in parentheses
    """
    
    # Find all parenthesized expressions
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, text)
    
    actions = []
    for match in matches:
        # Filter out non-action parenthesized content
        if _looks_like_action(match):
            actions.append(f"({match})")
    
    return actions


def extract_numbered_actions(text: str) -> List[str]:
    """
    Extract numbered or bulleted action lists.
    
    Args:
        text (str): Text to search
        
    Returns:
        List[str]: List of extracted actions
    """
    
    # Pattern for numbered/bulleted items
    pattern = r'(?:^|\n)\s*(?:\d+[\.\)\:]|[-\*\+])\s*([^\n]+)'
    matches = re.findall(pattern, text, re.MULTILINE)
    
    actions = []
    for match in matches:
        match = match.strip()
        if _looks_like_action(match):
            actions.append(match)
    
    return actions


def extract_lines_starting_with_paren(text: str) -> List[str]:
    """
    Extract lines that start with '(' as potential actions.
    
    Args:
        text (str): Text to search
        
    Returns:
        List[str]: List of extracted actions
    """
    
    lines = text.split('\n')
    actions = []
    for line in lines:
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            # Remove outer parentheses for checking
            inner = line[1:-1].strip()
            if _looks_like_action(inner):
                actions.append(line)
    
    return actions


def clean_action(action: str) -> str:
    """
    Clean and standardize individual action format.
    
    Args:
        action (str): Raw action string
        
    Returns:
        str: Cleaned action string
    """
    
    action = action.strip()
    
    # Remove action prefix if present
    action = re.sub(r'^:action\s+', '', action, flags=re.IGNORECASE)
    
    # Ensure proper parentheses
    if not action.startswith('(') and not action.endswith(')'):
        action = f"({action})"
    elif action.startswith('(') and not action.endswith(')'):
        action = f"{action})"
    elif not action.startswith('(') and action.endswith(')'):
        action = f"({action}"
    
    # Normalize internal spacing
    action = re.sub(r'\s+', ' ', action)
    
    return action


def is_valid_action(action: str) -> bool:
    """
    Validate if action string is properly formatted.
    
    Args:
        action (str): Action string to validate
        
    Returns:
        bool: True if action is valid
    """
    
    if not action or len(action.strip()) < 3:
        return False
    
    action = action.strip()
    
    # Should have parentheses or be convertible to parentheses format
    if not (action.startswith('(') and action.endswith(')')):
        # Try to see if it's a valid action name + parameters
        parts = action.split()
        if len(parts) < 1:
            return False
    
    # Should not contain invalid characters for PDDL
    # Allow common punctuation at the end
    cleaned = re.sub(r'[.,;]+$', '', action)
    if re.search(r'[^\w\s\(\)\-_]', cleaned):
        return False
    
    return True


def extract_reasoning(text: str) -> str:
    """
    Extract reasoning/explanation from response.
    
    Args:
        text (str): Full response text
        
    Returns:
        str: Extracted reasoning
    """
    
    # Look for explanation sections
    reasoning_patterns = [
        r'(?:explanation|reasoning|because|since):\s*(.*?)(?:plan:|solution:|$)',
        r'(?:first|initially),?\s+(.*?)(?:then|next|finally)',
        r'(?:the strategy is|approach is)\s+(.*?)(?:\n\n|$)'
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_confidence_indicators(text: str) -> str:
    """
    Extract confidence indicators from response.
    
    Args:
        text (str): Response text
        
    Returns:
        str: Confidence level (high/medium/low/unknown)
    """
    
    text_lower = text.lower()
    
    high_confidence = ['definitely', 'certainly', 'clearly', 'optimal', 'best solution']
    medium_confidence = ['should work', 'likely', 'probably', 'appears to']
    low_confidence = ['might', 'possibly', 'unsure', 'not certain', 'may not']
    
    if any(phrase in text_lower for phrase in high_confidence):
        return "high"
    elif any(phrase in text_lower for phrase in low_confidence):
        return "low"
    elif any(phrase in text_lower for phrase in medium_confidence):
        return "medium"
    else:
        return "unknown"


def detect_format_issues(text: str) -> List[str]:
    """
    Detect potential formatting issues in response.
    
    Args:
        text (str): Response text to analyze
        
    Returns:
        List[str]: List of detected issues
    """
    
    issues = []
    
    if not text or not text.strip():
        issues.append("Empty response")
        return issues
    
    # Check for common issues
    if len(text) > 5000:
        issues.append("Response too long")
    
    if not re.search(r'\(.*\)', text):
        issues.append("No parenthesized actions found")
    
    if text.count('(') != text.count(')'):
        issues.append("Mismatched parentheses")
    
    if 'error' in text.lower() or 'sorry' in text.lower():
        issues.append("Contains error indicators")
    
    # Check for incomplete response
    if text.endswith('...') or 'incomplete' in text.lower():
        issues.append("Appears incomplete")
    
    return issues


def format_for_validation(plan_actions: List[str]) -> str:
    """
    Format plan actions for external validation tools.
    
    Args:
        plan_actions (List[str]): List of plan actions
        
    Returns:
        str: Formatted plan string
    """
    
    if not plan_actions:
        return ""
    
    formatted_lines = []
    for action in plan_actions:
        # Ensure proper formatting
        if not action.startswith('('):
            action = f"({action})"
        formatted_lines.append(action)
    
    return '\n'.join(formatted_lines)


def extract_metrics(response_data: Dict) -> Dict:
    """
    Extract metrics from formatted response data.
    
    Args:
        response_data (Dict): Formatted response dictionary
        
    Returns:
        Dict: Metrics and statistics
    """
    
    return {
        "plan_length": response_data.get("plan_length", 0),
        "has_reasoning": bool(response_data.get("reasoning", "")),
        "confidence_level": response_data.get("confidence", "unknown"),
        "format_issues_count": len(response_data.get("format_issues", [])),
        "response_length": len(response_data.get("raw_response", "")),
        "has_format_issues": len(response_data.get("format_issues", [])) > 0
    }