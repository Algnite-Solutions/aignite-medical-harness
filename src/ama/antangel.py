"""Parse AntAngel's tagged calls when the serving endpoint lacks its tool parser."""
from __future__ import annotations

import json
import re
from uuid import uuid4


def parse_tool_calls(message: dict, tools: list[dict]) -> dict:
    text = message.get('content')
    if message.get('tool_calls') or not isinstance(text, str) or '<tool_call>' not in text:
        return message
    definitions = {tool['function']['name']: tool['function'] for tool in tools}
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if len(blocks) != text.count('<tool_call>'):
        raise ValueError('incomplete AntAngel tool call')
    calls = []
    for block in blocks:
        name, _, body = block.strip().partition('\n')
        name = name.strip()
        if name not in definitions:
            raise ValueError(f'unknown AntAngel tool: {name}')
        properties = definitions[name].get('parameters', {}).get('properties', {})
        arguments = {}
        pattern = r'\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>'
        position = 0
        for match in re.finditer(pattern, body, re.S):
            if body[position:match.start()].strip():
                raise ValueError('malformed AntAngel arguments')
            key, value = match.group(1).strip(), match.group(2)
            if key in arguments or key not in properties:
                raise ValueError(f'invalid AntAngel argument: {key}')
            # String arguments are literal; typed arguments use JSON, never eval.
            arguments[key] = value if properties[key].get('type') == 'string' else json.loads(value)
            position = match.end()
        if body[position:].strip():
            raise ValueError('malformed AntAngel arguments')
        calls.append({'id': 'call_' + uuid4().hex[:16], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
    return {**message, 'tool_calls': calls}
