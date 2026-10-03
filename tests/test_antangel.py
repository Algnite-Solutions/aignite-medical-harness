import json
import pytest
from ama.antangel import parse_tool_calls

TOOLS = [{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'key':{'type':'string'},'n':{'type':'integer'}}}}}]


def test_tags_preserve_raw_and_string_arguments():
    raw = '<tool_call>lookup\n<arg_key>key</arg_key>\n<arg_value>001</arg_value>\n<arg_key>n</arg_key><arg_value>2</arg_value>\n</tool_call>'
    result = parse_tool_calls({'role':'assistant','content':raw}, TOOLS)
    assert result['content'] == raw
    assert json.loads(result['tool_calls'][0]['function']['arguments']) == {'key':'001','n':2}
    assert parse_tool_calls({'content':'Final diagnosis'},TOOLS) == {'content':'Final diagnosis'}


def test_multiple_calls_have_unique_ids():
    result = parse_tool_calls({'content':'<tool_call>lookup\n</tool_call>' * 2},TOOLS)
    assert len({c['id'] for c in result['tool_calls']}) == 2


@pytest.mark.parametrize('raw', ['<tool_call>lookup', '<tool_call>unknown\n</tool_call>', '<tool_call>lookup\ninvalid</tool_call>'])
def test_malformed_calls_fail_explicitly(raw):
    with pytest.raises(ValueError):
        parse_tool_calls({'content':raw},TOOLS)
