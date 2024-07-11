import re
import json

def nice_json_dump(graph: dict) -> str:
    """Pretty print a graph as a JSON string.
    @param graph: The graph to print.
    @return: The pretty printed graph as a JSON string.
    """
    print_output = re.sub(r'",\s+', '", ', json.dumps(graph, sort_keys=True, indent=4))
    # newlines after commas
    print_output = re.sub(r",\n(?!(\s*(\[|\")))\s*", ",", print_output)
    # spaces after opening brackets
    print_output = re.sub(r"\[\s*(?=\d)", "[", print_output)
    # spaces before closing brackets
    print_output = re.sub(r"(?<=\d)\s*\]", "]", print_output)
    return print_output