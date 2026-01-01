import yaml
from typing import Any, Callable, Dict, List, Tuple
import importlib
from schemas.tool_schemas import OpenAIFunctionToolSchema


def load_tool_from_config(tool_config: dict) -> Tuple[str, dict, Any]:
    """
    Dynamically load a tool class from YAML config and instantiate it.

    Args:
        tool_config (dict): Tool configuration, must contain:
            - class_name (str): Full module path of the tool class, e.g., 'tools.search_tool.GoogleSearchTool'
            - tool_schema (dict): OpenAI tool format schema definition (dict)
            - config (dict, optional): Parameters passed to tool class initialization

    Returns:
        tuple[str, dict, Any]:
            - function_name: Tool function name (for matching LLM output)
            - raw_schema: Raw schema dict
            - instance: Tool instance

    Raises:
        ImportError: If class cannot be loaded, schema validation fails, or instantiation fails
    """
    class_path = tool_config["class_name"]
    raw_schema = tool_config["tool_schema"]
    config = tool_config.get("config", {})

    # === 1. Validate and construct OpenAIFunctionToolSchema instance ===
    try:
        validated_schema = OpenAIFunctionToolSchema(**raw_schema)
    except Exception as e:
        raise ImportError(f"Invalid tool schema: does not conform to OpenAIFunctionToolSchema. Error: {e}")

    function_name = validated_schema.function.name

    # === 2. Dynamically import class ===
    try:
        *modules, class_name = class_path.split(".")
        module_path = ".".join(modules)
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
    except AttributeError:
        raise ImportError(f"Module '{module_path}' has no class '{class_name}'")

    # === 3. Instantiate tool (pass config and OpenAIFunctionToolSchema instance) ===
    try:
        instance = tool_class(config=config, tool_schema=validated_schema)
    except TypeError as e:
        raise ImportError(
            f"Failed to instantiate {class_path}. "
            f"Make sure it inherits from BaseTool and accepts (config, tool_schema: OpenAIFunctionToolSchema). Error: {e}"
        )

    # === 4. Return: function name, schema dict, instance ===
    return function_name, raw_schema, instance


def load_tool_from_config_path(tool_config_path: str) -> List[Tuple[str, dict, Any]]:
    """
    Load tool configurations from a YAML config file path and instantiate all tools.

    Args:
        tool_config_path (str): Path to YAML config file, should contain 'tools' key with list of tool configs

    Returns:
        List[tuple[str, dict, Any]]: List of tool configurations, each element contains:
            - function_name: Tool function name (for matching LLM output)
            - raw_schema: Raw schema dict
            - instance: Tool instance

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML file format is incorrect
        KeyError: If YAML file is missing 'tools' key
        ImportError: If tool class cannot be loaded or instantiation fails
    """
    try:
        with open(tool_config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Tool config file not found: {tool_config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config file '{tool_config_path}': {e}")
    
    # Check if 'tools' key exists
    if 'tools' not in config_data:
        raise KeyError(f"YAML config file '{tool_config_path}' must contain a 'tools' key")
    
    tools_config = config_data['tools']
    if not isinstance(tools_config, list):
        raise ValueError(f"'tools' key in '{tool_config_path}' must contain a list of tool configurations")
    
    # Load all tools
    loaded_tools = []
    for tool_config in tools_config:
        tool_result = load_tool_from_config(tool_config)
        loaded_tools.append(tool_result)
    
    return loaded_tools

