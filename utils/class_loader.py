import importlib

def load_class_from_string(class_path: str):
    """
    Load a class from a string path.
    
    Args:
        class_path: The full path of the class, e.g., 'tools.reward_calculator.RewardCalculator'
        
    Returns:
        The loaded class object
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to load class {class_path}: {str(e)}")

