from typing import Any, Optional
from uuid import uuid4
import json


class BaseInteraction:
    """Base class for interaction management."""
    
    def __init__(self, config: dict[str, Any], *args, **kwargs):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")
        self._instance_dict: dict[str, Any] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, 
                                identity: dict[str, Any] = None, **kwargs) -> str:
        """Create an interaction instance.

        Args:
            instance_id: The instance id of the interaction.
            identity: The identity of the interaction for a specific task.

        Returns:
            The instance id of the interaction.
        """
        # Ensure identity is a dict
        if not isinstance(identity, dict):
            try:
                identity = json.loads(identity)
            except:
                print(f"[DEBUG BaseInteraction] Error in start_interaction: identity is not a dict")
                return None
        
        if instance_id is None:
            instance_id = str(uuid4())
            self._instance_dict[instance_id] = {
                "identity": identity,
            }
            return instance_id
        else:
            if instance_id not in self._instance_dict:
                self._instance_dict[instance_id] = {
                    "identity": identity,
                }
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Generates a response for the current turn of interaction.
        Returns a tuple containing:
        - should_terminate_sequence (bool): True if the interaction sequence should end.
        - response_content (str): The textual content of the response.
        - current_turn_score (float): The score for this specific turn/response.
        - additional_data (dict): Any extra information or metadata.
        """
        should_terminate_sequence: bool = False
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.8
        additional_data: dict[str, Any] = {}
        return should_terminate_sequence, response_content, current_turn_score, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Calculates a score for the interaction.
        """
        score = 0.0
        return score

    async def finalize_interaction(self, instance_id: str, **kwargs) -> bool:
        """
        Finalizes the interaction session and releases any associated state or resources.
        """
        try:
            del self._instance_dict[instance_id]
            return True
        except Exception as e:
            print(f"Error in finalize_interaction: {e}")
            return False

