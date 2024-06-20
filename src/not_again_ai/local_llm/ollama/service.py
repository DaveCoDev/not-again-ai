from typing import Any

from ollama import Client

from not_again_ai.base.file_system import readable_size


def list_models(client: Client) -> list[dict[str, Any]]:
    """List models that are available locally.

    Args:
        client (Client): The Ollama client.

    Returns:
        list[dict[str, Any]]: A list of dictionaries (each corresponding to an available model) with the following keys:
            name (str): Name of the model
            model (str): Name of the model. This should be the same as the name.
            modified_at (str): The date and time the model was last modified.
            size (int): The size of the model in bytes.
            size_readable (str): The size of the model in a human-readable format.
            details (dict[str, Any]): Additional details about the model.
    """
    response = client.list().get("models", [])

    response_data = []
    for model_data in response:
        curr_model_data = {}
        curr_model_data["name"] = model_data["name"]
        curr_model_data["model"] = model_data["model"]
        curr_model_data["modified_at"] = model_data["modified_at"]
        curr_model_data["size"] = model_data["size"]
        curr_model_data["size_readable"] = readable_size(model_data["size"])
        curr_model_data["details"] = model_data["details"]

        response_data.append(curr_model_data)

    return response_data


def is_model_available(model_name: str, client: Client) -> bool:
    """Check if a model is available locally.

    Args:
        model_name (str): The name of the model.
        client (Client): The Ollama client.

    Returns:
        bool: True if the model is available locally, False otherwise.
    """
    # If model_name does not have a ":", append ":latest"
    if ":" not in model_name:
        model_name = f"{model_name}:latest"
    models = list_models(client)
    return any(model["name"] == model_name for model in models)


def show(model_name: str, client: Client) -> dict[str, Any]:
    """Show information about a model including the modelfile, available parameters, template, and additional details.

    Args:
        model_name (str): The name of the model.
        client (Client): The Ollama client.
    """
    response = client.show(model_name)

    response_data = {}
    response_data["modelfile"] = response["modelfile"]
    response_data["parameters"] = response["parameters"]
    response_data["template"] = response["template"]
    response_data["details"] = response["details"]
    return response_data


def pull(model_name: str, client: Client) -> Any:
    """Pull a model from the Ollama server and returns the status of the pull operation."""
    return client.pull(model_name)


def delete(model_name: str, client: Client) -> Any:
    """Delete a model from the local filesystem and returns the status of the delete operation."""
    return client.delete(model_name)
