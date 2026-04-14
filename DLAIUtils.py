import os
import sys

from dotenv import find_dotenv, load_dotenv


class Utils:
    def __init__(self):
        pass

    def _load_env(self):
        load_dotenv(find_dotenv())

    def _get_secret(self, name):
        self._load_env()

        if self.is_colab():  # google colab
            try:
                from google.colab import userdata
            except ImportError:
                return os.getenv(name)

            value = userdata.get(name)
            if value:
                return value

        return os.getenv(name)

    def create_dlai_index_name(self, index_name):
        openai_key = self._get_secret("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Load it before creating the Pinecone index."
            )

        return f'{index_name}-{openai_key[-36:].lower().replace("_", "-")}'

    def is_colab(self):
        return "google.colab" in sys.modules

    def get_openai_api_key(self):
        return self._get_secret("OPENAI_API_KEY")

    def get_pinecone_api_key(self):
        return self._get_secret("PINECONE_API_KEY")

    def list_pinecone_index_names(self, pinecone_client):
        raw_indexes = pinecone_client.list_indexes()
        return self._normalize_pinecone_index_names(raw_indexes)

    def _normalize_pinecone_index_names(self, raw_indexes):
        if raw_indexes is None:
            return []

        if isinstance(raw_indexes, dict):
            error = raw_indexes.get("error")
            if error:
                raise RuntimeError(f"Unable to list Pinecone indexes: {error}")
            raw_indexes = raw_indexes.get("indexes", raw_indexes.get("data", raw_indexes))

        index_container = getattr(raw_indexes, "index_list", raw_indexes)
        index_entries = getattr(index_container, "indexes", index_container)

        if index_entries is None:
            to_dict = getattr(index_container, "to_dict", None)
            if callable(to_dict):
                data = to_dict()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(
                        f"Unable to list Pinecone indexes: {data['error']}"
                    )
            return []

        names = []
        for index in index_entries:
            if isinstance(index, str):
                names.append(index)
                continue

            if isinstance(index, dict):
                name = index.get("name")
            else:
                name = getattr(index, "name", None)

            if name:
                names.append(name)

        return names
