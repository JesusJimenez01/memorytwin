import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from memorytwin.escriba.storage import MemoryStorage
import json

def inspect_latest():
    try:
        storage = MemoryStorage()
        # Get the very last episode
        episodes = storage.get_timeline(limit=1)

        if episodes:
            episode = episodes[0]
            print(f"✅ Último episodio encontrado (ID: {episode.id})")
            print("-" * 50)
            # Print full JSON representation
            print(episode.model_dump_json(indent=2))
            print("-" * 50)
        else:
            print("❌ No se encontraron episodios en la base de datos local.")
            print(f"Ruta buscada: {storage.sqlite_path}")

    except Exception as e:
        print(f"Error al leer la base de datos: {e}")

if __name__ == "__main__":
    inspect_latest()
