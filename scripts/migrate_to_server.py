"""Script para migrar memorias de local a ChromaDB Server."""
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.escriba.storage_chromadb_server import ChromaDBServerStorage

# Origen: local
local = MemoryStorage()
local_stats = local.get_statistics()
print(f"Local: {local_stats['total_episodes']} episodios")

# Destino: servidor
server = ChromaDBServerStorage()

# Migrar todos los episodios
timeline = local.get_timeline(limit=100)
print(f"Migrando {len(timeline)} episodios...")

for episode in timeline:
    server.store_episode(episode)
    print(f"  + {episode.task[:50]}...")

print("\n✓ Migración completada!")
server_stats = server.get_statistics()
print(f"Server ahora tiene: {server_stats['total_episodes']} episodios")
