import asyncio
import grpc
from concurrent import futures

# Mock service definitions — replace with .proto later
class MeshService:
    """Simple P2P mesh communication for gradient exchange."""
    def __init__(self):
        self.peers = {}

    def register_peer(self, node_id, address):
        self.peers[node_id] = address
        print(f"[Mesh] Registered peer {node_id} at {address}")

    def get_peers(self):
        return list(self.peers.items())

mesh_service = MeshService()

async def start_mesh_server(host="0.0.0.0", port=50051):
    """Starts a lightweight async gRPC-like server for peers."""
    server = grpc.aio.server()
    # TODO: add proto registration
    addr = f"{host}:{port}"
    await server.start()
    print(f"[Mesh] Server started on {addr}")
    await server.wait_for_termination()
