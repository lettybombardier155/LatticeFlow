import torch
import asyncio
import aiohttp

class SyncManager:
    """
    Handles synchronization of gradients between nodes.
    Simplified — in production should use gRPC streaming or Ray actors.
    """

    def __init__(self, node):
        self.node = node

    async def sync_gradients(self):
        """Synchronize gradients asynchronously between peers."""
        peers = self.node.peers
        if not peers:
            return  # Single-node fallback

        tasks = []
        for peer in peers:
            tasks.append(self._send_gradients(peer))
        await asyncio.gather(*tasks)

    async def _send_gradients(self, peer_url):
        grads = [p.grad.clone().cpu().tolist() for p in self.node.model.parameters() if p.grad is not None]
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"http://{peer_url}/sync", json={"grads": grads}):
                    pass
            except Exception as e:
                print(f"[SyncManager] Failed to sync with {peer_url}: {e}")
