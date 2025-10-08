import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from .sync_manager import SyncManager

class Node:
    """
    A Node represents one participant in the distributed training mesh.
    It trains its local submodel and synchronizes gradients with its peers.
    """

    def __init__(self, node_id: str, model: nn.Module, optimizer: optim.Optimizer, peers: list[str]):
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.peers = peers
        self.sync_manager = SyncManager(self)

    async def train_step(self, data, target, loss_fn):
        """Performs one local training step."""
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = loss_fn(output, target)
        loss.backward()
        await self.sync_manager.sync_gradients()
        self.optimizer.step()
        return loss.item()

    async def run(self, dataloader, loss_fn, epochs=1):
        """Runs local training with async gradient synchronization."""
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                loss = await self.train_step(data, target, loss_fn)
                if batch_idx % 10 == 0:
                    print(f"[Node {self.node_id}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
