import numpy as np
import torch
import torch.nn.functional as F


class MemoryBank:
    def __init__(
        self,
        size,
        key_dim,
        value_dim,
        num_labels,
        momentum=0.9,
        device="cpu",
        dtype=torch.float32,
        search="exact",
        faiss_nlist=100,
        faiss_nprobe=10,
        seed=0,
    ):
        self.size = int(size)
        self.momentum = float(momentum)
        self.search_mode = search
        self.faiss_nlist = int(faiss_nlist)
        self.faiss_nprobe = int(faiss_nprobe)
        self.keys = torch.zeros(self.size, key_dim, dtype=dtype, device=device)
        self.values = torch.zeros(self.size, value_dim, dtype=dtype, device=device)
        self.labels = torch.zeros(self.size, num_labels, dtype=torch.bool)
        self.members = None
        self.index = None
        self.rng = np.random.default_rng(seed)

    @property
    def device(self):
        return self.keys.device

    @torch.no_grad()
    def write(self, indices, keys=None, values=None, labels=None):
        indices = torch.as_tensor(indices, dtype=torch.long)
        rows = indices.to(self.device)
        if keys is not None:
            self.keys[rows] = F.normalize(keys.float().to(self.device), dim=-1).to(self.keys.dtype)
        if values is not None:
            self.values[rows] = values.to(self.device, self.values.dtype)
        if labels is not None:
            self.labels[indices.cpu()] = labels.detach().cpu() > 0

    @torch.no_grad()
    def momentum_update(self, indices, keys):
        rows = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        old = self.keys[rows].float()
        new = F.normalize(keys.float().to(self.device), dim=-1)
        mixed = F.normalize(self.momentum * old + (1.0 - self.momentum) * new, dim=-1)
        self.keys[rows] = mixed.to(self.keys.dtype)

    def finalize(self):
        labels = self.labels.numpy()
        self.members = [np.nonzero(labels[:, c])[0] for c in range(labels.shape[1])]
        self.refresh_index()

    def refresh_index(self):
        if self.search_mode != "faiss":
            self.index = None
            return
        import faiss

        data = np.ascontiguousarray(self.keys.float().cpu().numpy())
        dim = data.shape[1]
        if self.size >= 39 * self.faiss_nlist:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(data)
            index.nprobe = self.faiss_nprobe
            self._quantizer = quantizer
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(data)
        self.index = index

    @torch.no_grad()
    def _exact_search(self, query, top_r, exclude):
        scores = (query.to(self.device, self.keys.dtype) @ self.keys.t()).float()
        if exclude is not None:
            rows = torch.arange(scores.size(0), device=scores.device)
            scores[rows, exclude.to(scores.device)] = float("-inf")
        return scores.topk(top_r, dim=-1).indices

    @torch.no_grad()
    def search(self, query, top_r, exclude=None):
        if exclude is not None:
            exclude = torch.as_tensor(exclude, dtype=torch.long).view(-1)
        top_r = max(1, min(int(top_r), self.size - (0 if exclude is None else 1)))
        if self.index is None:
            return self._exact_search(query, top_r, exclude)
        _, found = self.index.search(np.ascontiguousarray(query.detach().float().cpu().numpy()), top_r + 1)
        found = torch.from_numpy(found).long()
        rows = []
        for b in range(found.size(0)):
            row = found[b][found[b] >= 0]
            if exclude is not None:
                row = row[row != int(exclude[b])]
            if row.numel() < top_r:
                ex = None if exclude is None else exclude[b : b + 1]
                row = self._exact_search(query[b : b + 1], top_r, ex)[0].cpu()
            rows.append(row[:top_r])
        return torch.stack(rows).to(self.device)

    def _choice(self, candidates):
        return int(candidates[self.rng.integers(candidates.size)])

    def sample_contrastive(self, primary, confusing, self_indices=None):
        labels = self.labels.numpy()
        positives, negatives = [], []
        for b in range(len(primary)):
            cp, cn = int(primary[b]), int(confusing[b])
            own = -1 if self_indices is None else int(self_indices[b])
            cand = self.members[cp]
            cand = cand[cand != own]
            if cand.size:
                positives.append(self._choice(cand))
            else:
                positives.append(own if own >= 0 else int(self.rng.integers(self.size)))
            cand = self.members[cn]
            cand = cand[(~labels[cand, cp]) & (cand != own)]
            if cand.size == 0:
                cand = np.nonzero(~labels[:, cp])[0]
                cand = cand[cand != own]
            negatives.append(self._choice(cand) if cand.size else int(self.rng.integers(self.size)))
        return torch.tensor(positives, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

    def state_dict(self):
        return {
            "keys": self.keys.cpu(),
            "values": self.values.cpu(),
            "labels": self.labels.cpu(),
            "momentum": self.momentum,
        }

    def load_state_dict(self, state):
        self.keys = state["keys"].to(self.device, self.keys.dtype)
        self.values = state["values"].to(self.device, self.values.dtype)
        self.labels = state["labels"].bool().cpu()
        self.size = self.keys.size(0)
        self.momentum = float(state.get("momentum", self.momentum))
        self.finalize()

    @classmethod
    def from_state(cls, state, device="cpu", dtype=torch.float32, search="exact", faiss_nlist=100, faiss_nprobe=10):
        bank = cls(
            state["keys"].size(0),
            state["keys"].size(1),
            state["values"].size(1),
            state["labels"].size(1),
            momentum=state.get("momentum", 0.9),
            device=device,
            dtype=dtype,
            search=search,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
        )
        bank.load_state_dict(state)
        return bank
