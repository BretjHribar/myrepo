"""Compact, occupied-orbital-unitary-invariant electronic descriptors.

Löwdin atomic subspaces partition the orthogonal AO density matrix. This is
basis-dependent data and must be checked on the eventual evaluation basis.
No nuclear charges or fitted atomic offsets enter the learned descriptors.
"""
import numpy as np
import torch


def overlap_roots(overlap):
    values, vectors = np.linalg.eigh(overlap)
    values = values.clip(min=1e-10)
    return ((vectors * np.sqrt(values)) @ vectors.T,
            (vectors / np.sqrt(values)) @ vectors.T)


def quantum_numpy(dm, root, slices):
    projected = root @ np.asarray(dm).sum(0) @ root
    charges = np.array([np.trace(projected[a:b, a:b]) for a, b in slices])
    coherence = np.array([[np.sum(projected[a:b, c:d]**2) for c, d in slices] for a, b in slices])
    return charges, coherence


def quantum_torch(dm, root, slices):
    projected = root @ dm.sum(-3) @ root
    charges = torch.stack([projected[..., a:b, a:b].diagonal(dim1=-2, dim2=-1).sum(-1)
                           for a, b in slices], -1)
    coherence = torch.stack([torch.stack([projected[..., a:b, c:d].square().sum((-1, -2))
                                          for c, d in slices], -1) for a, b in slices], -2)
    return {"charge": charges, "coherence": coherence, "projected": projected}
