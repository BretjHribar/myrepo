"""Matrix-free real spatial-density operators in a fixed fermion sector.

Stores occupation links, not full Hilbert-space density matrices. The
fermionic word action is shared with the independently checked spin/block
engine. The symmetric one-body generator convention is documented here to
avoid double counting off-diagonal density entries.
"""
import torch
from .gpu_fock_blocks import word_connections


class DensityLinks:
    def __init__(self, bits, sites):
        if bits.device.type != "cuda":
            raise RuntimeError("Density-link numerical work requires CUDA")
        self.bits, self.sites, self.links = bits, sites, []
        for first in range(sites):
            for second in range(first, sites):
                words = torch.tensor([((first, spin, 1), (second, spin, 0)) for spin in range(2)], device=bits.device)
                rows, columns, factors = word_connections(words, torch.ones(2, device=bits.device), bits, bits)
                self.links.append((first, second, rows, columns, factors))
        self.first = torch.tensor([item[0] for item in self.links], device=bits.device)
        self.second = torch.tensor([item[1] for item in self.links], device=bits.device)

    @torch.no_grad()
    def image(self, operator, vectors):
        single = vectors.ndim == 1
        vectors = vectors[:, None] if single else vectors
        result = torch.zeros_like(vectors)
        for first, second, rows, columns, factors in self.links:
            coefficient = operator[first, second]
            result.index_add_(0, rows, (coefficient*factors)[:, None]*vectors[columns])
            if first != second:
                result.index_add_(0, columns, (coefficient*factors)[:, None]*vectors[rows])
        return result[:, 0] if single else result

    @torch.no_grad()
    def project(self, basis):
        result = []
        for first, second, rows, columns, factors in self.links:
            matrix = basis[rows].T @ (factors[:, None]*basis[columns])
            if first != second:
                matrix = matrix+matrix.T
            result.append((matrix+matrix.T)/2)
        return torch.stack(result)

    def reduced(self, operators, projected_generators):
        # For p<q the generator is E_pq+E_qp. Its coefficient is Q_pq,
        # rather than 2 Q_pq; both fermionic directions are already included.
        coefficients = operators[:, self.first, self.second]
        return torch.einsum("ap,pij->aij", coefficients, projected_generators)

    @torch.no_grad()
    def response_images(self, operators, states):
        images = states.new_empty(len(self.links), len(self.bits), states.shape[1])
        for index, (first, second, rows, columns, factors) in enumerate(self.links):
            image = images[index]; image.zero_()
            image.index_add_(0, rows, factors[:, None]*states[columns])
            if first != second:
                image.index_add_(0, columns, factors[:, None]*states[rows])
        coefficients = operators[:, self.first, self.second]
        return (coefficients @ images.flatten(1)).reshape(len(operators), len(self.bits), states.shape[1])

    def density(self, reduced_ensemble, projected_generators):
        values = torch.einsum("ij,pji->p", reduced_ensemble, projected_generators)
        values = torch.where(self.first == self.second, values, values/2)
        result = values.new_zeros(self.sites, self.sites)
        result[self.first, self.second] = values
        result[self.second, self.first] = values
        return result

    def storage_bytes(self):
        return sum(tensor.numel()*tensor.element_size() for item in self.links for tensor in item[2:])
