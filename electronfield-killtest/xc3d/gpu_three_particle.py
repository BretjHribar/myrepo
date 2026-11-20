"""Independent CUDA reference action for three physical electrons.

The alpha-alpha-beta Cartesian tensor is antisymmetric in its first two
axes. This is a bounded few-particle reference primitive for the proposed
open-shell accuracy tests, not an XC model or a speed experiment.
"""
import torch
from .gpu_fock_blocks import word_connections
from .gpu_manybody_ops import fixed_sector


class ThreeParticleOperator:
    def __init__(self, h, eri):
        if h.device.type != "cuda" or eri.device != h.device:
            raise RuntimeError("Three-electron reference work requires CUDA")
        self.h, self.sites = h, len(h)
        self.pair = eri.permute(0, 2, 1, 3).contiguous().reshape(len(h)**2, len(h)**2)

    def one(self, value, axis):
        moved = value.movedim(axis, 0)
        return (self.h @ moved.reshape(self.sites, -1)).reshape_as(moved).movedim(0, axis)

    def two(self, value, first, second):
        order = [first, second]+[axis for axis in range(3) if axis not in (first, second)]
        reverse = [order.index(axis) for axis in range(3)]
        moved = value.permute(order)
        return (self.pair @ moved.reshape(self.sites**2, -1)).reshape_as(moved).permute(reverse)

    def __call__(self, vector):
        value = vector.reshape((self.sites,)*3)
        first = self.one(value, 0)
        result = first-first.transpose(0, 1)+self.one(value, 2)+self.two(value, 0, 1)
        first = self.two(value, 0, 2)
        result = result+first-first.transpose(0, 1)
        return ((result-result.transpose(0, 1))/2).flatten()


class ThreeParticleFockAction:
    def __init__(self, operator, bits):
        self.operator, self.bits = operator, bits
        sites = torch.arange(operator.sites, device=bits.device)
        up_mask = ((bits[:, None] >> (2*sites[None])) & 1).bool()
        down_mask = ((bits[:, None] >> (2*sites[None]+1)) & 1).bool()
        if not bool((up_mask.sum(-1) == 2).all() & (down_mask.sum(-1) == 1).all()):
            raise ValueError("The independent reference requires exactly two alpha and one beta electrons")
        up = sites[None].expand(len(bits), -1)[up_mask].reshape(len(bits), 2)
        down = sites[None].expand(len(bits), -1)[down_mask]
        self.first = (up[:, 0]*operator.sites+up[:, 1])*operator.sites+down
        self.second = (up[:, 1]*operator.sites+up[:, 0])*operator.sites+down
        inversions = (up > down[:, None]).sum(-1)
        self.sign = 1-2*inversions.remainder(2)
        self.sqrt_two = operator.h.new_tensor(2.).sqrt()

    def to_tensor(self, vector):
        value = vector.new_zeros(self.operator.sites**3)
        coefficient = self.sign*vector/self.sqrt_two
        value[self.first], value[self.second] = coefficient, -coefficient
        return value

    def to_fock(self, vector):
        return self.sqrt_two*self.sign*vector[self.first]

    def __call__(self, vector):
        return self.to_fock(self.operator(self.to_tensor(vector)))


class ThreeElectronDoubletProjector:
    """P_(S=1/2)=(15/4-S^2)/3 at M=1/2; sparse CUDA spin links."""
    def __init__(self, bits, sites):
        target, _ = fixed_sector(sites, 1, 2, bits.device)
        words = torch.tensor([((site, 1, 1), (site, 0, 0)) for site in range(sites)], device=bits.device)
        self.rows, self.columns, self.factors = word_connections(words, torch.ones(sites, device=bits.device), target, bits)
        self.source_size, self.target_size = len(bits), len(target)

    def spin_squared(self, vector):
        lowered = vector.new_zeros(self.target_size)
        lowered.scatter_add_(0, self.rows, self.factors*vector[self.columns])
        raised = vector.new_zeros(self.source_size)
        raised.scatter_add_(0, self.columns, self.factors*lowered[self.rows])
        return raised-.25*vector

    def __call__(self, vector):
        return (3.75*vector-self.spin_squared(vector))/3
