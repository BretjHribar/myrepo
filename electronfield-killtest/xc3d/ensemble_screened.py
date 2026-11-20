"""Coherence-resolved ensemble cancellation plus our learned screening state.

This construction cancels Hartree and baseline XC for rank-one, N<=1 charge
and spin ensembles. It does not assert exact exchange for general determinants.
Disconnected occupied subspaces have zero cross-cancellation. Learned terms
are suppressed where the analytic identity applies, so it does not compete
with chemical energy fitting as an additional loss.
"""
import numpy as np
import torch
from pyscf import df, dft
from .screened_state import ScreenedStateXC


def whitened_integrals(mol, inverse_root):
    auxiliary = df.addons.make_auxmol(mol)
    metric = auxiliary.intor("int2c2e")
    three = df.incore.aux_e2(mol, auxiliary, intor="int3c2e", aosym="s1")
    orthogonal = np.einsum("ia,ijp,jb->abp", inverse_root, three, inverse_root, optimize=True)
    # Q_ij = L^-1 B_ij, with V=L L^T. Frobenius products of Q yield RI Coulomb integrals.
    return np.linalg.solve(np.linalg.cholesky(metric), orthogonal.reshape(-1, len(metric)).T).T.reshape(orthogonal.shape)


def regional_hartree(projected, integrals, slices):
    fitted = torch.stack([torch.einsum("...ij,ijp->...p", projected[..., a:b, :], integrals[a:b])
                          for a, b in slices], -2)
    return fitted @ fitted.transpose(-1, -2)


class PBEDensity(torch.autograd.Function):
    """LibXC energy density with its exact first functional derivative.

SCF uses first derivatives; fixed-state parameter fitting does not differentiate
this function twice. A future response Hessian must implement LibXC fxc here.
"""
    @staticmethod
    def forward(ctx, rho):
        array = rho.detach().cpu().double().numpy()
        eps, vxc, _, _ = dft.numint.NumInt().eval_xc_eff("pbe", array, deriv=1, spin=1)
        ctx.save_for_backward(torch.tensor(vxc, device=rho.device, dtype=rho.dtype))
        return torch.tensor(eps*array[:, 0].sum(0), device=rho.device, dtype=rho.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (vxc,) = ctx.saved_tensors
        return gradient[None, None, :] * vxc


class EnsembleScreenedXC(ScreenedStateXC):
    needs_ensemble = True

    def __init__(self, width=32, channels=8, operator="diffusive", quantum=True):
        super().__init__(width, channels, operator, quantum=True)

    def forward(self, rho, tau, template, prepared=None, quantum=None):
        if quantum is None or "hartree_matrix" not in quantum or "pbe_density" not in quantum:
            raise ValueError("Ensemble-screened functional requires consistent Coulomb and baseline-XC fields")
        charge = quantum["charge"].clamp_min(1e-12)
        shared = quantum["coherence"]
        eta = shared.sum(-1)/charge
        transition = (eta-1).clamp(0, 1)
        open_orbital = 1-transition.square()*(3-2*transition)
        purity = (shared.diagonal(dim1=-2, dim2=-1)/charge.square()).clamp(0, 1)
        # Select an effective local rank below two. Applying a small rank-one
        # weight to every many-orbital core would cancel a large core energy.
        # The smooth endpoints preserve the exact rank-one construction.
        rank_one = (2*purity-1).clamp(0, 1)
        rank_one = rank_one.square()*(3-2*rank_one)
        weight = open_orbital*rank_one
        coherence = shared/(charge[..., :, None]*charge[..., None, :])
        cancellation = -.5*(quantum["hartree_matrix"]*coherence*
                             weight[..., :, None]*weight[..., None, :]).sum((-1, -2))
        grid_weight = torch.einsum("ga,...a->...g", template.partition.to(rho.dtype), weight)
        weights64 = getattr(template, "weights64", template.weights).double()
        remove_baseline = (quantum["pbe_density"].double()*grid_weight.double()*weights64).sum(-1)
        learned = super().forward(rho, tau, template, prepared, quantum=quantum, suppression=weight)
        return cancellation-remove_baseline+learned
