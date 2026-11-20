"""Bounded CUDA density-fitting factors for the physical molecular operator.

No full four-index ERI, full three-index tensor, or full inverse metric is
constructed by this producer. Density fitting is an input approximation,
not a new model or a chemical-accuracy guarantee. Its errors need measuring.
"""
from contextlib import contextmanager
import torch


@contextmanager
def cuda_integrals_only():
    """Turn any accidental PySCF CPU integral route into an explicit failure."""
    from pyscf import gto
    from pyscf.gto import moleintor
    targets = [(gto.Mole, name) for name in ("intor", "intor_symmetric", "intor_asymmetric", "energy_nuc")]
    targets += [(moleintor, name) for name in ("getints", "getints2c", "getints3c", "getints4c") if hasattr(moleintor, name)]
    original = [(obj, name, getattr(obj, name)) for obj, name in targets]
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU integrals/energies are forbidden; CUDA-only factor generation has no CPU fallback")
    for obj, name, _ in original: setattr(obj, name, forbidden)
    try: yield
    finally:
        for obj, name, value in original: setattr(obj, name, value)


class GPUCoulombFactorSource:
    """Regenerable real symmetric AO factors using bounded 2D integral blocks.

For metric J=R R^T and B[ij,P]=(ij|P), factors are R^-1 B^T.
A requested factor-column block needs only R^-T times selected identity
columns. All auxiliary/Cartesian/contraction transformations stay CUDA.
"""
    def __init__(self, mol, auxiliary_basis, *, pair_batch_size=4096,
                 auxiliary_batch_size=128, factor_chunk_size=2,
                 maximum_working_bytes=5_500_000_000, screening_cutoff=1e-14,
                 metric_reconstruction_relative_tolerance=1e-10):
        import cupy as cp
        from pyscf import df
        from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt, int2c2e
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; no CPU numerical fallback")
        if min(pair_batch_size, auxiliary_batch_size, factor_chunk_size) < 1:
            raise ValueError("Positive bounded integral/factor blocks are required")
        if mol._ecp: raise ValueError("GPU ECP one-body domain is not yet validated for this molecular adapter")
        self.mol = mol
        self.auxmol = df.addons.make_auxmol(mol, auxiliary_basis)
        self.sites, self.count = mol.nao_nr(), self.auxmol.nao_nr()
        self.dtype, self.device = torch.float64, torch.device("cuda", torch.cuda.current_device())
        self.auxiliary_basis = auxiliary_basis
        self.factor_chunk_size = factor_chunk_size
        self.maximum_working_bytes = maximum_working_bytes
        self.screening_cutoff = screening_cutoff
        self.progress_callback = None
        # Conservatively reserve metric, factorization and validation buffers
        # before any large numerical allocation. All values here are shapes.
        minimum_metric_work = 4*self.count*self.count*8+64_000_000
        free, _ = torch.cuda.mem_get_info()
        if minimum_metric_work > min(maximum_working_bytes, free):
            raise MemoryError(f"Auxiliary metric workspace exceeds the declared/free CUDA budget: {minimum_metric_work}")
        with cuda_integrals_only():
            self.opt = Int3c2eOpt(mol, self.auxmol).build(cutoff=screening_cutoff)
            self.evaluate, _, pair_offsets, aux_offsets = self.opt.int3c2e_evaluator(
                ao_pair_batch_size=pair_batch_size, aux_batch_size=auxiliary_batch_size,
                reorder_aux=False, cart=True)
            self.pair_offsets = [int(value) for value in pair_offsets]
            self.aux_offsets = [int(value) for value in aux_offsets]
            self.cartesian_sites = int(self.opt.mol.nao_nr(cart=True))
            self.cartesian_auxiliary = int(self.opt.auxmol.nao_nr(cart=True))
            self.pair_addresses = torch.from_dlpack(self.opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]).long()
            if self.pair_offsets[-1] != len(self.pair_addresses):
                raise RuntimeError("Integral batching and sorted Cartesian pair mapping disagree")
            metric = torch.from_dlpack(int2c2e(self.auxmol))
        if metric.device != self.device or metric.dtype != self.dtype or metric.shape != (self.count, self.count):
            raise RuntimeError("The auxiliary metric was not returned on CUDA in the original basis")
        if not bool(torch.isfinite(metric).all()): raise RuntimeError("Nonfinite Coulomb metric")
        self.cholesky = torch.linalg.cholesky(metric)
        # Validation samples fixed columns of the actual metric. This checks
        # all rows without creating another full naux-by-naux product.
        columns = torch.arange(0, self.count, max(1, self.count//16), device=self.device)
        reconstructed = self.cholesky @ self.cholesky[columns].T
        error = (reconstructed-metric[:, columns]).norm()/metric[:, columns].norm().clamp_min(1e-30)
        if not bool(error < metric_reconstruction_relative_tolerance):
            raise RuntimeError("CUDA metric factorization failed its fixed reconstruction guard")
        self.metric_relative_error = float(error)
        self.minimum_metric_cholesky_diagonal = float(self.cholesky.diagonal().min())
        del metric, reconstructed
        # Output, both AO transforms, integral block and whitening/solve work.
        largest_pair_batch = max(b-a for a, b in zip(self.pair_offsets, self.pair_offsets[1:]))
        largest_aux_batch = max(b-a for a, b in zip(self.aux_offsets, self.aux_offsets[1:]))
        self.estimated_working_bytes = (self.cholesky.numel()*8
            +4*factor_chunk_size*(self.cartesian_sites**2+self.sites**2)*8
            +2*largest_pair_batch*largest_aux_batch*8
            +4*(self.count+self.cartesian_auxiliary)*factor_chunk_size*8+64_000_000)
        free, _ = torch.cuda.mem_get_info()
        remaining = self.estimated_working_bytes-self.cholesky.numel()*8
        if self.estimated_working_bytes > maximum_working_bytes or remaining > free:
            raise MemoryError(f"Bounded Coulomb factor output/transform exceeds declared/free CUDA budget: {self.estimated_working_bytes}")

    def chunks(self, start=0, stop=None):
        import cupy as cp
        stop = self.count if stop is None else stop
        if not 0 <= start <= stop <= self.count: raise ValueError("Factor interval outside the frozen metric")
        for first in range(start, stop, self.factor_chunk_size):
            last = min(stop, first+self.factor_chunk_size)
            width = last-first
            rhs = self.cholesky.new_zeros(self.count, width)
            rhs[torch.arange(first, last, device=self.device), torch.arange(width, device=self.device)] = 1.
            whitener = torch.linalg.solve_triangular(self.cholesky.T, rhs, upper=True)
            # Sorted Cartesian auxiliary coefficients times R^-T; this
            # includes every original auxiliary function and its contraction.
            with cuda_integrals_only():
                mix = torch.from_dlpack(self.opt.auxmol.C_dot_mat(cp.from_dlpack(whitener)))
            if mix.shape != (self.cartesian_auxiliary, width):
                raise RuntimeError("Sorted auxiliary transformation changed the factor indexing")
            cartesian = self.cholesky.new_zeros(width, self.cartesian_sites**2)
            for pair_batch, (p0, p1) in enumerate(zip(self.pair_offsets, self.pair_offsets[1:])):
                accumulated = self.cholesky.new_zeros(p1-p0, width)
                for aux_batch, (a0, a1) in enumerate(zip(self.aux_offsets, self.aux_offsets[1:])):
                    with cuda_integrals_only():
                        integral = torch.from_dlpack(self.evaluate(pair_batch, aux_batch))
                    if integral.shape != (p1-p0, a1-a0): raise RuntimeError("Unexpected CUDA integral block dimensions")
                    accumulated.addmm_(integral, mix[a0:a1])
                addresses = self.pair_addresses[p0:p1]
                transposed = addresses.remainder(self.cartesian_sites)*self.cartesian_sites+addresses.div(self.cartesian_sites, rounding_mode="floor")
                cartesian[:, addresses] = accumulated.T
                cartesian[:, transposed] = accumulated.T
                if self.progress_callback and (pair_batch % 50 == 0 or pair_batch+2 == len(self.pair_offsets)):
                    self.progress_callback({"factor_start": first, "factor_stop": last,
                        "completed_pair_batches": pair_batch+1, "total_pair_batches": len(self.pair_offsets)-1})
            with cuda_integrals_only():
                values = torch.from_dlpack(self.opt.mol.apply_CT_mat_C(cp.from_dlpack(cartesian.reshape(width, self.cartesian_sites, self.cartesian_sites))))
            if values.shape != (width, self.sites, self.sites) or not bool(torch.isfinite(values).all()):
                raise RuntimeError("Invalid full physical AO factor output")
            if float((values-values.transpose(-1, -2)).abs().max()) > 1e-10:
                raise RuntimeError("Coulomb factors lost real symmetry; do not silently symmetrize")
            yield first, values

    def __iter__(self):
        for _, values in self.chunks():
            yield from values.unbind(0)

    def summary(self):
        return {"basis_functions": self.sites, "cartesian_basis_functions": self.cartesian_sites,
            "auxiliary_basis": self.auxiliary_basis, "factor_count": self.count,
            "cartesian_auxiliary_functions": self.cartesian_auxiliary,
            "factor_chunk_size": self.factor_chunk_size, "pair_batches": len(self.pair_offsets)-1,
            "auxiliary_batches": len(self.aux_offsets)-1,
            "maximum_generated_integral_block_shape": [max(b-a for a,b in zip(self.pair_offsets,self.pair_offsets[1:])), max(b-a for a,b in zip(self.aux_offsets,self.aux_offsets[1:]))],
            "estimated_working_bytes": self.estimated_working_bytes,
            "metric_relative_reconstruction_error": self.metric_relative_error,
            "minimum_cholesky_diagonal": self.minimum_metric_cholesky_diagonal,
            "screening_cutoff": self.screening_cutoff,
            "full_inverse_metric_allocated": False, "full_three_index_tensor_allocated": False,
            "full_four_index_tensor_allocated": False, "cpu_integrals_allowed": False,
            "chemical_accuracy_established": False}
