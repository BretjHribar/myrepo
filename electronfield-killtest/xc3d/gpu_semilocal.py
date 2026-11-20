"""Unpolarized PBE energy density in PyTorch CUDA, for cached singlet inputs.

Numerical equations/parameters follow LibXC's primary Maple definitions:
https://gitlab.com/libxc/libxc/-/blob/7.0.0/maple/gga_exc/gga_x_pbe.mpl
https://gitlab.com/libxc/libxc/-/blob/7.0.0/maple/gga_exc/gga_c_pbe.mpl
https://gitlab.com/libxc/libxc/-/blob/7.0.0/maple/lda_exc/lda_c_pw.mpl
Only mathematical equations/constants are re-expressed here; no native LibXC
calls or CPU fallback. This standard energy term is infrastructure, not a
frontier reproduction experiment or our proposed research contribution.
"""
import math
import torch


def pbe_singlet(rho):
    if rho.device.type != "cuda":
        raise RuntimeError("GPU-only numerical policy")
    total = rho.sum(-3)
    n = total[..., 0, :].clamp_min(1e-24)
    gradient2 = total[..., 1:4, :].square().sum(-2)
    kf = (3*math.pi**2*n).pow(1/3)
    s2 = gradient2/(4*kf.square()*n.square())
    exchange = -.75*(3/math.pi)**(1/3)*n.pow(4/3)*(1+.804-.804/(1+.2195149727645171*s2/.804))
    rs = (3/(4*math.pi*n)).pow(1/3)
    root = rs.sqrt()
    polynomial = 7.5957*root+3.5876*rs+1.6382*rs*root+.49294*rs.square()
    lda = -2*.0310907*(1+.21370*rs)*torch.log1p(1/(2*.0310907*polynomial))
    beta, gamma = .06672455060314922, (1-math.log(2))/math.pi**2
    ks2 = 4*kf/math.pi
    t2 = gradient2/(4*ks2*n.square())
    a = (beta/gamma)/torch.expm1(-lda/gamma).clamp_min(1e-30)
    at = a*t2
    # Equivalent to t^2(1+A*t^2)/(1+A*t^2+A^2*t^4), avoiding fourth powers of t.
    fraction = t2*(1+at)/(1+at+at.square())
    correlation = n*(lda+gamma*torch.log1p((beta/gamma)*fraction))
    return exchange+correlation


def pbe_spin(rho):
    """Spin PBE with modified-PW92 interpolation, on CUDA only."""
    if rho.device.type != "cuda":
        raise RuntimeError("GPU-only numerical policy")
    spin_n = rho[..., :, 0, :].clamp_min(1e-24)
    spin_gradient2 = rho[..., :, 1:4, :].square().sum(-2)
    spin_kf = (6*math.pi**2*spin_n).pow(1/3)
    spin_s2 = spin_gradient2/(4*spin_kf.square()*spin_n.square())
    enhancement = 1+.804-.804/(1+.2195149727645171*spin_s2/.804)
    exchange = (-.75*(6/math.pi)**(1/3)*spin_n.pow(4/3)*enhancement).sum(-2)
    n = spin_n.sum(-2)
    zeta = ((spin_n[..., 0, :]-spin_n[..., 1, :])/n).clamp(-1+1e-14, 1-1e-14)
    rs = (3/(4*math.pi*n)).pow(1/3)
    root = rs.sqrt()
    def pw(a, alpha, b1, b2, b3, b4):
        polynomial = b1*root+b2*rs+b3*rs*root+b4*rs.square()
        return -2*a*(1+alpha*rs)*torch.log1p(1/(2*a*polynomial))
    g0 = pw(.0310907, .21370, 7.5957, 3.5876, 1.6382, .49294)
    g1 = pw(.01554535, .20548, 14.1189, 6.1977, 3.3662, .62517)
    g2 = pw(.0168869, .11125, 10.357, 3.6231, .88026, .49671)
    fz = ((1+zeta).pow(4/3)+(1-zeta).pow(4/3)-2)/(2**(4/3)-2)
    lda = g0+zeta.pow(4)*fz*(g1-g0+g2/1.7099209341613656)-fz*g2/1.7099209341613656
    phi = ((1+zeta).pow(2/3)+(1-zeta).pow(2/3))/2
    gradient2 = rho[..., :, 1:4, :].sum(-3).square().sum(-2)
    kf = (3*math.pi**2*n).pow(1/3)
    t2 = gradient2/(4*phi.square()*(4*kf/math.pi)*n.square())
    beta, gamma = .06672455060314922, (1-math.log(2))/math.pi**2
    a = (beta/gamma)/torch.expm1(-lda/(gamma*phi.pow(3))).clamp_min(1e-30)
    at = a*t2
    correlation = n*(lda+gamma*phi.pow(3)*torch.log1p((beta/gamma)*t2*(1+at)/(1+at+at.square())))
    return exchange+correlation
