# %%
import torch

def torch_1d_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: float or None = None,
    right: float or None = None,
) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: 1d sequence of floats. x-coordinates. Must be increasing
        fp: 1d sequence of floats. y-coordinates. Must be same length as xp
        left: Value to return for x < xp[0], default is fp[0]
        right: Value to return for x > xp[-1], default is fp[-1]

    Returns:
        The interpolated values, same shape as x.
    """
    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    if fp.shape[0] == len(x):
        fp = fp.permute(1,2,0)
        permute_order = (2,0,1)
    elif fp.shape[1] == len(x):
        fp = fp.permute(0,2,1)
        permute_order = (0,2,1)
    else:
        permute_order = (0,1,2)

    answer = torch.where(
        x < xp[0],
        left,
        (fp[:,:,i - 1] * (xp[i] - x) + fp[:,:,i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )
    answer = torch.where(x > xp[-1], right, answer)
    answer = answer.permute(permute_order)
    return answer
# %%
if __name__ == "__main__":
    fp = torch.tensor(proj)
    xp = torch.tensor(geo.pfDetGamma)
    x = geo.fDetGammaEqualSpace * (torch.arange(geo.iDetColNum) - geo.iDetColNum/2 + 0.5)
    f = torch_1d_interp(x, xp, fp)
# %%
