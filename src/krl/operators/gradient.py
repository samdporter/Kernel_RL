from cil.optimisation.operators import LinearOperator
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gradient(LinearOperator):
    def __init__(self, voxel_sizes, method='forward', bnd_cond='Neumann'):
        self.voxel_sizes = voxel_sizes
        self.method = method
        self.bnd_cond = bnd_cond
    def direct(self, x, out = None):
        x_arr = torch.tensor(x.as_array(), device=device)
        res = []
        for i in range(x.ndim):
            if self.method == 'forward':
                res.append(self.forward_diff(x_arr, i))
            elif self.method == 'backward':
                res.append(self.backward_diff(x_arr, i))
            elif self.method == 'central':
                res.append((self.forward_diff(x_arr, i) + self.backward_diff(x_arr, i)) / 2)
            else:
                raise ValueError('Not implemented')
            if self.voxel_sizes[i] != 1.0:
                res[-1] /= self.voxel_sizes[i]
        result = torch.stack(res, dim=-1)
        if out is None:
            out = x.clone()
        out.fill(result)
        return out

    def adjoint(self, x, out = None):
        res = []
        x_arr = torch.tensor(x, device=device)
        for i in range(x.size(-1)):
            if self.method == 'forward':
                res.append(-self.backward_diff(x_arr[..., i], i))
            elif self.method == 'backward':
                res.append(-self.forward_diff(x_arr[..., i], i))
            elif self.method == 'central':
                res.append((-self.forward_diff(x_arr[..., i], i) - self.backward_diff(x_arr[..., i], i)) / 2)
            else:
                raise ValueError('Not implemented')
            if self.voxel_sizes[i] != 1.0:
                res[-1] /= self.voxel_sizes[i]
        result = torch.stack(res, dim=-1).sum(dim=-1)
        if out is None:
            out = x.clone()
        out.fill(result)
        return out       

    def forward_diff(self, x, direction):
        append_tensor = x.select(direction, 0 if self.bnd_cond == 'Periodic' else -1).unsqueeze(direction)
        out = torch.diff(x, n=1, dim=direction, append=append_tensor)
        if self.bnd_cond == 'Neumann':
            out.select(direction, -1).zero_()
        return out

    def backward_diff(self, x, direction):
        flipped_x = x.flip(direction)
        append_tensor = flipped_x.select(direction, 0 if self.bnd_cond == 'Periodic' else -1).unsqueeze(direction)
        out = -torch.diff(flipped_x, n=1, dim=direction, append=append_tensor).flip(direction)
        if self.bnd_cond == 'Neumann':
            # Left boundary: Set first slice of out to the first slice of x
            out.select(direction, 0).copy_(x.select(direction, 0))
            # Right boundary: Set last slice of out to the negative of the penultimate slice of x
            out.select(direction, -1).copy_(-x.select(direction, -2))
        return out