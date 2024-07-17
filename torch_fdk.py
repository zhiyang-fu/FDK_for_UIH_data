import torch
from torch.fft import fft, ifft
import numpy as np
from glob import glob

def nextpow2(n):
    i = 1
    while (2**i) < n:
        i += 1
    return 2 ** i

class FDK(torch.nn.Module):
    def __init__(self, 
                 angles=None, DSD=None, DSO=None,
                 nz=None, nx=None, fov=None, dz=None, 
                 nu=None, nv=None, du=None, dv=None,
                 n_offset_v=0.0, n_offset_z=0.0, n_offset_u = 0.0,
                 geo=None, flat=False,
                 ):
        super().__init__()

        pi = torch.acos(torch.zeros(1)) * 2.0
        self.pi = pi.item()
        self.flat = flat

        if geo is not None:
            if angles is None:
                angles = torch.tensor(geo['pfViewAngle'], dtype=torch.float32)
            DSD = geo['fSDD']
            DSO = geo['fSID']

            if nz is None:
                nz = 1
            if nx is None:
                nx = 512
            if fov is None:
                fov = geo['ScanFoV']
            if dz is None:
                dz = geo['fSliceThickness']

            nu = geo['iDetColNum']
            nv = geo['iDetRowNum']
            du = geo['fDetGammaEqualSpace'] * DSD
            dv = geo['fDetRowEqualSpace']



        self.nz, self.ny, self.nx = nz, nx, nx
        self.dx, self.dy, self.dz = fov/nx, fov/nx, dz

        self.nv, self.nu = nv, nu
        self.dv, self.du = dv, du
        
        self.DSD, self.DSO = DSD, DSO

        self.angles = angles
        self.num_angles = len(angles)
        self.register_buffer("cos", torch.tensor(np.cos(self.angles), dtype=torch.float))
        self.register_buffer("sin", torch.tensor(np.sin(self.angles), dtype=torch.float))

        self.sx = self.dx * self.nx
        self.sy = self.dy * self.ny
        self.sz = self.dz * self.nz
        self.su = self.du * self.nu
        self.sv = self.dv * self.nv

        self.vol_scale = (self.dy*self.dx*self.dz)/ (self.du*self.dv)
        self.adj_scale = self.vol_scale * (self.DSD/self.DSO)
        self.adj_scale *= self.dx/self.du


        self.offset_v = n_offset_v * self.sv 
        self.offset_z = n_offset_z * self.sz 
        self.n_offset_u = n_offset_u

        self.register_buffer("xs",
        (torch.arange(self.nx)-self.nx/2+0.5)*self.dx)
        self.register_buffer("ys",
        (torch.arange(self.ny)-self.ny/2+0.5)*self.dy)
        self.register_buffer("zs",
        (torch.arange(self.nz)-self.nz/2+0.5)*self.dz + self.offset_z)

        # n_offset_u in self.us only influences pre_weighting; ends up CT value offset in image
        self.register_buffer("us",
        (torch.arange(self.nu)-self.nu/2+0.5 + self.n_offset_u)*self.du)
        self.register_buffer("vs",
        (torch.arange(self.nv)-self.nv/2+0.5)*self.dv + self.offset_v)

        self.hsx = self.dx * (self.nx/2-0.5)
        self.hsy = self.dy * (self.ny/2-0.5)
        self.hsz = self.dz * (self.nz/2-0.5)
        self.hsu = self.du * (self.nu/2-0.5)
        self.hsv = self.dv * (self.nv/2-0.5)

        #scaling
        self.inv_scale = 2*self.pi/self.num_angles
        self.inv_scale /= self.adj_scale
        self.inv_scale /= (self.DSD/self.DSO)

        #pre-calculate weight and filter
        self.nfft = max(64, nextpow2(self.nu*2))
        self.register_buffer("f", self._make_ramlak_filter())

        #pre-calculate FOV radius
        if self.flat:
            # fan angle
            max_gamma = torch.atan( torch.tensor(self.hsu / self.DSD) )
        else:
            max_gamma = torch.tensor([self.hsu/self.DSD])

        self.max_radius = self.DSO * torch.sin(max_gamma).item()

    def forward(self,x):
        return self._dot(x)

    def fp(self,x):
        #input: B,nz,nx,ny
        #output: B,na,nv,nu
        return self._dot(x)

    def bp(self,y):
        #input: B,na,nv,nu
        #output: B,nz,nx,ny
        w = self._calc_preweight()
        x = self._adj(y*w)
        return x

    def inv(self,y):
        w = self._calc_preweight()
        yf = self._filter_sinogram(y*w)
        x = self._adj(yf)
        return x * self.inv_scale

    def _make_ramlak_filter(self):
        n = self.nfft

        nn = torch.cat([torch.arange(1, n//2+1, 2), torch.arange(n//2-1,0,-2)])
        f = torch.zeros(n)
        f[0] = 0.25
        f[1::2] = -1 / (self.pi*nn).pow(2)

        #fft the filter
        f_ker = 0.5 * fft(f)/self.du**2
        fac = 1.0
        # fac = torch.tensor(
        #         0.5*(1+np.cos(2*np.pi*np.arange(n)/(n-1),dtype=np.float32))
        #         )

        return f_ker * fac

    def _make_shepplogan_filter(self):
        #discreate version in space
        #https://gray.mgh.harvard.edu/attachments/article/166/166_HST_S14_lect1_v2.pdf
        n = self.nfft

        nn = torch.cat([torch.tensor([0]), torch.arange(1, n//2), torch.tensor([-n//2]), torch.arange(n//2-1,0,-1)])
        f = -2 / ( (self.pi).pow(2) * (4* nn.pow(2)-1) )

        #fft the filter
        f_ker = 0.5 * fft(f)/self.du**2
        fac = 1.0
        return f_ker * fac


    def _calc_preweight(self):
        """cosine preweighting"""
        [uu, vv] = torch.meshgrid(self.us, self.vs, indexing='xy')
        if self.flat:
            w = self.DSD / torch.sqrt( self.DSD ** 2 + uu ** 2 + vv ** 2)
        else:
            w = torch.cos(uu/self.DSD) * self.DSD/torch.sqrt(self.DSD ** 2 + vv ** 2)
        return w

    def _filter_sinogram(self,sino):
        pad = self.nfft - self.nu
        if sino.ndim == 3:
            sino = sino.unsqueeze(0)
            B = 1
        elif sino.ndim == 4:
            B = sino.shape[0]
        else:
            raise ValueError('support input dimensions are 3 or 4')
        # sinof = torch.empty(B,self.num_angles,self.nv,self.nu,device=self.device)
        for ia in range(self.num_angles):
            sino_pad = torch.nn.functional.pad(
                sino[:,ia:ia+1,...],
                (pad//2, pad-pad//2)
            )

            sino_fft = fft(sino_pad)
            sino_fft = sino_fft * self.f
            filtered_sino = torch.real(ifft(sino_fft))

            sino[:,ia:ia+1,...] = filtered_sino[..., pad//2:-(pad-pad//2)]
        return sino*self.du

    def _dot(self, volume):
        #volume
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
            B = 1
        elif volume.ndim == 4:
            #volume shape: B,nz,ny,nx
            B = volume.shape[0]
        else:
            raise AssertionError

        #B,nz,nx,ny -> 1,B*nz,nx, ny
        volume = volume.reshape(1,-1,self.nx,self.ny)
        #B*na,nv,nu
        projections = torch.zeros(self.num_angles,B,self.nv,self.nu,
        device=self.f.device)

        xx,yy = torch.meshgrid(self.xs/self.hsx, self.ys/self.hsy, indexing='xy') #size = (ny,nx)

        uu,vv = torch.meshgrid(self.us, self.vs, indexing='xy') # size = (nv,nu)
        dist = torch.sqrt((self.DSD)**2 + uu ** 2 + vv ** 2)/self.DSD *self.dy

        for ia in range(self.num_angles):
            #theta -> theta+pi/2
            cs, sn = -self.sin[ia], self.cos[ia]

            rx = -xx * sn + yy * cs
            ry =  xx * cs + yy * sn

            #Input: na, B*nz, nx, ny
            #Grid: na, ny,nx,2
            #Output: na, B*nz, ny, nx
            vol = torch.nn.functional.grid_sample(
                volume,
                torch.stack([rx,ry],dim=-1).unsqueeze(0),
                mode='bilinear',
                align_corners=True).squeeze(0)

            #B*nz,ny,nx -> ny, B*na, nz,nx
            vol = vol.reshape(B, self.nz, self.ny, self.nx)
            vol = vol.permute(2,0,1,3)

            for iy in range(self.ny):
                Ratio = (self.DSO - self.ys[iy]) / self.DSD
                pu = uu * Ratio
                pv = vv * Ratio
                # normalize to [-1,1]
                pu = pu/self.hsx
                pv -= self.offset_z

                if self.nz == 1:
                    pv = 0*pu
                else:
                    pv = pv / self.hsz

                #interpolate then sum/project along x (i.e., angle=0 deg)
                #Input: ny, B*na, nz, nx
                #grid: ny, nv,nu,2
                #output: ny, B*na, nv, nu
                projections[ia] += torch.nn.functional.grid_sample(
                    vol[iy:iy+1],
                    torch.stack([pu,pv],dim=-1).unsqueeze(0),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)

        #swap na and B
        projections = projections.transpose(0,1)
        return projections * dist

    def _adj(self,projections):
        if projections.ndim == 3:
            projections = projections.unsqueeze(0)
            B = 1
        elif projections.ndim == 4:
            #projections shape: B,nv,na,nu
            B = projections.shape[0]
        else:
            raise AssertionError
        volume = torch.zeros(self.nz, B, self.nx, self.ny, dtype=torch.float32,
        device=self.f.device)

        #B,na,nv,nu -> na,B,nv,nu
        projections = projections.transpose(0,1)

        # negative yy such that image recon is flipped left to right
        [xx,yy] = torch.meshgrid(self.xs,-self.ys,indexing='ij') #size = (nx, ny)

        for ia in range(self.num_angles):
            cs, sn = self.cos[ia], self.sin[ia]

            rx = -xx * sn + yy * cs
            ry =  xx * cs + yy * sn

            if self.flat:
                # data weighting for bp
                Ratio = self.DSD / (self.DSO-ry)
                Ratio2 = Ratio ** 2 # precompute
                pu = Ratio * rx
                #detector offset
                pu -= self.n_offset_u * self.du
            else:
                # data weighting for bp
                Ratio2 = self.DSD ** 2 / ((self.DSO-ry) ** 2 + rx ** 2)
                Ratio = torch.sqrt(Ratio2) # precompute

                gamma_offset = self.n_offset_u * self.du / self.DSD

                pu = self.DSD * (torch.atan(rx/(self.DSO-ry)) - gamma_offset)
            
            #rescale to [-1,1]
            pu = pu / self.hsu

            for iz in range(self.nz):
                pv = Ratio * self.zs[iz]
                pv -= self.offset_v
                #rescale to [-1,1]
                if self.nv == 1:
                    pv = 0*pu # won't be used
                else:
                    pv = pv / self.hsv

                #interp then sum along angles
                #Input: nz*na,B,nv,nu (nz=1,na=1)
                #Grid: nz*na,nx,ny,2
                #Output: nz*na,B,nx,ny
                volume[iz] += torch.nn.functional.grid_sample(
                        projections[ia:ia+1],
                        torch.stack([pu,pv],dim=-1).unsqueeze(0),
                        mode='bilinear',
                        align_corners=True
                        )[0] * Ratio2
                #1,B,nx,ny * Ratio nx,ny, scale then sum along z
                # volume[iz] += vol* Ratio**2

        #nz,B,nx,ny -> B,nz,nx,ny
        volume = volume.transpose(0,1)   #swap B and nz
        #nz of B,nx,ny
        # volume = torch.stack(volume,dim=1)
        # apply mask
        # mask = self.xs[:,None] ** 2 + self.ys[None,:] ** 2 < self.max_radius ** 2
        # volume = volume * mask

        return volume * self.adj_scale

# %%
# import nvidia_smi
# def print_gpu_usage(i=3):
#     nvidia_smi.nvmlInit()
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     print("Device {}: {}, Memory : ({:.2f}% free): {:.2f}(total), {:.2f} (free), {:.2f} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/1024**2/1000, info.free/1024**2/1000, info.used/1024**2/1000))
#     nvidia_smi.nvmlShutdown()

# %%
if __name__ == "__main__":
    # %%
    print_gpu_usage(4)