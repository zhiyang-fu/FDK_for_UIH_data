import numpy as np
import math
from math import ceil
from copy import deepcopy

PI = 3.14159265358979

class TestStruct:
    def __init__(self, ScanR, DistD, YL, ZL, dectorYoffset, dectorZoffset, 
                 XOffSet, YOffSet, ZOffSet, phantomXOffSet, phantomYOffSet, phantomZOffSet, 
                 DecFanAng, DecHeight, DecWidth, dx, dy, dz, h, BetaS, BetaE, AngleNumber,
                N_2pi, Radius, RecSize, RecSizeZ, delta, HSCoef, k1, GF, RecIm):
        self.ScanR = ScanR
        self.DistD = DistD
        self.YL = YL
        self.ZL = ZL
        self.dectorYoffset = dectorYoffset
        self.dectorZoffset = dectorZoffset
        self.XOffSet = XOffSet
        self.YOffSet = YOffSet
        self.ZOffSet = ZOffSet
        self.phantomXOffSet = phantomXOffSet
        self.phantomYOffSet = phantomYOffSet
        self.phantomZOffSet = phantomZOffSet
        self.DecFanAng = DecFanAng
        self.DecHeight = DecHeight
        self.DecWidth = DecWidth
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.h = h
        self.BetaS = BetaS
        self.BetaE = BetaE
        self.AngleNumber = AngleNumber
        self.N_2pi = N_2pi
        self.Radius = Radius
        self.RecSize = RecSize
        self.RecSizeZ = RecSizeZ
        self.delta = delta
        self.HSCoef = HSCoef
        self.k1 = k1
        self.GF = GF
        self.RecIm = RecIm

def mapConfigVariablesToHelical(cfg):

    
    sid = cfg.scanner.sid # source to isocenter distance, in mm
    sdd = cfg.scanner.sdd # source to detector distance, in mm
    YL = int(cfg.scanner.detectorColCount)
    ZL = int(cfg.scanner.detectorRowCount) #
    N_2pi =  cfg.protocol.viewsPerRotation #
    ViewN = cfg.protocol.viewCount
    N_Turn = (cfg.protocol.viewCount-1)/cfg.protocol.viewsPerRotation
    h = cfg.protocol.tableSpeed*cfg.protocol.rotationTime # in xcist, in unit of mm/s
    startAngle = cfg.protocol.startAngle
    dectorYoffset = -cfg.scanner.detectorColOffset   # coloffset is in unit of cols, could be float
    dectorZoffset = cfg.scanner.detectorRowOffset

    # The following lines are used to define the reconstruction paramters
    k1 = 5  # The order to define the 3D weighting function
    delta = 60  # The range to define smoothness of 2D weigthing function
    HSCoef = 0.6  # This is used to define the half-scan range

    # nMod = ceil(cfg.scanner.detectorColCount/cfg.scanner.detectorColsPerMod)
    rowSize = cfg.scanner.detectorRowSize  # size of 1 detector row in unit of mm, float
    ColSize = cfg.scanner.detectorColSize  # size of 1 detector col in unit of mm, float

    imageSize = cfg.recon.imageSize # how many pixels, currently a single int, has to be square
    sliceCount = cfg.recon.sliceCount # how many slices in recon for 3d recon
    sliceThickness = cfg.recon.sliceThickness # float in mm
    objR =0.5* cfg.recon.fov #FOV in mm

    kernelType = cfg.recon.kernelType
    centerOffset = deepcopy(cfg.recon.centerOffset)
    # Pass desired X as Y
    centerOffset[1] = deepcopy(cfg.recon.centerOffset[0])
    # Pass desired Y as X
    centerOffset[0] = -deepcopy(cfg.recon.centerOffset[1])
    # kernelType = cfg.recon.kernelType
    # centerOffset = deepcopy(cfg.recon.centerOffset)
    # # Pass desired X as Y
    # centerOffset[1] = deepcopy(cfg.recon.centerOffset[0])
    # # Pass desired Y as -X
    # centerOffset[0] = -deepcopy(cfg.recon.centerOffset[1])

    return  sid, sdd, YL, ZL, ViewN, N_Turn, N_2pi, h, startAngle, dectorYoffset, dectorZoffset, \
            k1, delta, HSCoef, rowSize, ColSize, imageSize,sliceCount, sliceThickness, centerOffset, objR, kernelType

def helical_equiAngle(cfg, prep):
    prep = prep[:,:,::-1]

    if cfg.protocol.rotationDirection == -1:	
        prep = prep[::-1,:,:]
    Proj = prep.transpose(0,2,1)

    # scanner & recon geometry
    SO, DD, YL, ZL, ViewN, N_Turn, N_2pi, h, startAngle, dectorYoffset, dectorZoffset, \
    k1, delta, HSCoef, rowSize, ColSize, imageSize, sliceCount, sliceThickness, centerOffset, ObjR,kernelType \
        = mapConfigVariablesToHelical(cfg)

    # print(SO, DD, YL, ZL, ViewN, N_Turn, N_2pi, h, dectorYoffset, dectorZoffset, imageSize, sliceCount, ObjR, ColSize)
    
    DecAngle = math.atan(ColSize/2/DD)*2*YL
    HSCoef = (DecAngle/PI+0.52)
    DecHeight = rowSize * ZL
    h = h/DecHeight
    dx = 2*ObjR/imageSize
    dy = 2*ObjR/imageSize
    dz = sliceThickness

    YLCW = (YL-1)*0.5
    YLC= (YL-1)*0.5 - dectorYoffset  # Detector center along the horizontal direction of detector array
    ZLC= (ZL-1)*0.5 - dectorZoffset # Detector center along the vertical direction of detector array
    
    BetaS = -N_Turn*PI  + PI*startAngle/180
    BetaE =  N_Turn*PI  + PI*startAngle/180

    DecWidth = math.tan(DecAngle*0.5)*(SO)*2 #at iso center
    dYL   =  DecWidth/YL
    dZL   =  DecHeight/ZL
    DeltaFai= (BetaE-BetaS)/(ViewN-1)
    DeltaTheta = DeltaFai
    DeltaT     = dYL  
    dYA = DecAngle/YL
    PProj    = np.zeros((ViewN,YL,ZL))

    ## rebinning the projection
    print("* Rebinning the projection...")
    for i in range(ViewN):
        Theta=(i)*DeltaTheta                   # the view for the parallel projection
        for j in range(YL):
            t      = (j-YLCW)*DeltaT     # the distance from origin to ray for parallel beam
            Beta   = math.asin(t/(SO))            # the fan_angle for cone_beam projection
            Fai    = Theta+Beta              # the view for cone_beam projecton
            a      = Beta  # the position of this ray on the flat detector
            FaiIndex        =  (Fai/DeltaFai)
            UIndex          =  (a/dYA)+YLC
            FI              =  math.ceil(FaiIndex)
            UI              =  math.ceil(UIndex)
            coeXB           =  FI-FaiIndex
            coeXU           =  1-coeXB
            coeYB           =  UI-UIndex
            coeYU           =  1-coeYB
            if (FI<=0):
                IndexXU = 0
                IndexXB = 0
            elif(FI > ViewN-1):
                IndexXU = ViewN-1
                IndexXB = ViewN-1
            else:
                IndexXU = FI
                IndexXB = FI-1

            if (UI<=0):
                IndexYU = 0
                IndexYB = 0
            elif(UI>YL-1):
                IndexYU = YL-1
                IndexYB = YL-1
            else:
                IndexYU=UI
                IndexYB=UI-1
            PProj[i,j,:]=coeXB*coeYB*Proj[IndexXB,IndexYB,:]+ \
                        coeXU*coeYB*Proj[IndexXU,IndexYB,:]+ \
                        coeXB*coeYU*Proj[IndexXB,IndexYU,:]+ \
                        coeXU*coeYU*Proj[IndexXU,IndexYU,:]

    Proj = PProj.transpose(1,2,0)

    #scio.savemat('testrebin.mat', {'rebin': PProj})

    #Perform Ramp filtering
    print("* Applying the filter...")
    Dg=Proj
    nn = int(math.pow(2, (math.ceil(math.log2(abs(YL))) + 1)))
    nn2 = nn*2
    FFT_F = createHSP(nn, kernelType)

    GF = Proj
    
    for ProjIndex in range(0, ViewN):
        for j in range(ZL):
            TempData = np.ones(YL)
            for k in range(YL):
                TempData[k] = Dg[k, j, ProjIndex]
            FFT_S = np.fft.fft(TempData, nn2)
            TempData = np.fft.ifft(FFT_S * FFT_F).imag
            for k in range(YL):
                GF[k, j, ProjIndex] = TempData[k]
    GF = GF/dYL

    #Backproject the filtered data into the 3D space
    # Load the compiled library
    recon = load_C_recon_lib()
    # Define arguments of the C function
    recon.fbp.argtypes = [ct.POINTER(TestStruct)]
    # Define the return type of the C function
    recon.fbp.restype = None

    # init the struct
    t = TestStruct()

    t.ScanR = SO
    t.DistD = DD
    t.YL = YL
    t.ZL = ZL
    t.DecFanAng = DecAngle
    t.DecHeight = DecHeight
    t.DecWidth = DecWidth
    t.dx = dx
    t.dy = dy 
    t.dz = dz
    t.h = h
    t.BetaS = BetaS
    t.BetaE = BetaE
    t.dectorYoffset = dectorYoffset
    t.dectorZoffset = dectorZoffset
    t.AngleNumber = ViewN
    t.N_2pi = N_2pi
    t.Radius = ObjR
    t.RecSize = imageSize
    t.RecSizeZ = sliceCount
    t.delta = delta
    t.HSCoef = HSCoef
    t.k1 = k1

    t.XOffSet = centerOffset[0]
    t.YOffSet = centerOffset[1]
    t.ZOffSet = centerOffset[2]
    t.phantomXOffSet = 0
    t.phantomYOffSet = 0
    t.phantomZOffSet = 0

    if cfg.recon.printReconParameters:
        print("* Reconstruction parameters:")
        print("* SID: {} mm".format(t.ScanR))
        print("* SDD: {} mm".format(t.DistD))
        print("* Fan angle: {} degrees".format(t.DecFanAng))
        # print("* Start view: {}".format(t.startAngle))
        print("* Number of detector cols: {}".format(t.YL))
        print("* Number of detector rows: {}".format(t.ZL))
        print("* Detector height: {} mm".format(t.DecHeight))
        print("* Detector X offset: {} mm".format(t.dectorYoffset))
        print("* Detector Z offset: {} mm".format(t.dectorZoffset))
        print("* Scan number of views: {} ".format(t.AngleNumber))
        print("* Recon FOV: {} mm".format(2 * t.Radius))
        print("* Recon XY pixel size: {} mm".format(t.RecSize))
        print("* Recon Slice thickness: {} mm".format(t.sliceThickness))
        print("* Recon X offset: {} mm".format(t.XOffSet))
        print("* Recon Y offset: {} mm".format(t.YOffSet))
        print("* Recon Z offset: {} mm".format(t.ZOffSet))
    # Generate a 3D ctypes array from numpy array
    print("* Converting projection data from a numpy array to a C array...")
    GF_ptr = double3darray2pointer(GF)
    t.GF = GF_ptr

    # RecIm = np.zeros(shape=(t.RecSize, t.RecSize, t.RecSize))
    print("* Allocating a C array for the recon results...")
    RecIm = np.zeros(shape=(t.RecSize, t.RecSize, t.RecSizeZ))
    RecIm_ptr = double3darray2pointer(RecIm)
    t.RecIm = RecIm_ptr

    # interface with C function
    print("* In C...")
    recon.fbp(ct.byref(t))

    # Convert ctypes 3D arrays to numpy arrays
    print("* Converting the recon results from a C array to a numpy array...")
    rec = double3dpointer2array(RecIm_ptr, *RecIm.shape)
    rec = rec.transpose(1,0,2).astype(np.float32)
    rec = rec[::-1,::-1]

    return rec

def fbp(t):
    ScanR = t.ScanR
    DistD = t.DistD
    DecL = t.DecFanAng
    YL = t.YL
    ZL = t.ZL
    DecHeight = t.DecHeight
    DecWidth = t.DecWidth
    h1 = t.h
    ObjR = t.Radius
    RecSize = t.RecSize
    RecSizeZ = t.RecSizeZ
    delta = t.delta
    HSCoef = t.HSCoef
    k1 = t.k1

    BetaS = t.BetaS
    N_2pi = t.N_2pi
    PN = t.AngleNumber
    dx = t.dx
    dy = t.dy
    dz = t.dz
    dectorYoffset = t.dectorYoffset
    dectorZoffset = t.dectorZoffset
    XOffSet = t.XOffSet
    YOffSet = t.YOffSet
    ZOffSet = t.ZOffSet
    XN = RecSize
    XNC = (XN-1)*0.5
    YN = RecSize
    YNC = (YN-1)*0.5
    ZN = RecSizeZ
    ZNC = (ZN-1)*0.5
    h = h1*DecHeight

    dYL= DecL/YL
    dZL= DecHeight/(ZL)
    YLC = (YL-1)*0.5
    ZLC = (ZL-1)*0.5 + dectorZoffset

    RadiusSquare= ObjR*ObjR
    DeltaFai = 2*pi/N_2pi
    N_pi = N_2pi/2

    dYL = DecWidth/YL
    dZL = DecHeight/ZL
    DeltaFai = 2*pi/N_2pi

    w = [0]*N_2pi

    for zi in range(ZN):
        print("   recon slice %d/%d..." % (zi, ZN))
        z = (zi-ZNC) * dz+ZOffSet
        Beta0 = 2 * pi * z / h
        s0 = math.ceil((Beta0-BetaS) / DeltaFai-0.5)
        s1 = s0-math.ceil(N_pi*HSCoef)
        s2 = s0+math.ceil(N_pi*HSCoef)-1

        if ((s1<PN)or(s2>0)):
            if (s1 < 0):
                s1 = 0
            if (s2 > PN-1):
                s2 = PN-1

            L = s2-s1+1
            Shift = N_pi - (s0-s1)

            if (L<2*delta):
                for k in range(L):
                    w[k+Shift]= math.pow(math.cos((pi/2)*(2*k-L+1)/L),2)
            else:
                for k in range(L):
                    if (0 <= k and k<delta):
                        w[k+Shift]= math.pow(math.cos((pi/2)*(delta-k-0.5)/delta),2)
                    elif(L-delta<=k and k < L):
                        w[k+Shift]= math.pow(math.cos((pi/2)*(k-(L-delta)+0.5)/delta),2)
                    else:
                        w[k+Shift] = 1

            for ProjInd in range(s1, s2+1):
                View = BetaS + ProjInd * DeltaFai
                d1 = N_pi-(s0-ProjInd)
                if (ProjInd < s0):
                    d2 = d1+N_pi
                else:
                    d2 = d1-N_pi

                for yi in range(YN):
                    y = -(yi-YNC)*dy-YOffSet
                    for xi in range(XN):
                        x = -(xi-XNC)*dx-XOffSet
                        UU = -x*math.cos(View)-y*math.sin(View)
                        Yr = -x*math.sin(View)+y*math.cos(View)
                        Zr = (z-h*(View+math.asin(Yr/ScanR))/(2.0*pi))*(DistD)/(math.sqrt(ScanR*ScanR-Yr*Yr)+UU)
                        U1 = Yr/dYL+YLC
                        U = math.ceil(U1)
                        V1 = Zr/dZL+ZLC
                        V = math.ceil(V1)
                        Dey = U-U1
                        Dez = V-V1

                        if ((U>0)and(U<YL)and(V>0)and(V<ZL)):
                            touying = Dey*Dez*t.GF[U-1][V-1][ProjInd] + Dey*(1-Dez)*t.GF[U-1][V][ProjInd] + (1-Dey)*Dez*t.GF[U][V-1][ProjInd] + (1-Dey)*(1-Dez)*t.GF[U][V][ProjInd]
                            weight1 = w[d1]
                            weight2 = w[d2]
                            Gama = abs((z-h*View/(2.0*pi))/(math.sqrt(ScanR*ScanR-Yr*Yr)+UU))
                            if (ProjInd < s0):
                                Gama_C = abs((z-h*(View+pi)/(2.0*pi))/(math.sqrt(ScanR*ScanR-Yr*Yr)-UU))
                            else:
                                Gama_C = abs((z-h*(View-pi)/(2.0*pi))/(math.sqrt(ScanR*ScanR-Yr*Yr)-UU))
                            m1 = pow(Gama,  k1)
                            m2 = pow(Gama_C, k1)
                            weight = (weight1*m2)/(weight2*m1+weight1*m2)
                            t.RecIm[yi][xi][zi] += weight*touying*DeltaFai

