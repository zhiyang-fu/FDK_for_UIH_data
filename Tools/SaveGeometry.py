import os
import math
import numpy as np
import yaml

from Script.Logger import *
from Toolbox.globalvar import *

from Toolbox.Dict import Dict
from Toolbox.NoiseRecon.Functor.getImgVolume import LoadMat

def Initialize_SaveGeometry(Meta, Para, Blackbox):
    """
    @description: save geometry at any functor, Meta Para, Blackbox are all NOT modified
    [FunctorInfo, Para, Blackbox] = Initialize_SaveGeometry(Meta, Para, Blackbox);
    ---------
    @Meta  : Functor-related variables.
    @Para : Includes AcqPara and Reconpara. AcqPara defines the acquisition/scan parameters such as focal spot bias, focal spot type, detector angle, SID, TotalViewNum, etc. 
            ReconPara defines the reconstruction parameters such as image thickness, FoV, center position, etc.
    @Blackbox  : Variables for each view such as view angle, mA, couch position, etc.
    -------
    """
    FunctorInfo = Dict()
    AcqPara = Para.AcqPara
    if 'path' not in Meta.keys():
        Meta.path = os.getcwd()
    if 'name' not in Meta.keys():
        Meta.name = 'geo.yml'
    if 'type' in Meta.keys():
        if Meta.type == 'SeriesUID':
            Meta.name = AcqPara.SeriesInstanceUID + '.yml'
        elif Meta.type == 'PatientUID':
            Meta.name = AcqPara.PatientUID + '.yml'

    os.makedirs(Meta.path, exist_ok=True)
    FunctorInfo.Meta = Meta
    FunctorInfo.Geo = InitGeo(AcqPara, Blackbox)
    return FunctorInfo, Para, Blackbox

def Process_SaveGeometry(Meta, FunctorInfo, InputData, Para, Blackbox):
    """
    @description :    
    ---------
    @Meta  : Functor-related variables.
    @FunctorInfo : The functor information which initialized in Initialize_AziRebin. It’s a global structure for storing the variables and data as a global memory disk. We can update
             and change the data and variables in it.
    @InputData  : The input data from previous functor.
    @Para : Includes AcqPara and Reconpara. AcqPara defines the acquisition/scan parameters such as focal spot bias, focal spot type, detector angle, SID, TotalViewNum, etc. 
            ReconPara defines the reconstruction parameters such as image thickness, FoV, center position, etc.
    @Blackbox: Variables for each view such as view angle, mA, couch position, etc.
    -------
    """
    geo = FunctorInfo.Geo.obtain_geo()
    geo = dict(geo) #convert to python dict
    #print geo
    # print(yaml.dump(geo, default_flow_style=False))
    # save geo
    with open(os.path.join(FunctorInfo.Meta.path, FunctorInfo.Meta.name), 'w') as f:
        yaml.dump(dict(geo), f) 
    return InputData, Para, Blackbox, FunctorInfo

# def Determine_SaveGeometry(Meta, Index, Size, Type, Para, FunctorInfo):
    # pass

def Destroy_SaveGeometry(Meta, FunctorInfo):
    pass

class InitGeo:
    def __init__(self, acq, bbox):
        geo = Dict()
        geo.SeriesInstanceUID = acq.SeriesInstanceUID


        # Scanner float
        geo.fSDD = acq.SDD.item()
        geo.fSID = acq.SID.item()
        geo.Scanner = self.getScannerType(acq.DMS_type)
        geo.ScanType = acq.ScanType
        geo.fScanFoV = acq.ScanFoV.item()
        geo.fTiltAngle = acq.Tilt
        geo.fAnodeAngle = acq.AnodeAngle.item()
        geo.fCollimatorSliceThickness = acq.CollimatorSliceThickness.item()
        geo.fkVp = acq.kV.item()
        geo.fmA = np.mean(bbox.mA).item()
        geo.fRotateTime = acq.RotateTime.item()
        geo.fmAs = geo.fmA * geo.fRotateTime
        geo.fRotateTime = acq.RotateTime.item()
        geo.fSliceThickness = acq.SliceThickness.item()

        #Scanner integer
        geo.iRotateDir = acq.RotateDir.item()
        geo.iSliceDir = acq.SliceDir.item()
        geo.iChanDir = acq.ChanDir.item()
        geo.iViewPerRevolution = acq.ViewPerRevolution.item()
        geo.iViewNum = len(bbox.ViewAngle)

        # Focal Spot and View Angle
        geo.FirstFlyDir = acq.FirstFlyDir
        geo.iFocalSpotNum = self.getFocalSpotNum(acq.FocalSpotType)
        geo.pfFocalSpotBias = acq.FocalSpotBias
        geo.pfFocalSpotBias_YZ = self.getDFSBias_YZ(geo)
        geo.pfFocalSpotBiasRadian = self.getDFSBias(geo)
        geo.pfViewAngle_bbox = bbox.ViewAngle
        geo.pfViewAngle = self.addDFSViewAngle(geo)

        ## Couch
        geo.iCouchDir = int(acq.CouchDir) if acq.CouchDir != 0 else acq.SliceDir.item()
        geo.pfCouchPosition = bbox.CouchPos
        
        # Detector
        geo.iDetRowNum = acq.SliceNum.item()
        geo.iDetColNum = acq.ChanNum.item()
        geo.bEqualizeDetAngle = True
        #detector angle within the fan, denoted as gamma
        geo.pfDetGamma = self.getDetGamma(acq.DetectorAngle, geo) #focal spot bias included 
        geo.pfDetChannelSpace = self.getDetChannelSpace(geo.Scanner) #arc spacing
        geo.fDetGammaEqualSpace = self.getDetGammaEqualSpace(geo)
        # pfDetGamma and fDetGammaEqualSpace will be used in FDK
        geo.fDetRowEqualSpace = self.getDetRowEqualSpace(geo)

        # geo.pfDetGammaSpace = geo.pfDetChannelSpace / geo.fSDD # fSDD changes for ZDFS
        #detetor angle in the cone angle direction, denoted as alpha
        # geo.pfDetRowCenter = geo.fDetRowEqualSpace * (np.arange(self.iDetRowNum) - self.iDetRowNum//2 + 0.5)
        self.geo = geo

    def obtain_geo(self,):
        # numpy array to list such that can be exported to yml files
        instances = ['pfFocalSpotBias',
                     'pfFocalSpotBias_YZ',
                     'pfFocalSpotBiasRadian',
                     'pfViewAngle',
                     'pfViewAngle_bbox',
                     'pfCouchPosition',
                     'pfDetGamma',
                     'pfDetChannelSpace',
                     ]
        for instance in instances:
            self.geo[instance] = self.geo[instance].tolist()
        return self.geo

    def getScannerType(self, DMS_type):
        if DMS_type == '16':
            strScanner = 'U64'
        elif DMS_type == '0':
            strScanner = 'U16'
        elif DMS_type == '18':
            strScanner = 'U82'
        elif DMS_type == '19':
            strScanner = 'U86'

        return strScanner

    def getFocalSpotNum(self, FocalSpotType):
        if FocalSpotType == 'USFS' or FocalSpotType == 'QSFS':
            iFocalSpotNum = 1  # Focal Spot Num
        elif FocalSpotType in {'XDFS','ZXDFS'}:
            iFocalSpotNum = 2  # Focal Spot Num
        return iFocalSpotNum

    def getDFSBias_YZ(self, geo):
        '''get the DFS bias
        Bias_YZ = [Y1 Y2; Z1 Z2]'''
        iFocalSpotNum = geo.iFocalSpotNum
        if iFocalSpotNum == 1:
            pfFocalSpotBias_Z = geo.pfFocalSpotBias[1]
        elif iFocalSpotNum == 2:
            pfFocalSpotBias_Z = geo.pfFocalSpotBias[[1, 3]]  #[R, L]
            if not (geo.FirstFlyDir == 'R'):
                pfFocalSpotBias_Z = np.array([pfFocalSpotBias_Z[1], pfFocalSpotBias_Z[0]])
        '''
        % Usually AnodeAngle = -7 degree
        % Bias_Z >0 means Bias_Y >0, and realSID = SID - Bias_Y
        %
        %  \
        %   o-----> +Z
        %   |\
        %   | \
        %   |  \
        %   V   \
        %   +Y
        %
        %   * ISOCenter
        
        % !! Notice "-" here !!!
        '''
        pfFocalSpotBias_Y = - pfFocalSpotBias_Z / np.tan(geo.fAnodeAngle*np.pi/180)
        pfFocalSpotBias_YZ = np.vstack([pfFocalSpotBias_Y, pfFocalSpotBias_Z])
        return pfFocalSpotBias_YZ

    def getDFSBias(self, geo):
        '''get the DFS bias'''
        '''
        % Changed by rengcai.yang @20160822
        % Change DFSAngle sign from "-" to "+"
        % Check "$/CT/System/PA/Recon/RIO/Algorithm survey/RIO DFS Angle Check.pptx" for details.
        '''
        iFocalSpotNum = geo.iFocalSpotNum
        if iFocalSpotNum == 1:
            pfFocalSpotBiasOri = +geo.iRotateDir * geo.pfFocalSpotBias[0]
        elif iFocalSpotNum == 2:
            pfFocalSpotBiasOri = +geo.iRotateDir * geo.pfFocalSpotBias[[0, 2]] # [R, L]
            if not (geo.FirstFlyDir == 'R'):
                pfFocalSpotBiasOri = pfFocalSpotBiasOri[-1::-1]

        SID = geo.fSID - geo.pfFocalSpotBias_YZ[0, :]

        pfFocalSpotBiasRadian = np.arctan(pfFocalSpotBiasOri/ SID)

        return pfFocalSpotBiasRadian #, pfFocalSpotBiasOri

    def addDFSViewAngle(self, geo):
        # modified from ChangeViewAngle() function
        # DFS: dynamical focal spot
        # also known as flying focal spot
        fViewAngle_new = np.zeros(geo.iViewNum, dtype=np.float32)
        fViewAngle_bbox = geo.pfViewAngle_bbox
        iFocalSpotNum = geo.iFocalSpotNum

        if iFocalSpotNum == 1:
            fViewAngle_new = fViewAngle_bbox + geo.pfFocalSpotBiasRadian
        elif iFocalSpotNum == 2:
            fViewAngle_new[0::2] = fViewAngle_bbox[0::2] + geo.pfFocalSpotBiasRadian[0]
            fViewAngle_new[1::2] = fViewAngle_bbox[1::2] + geo.pfFocalSpotBiasRadian[1]
        
        # fViewAngle_new[fViewAngle_new<0] = fViewAngle_new[fViewAngle_new<0]  + 2*np.pi
        # fViewAngle_new[fViewAngle_new>2*np.pi] = fViewAngle_new[fViewAngle_new>2*np.pi]  - \
        #     2*np.pi * np.floor(fViewAngle_new[fViewAngle_new>2*np.pi]/(2*np.pi))

        return fViewAngle_new

    # def getViewAngle(self, geo, pfViewAngle_bbox):
    #     #offset view angle to be within [0, 2*pi]
    #     fViewAngle = pfViewAngle_bbox - np.floor(pfViewAngle_bbox/(2*np.pi)) * 2*np.pi
    #     pfViewAngle_bbox = fViewAngle

    #     # if geo.ScanType == 'Axial':
    #         # raise NotImplementedError('Axial')
    #     # else:
    #     fViewAngle = self.addDFSViewAngle(geo, fViewAngle)
    #     return fViewAngle.astype(np.float32), pfViewAngle_bbox

    def getDetGamma(self, pfdetAngle_acq, geo):
        # based on GetChannelCenterPos
        if geo.iFocalSpotNum == 1:
            pfDetGamma = pfdetAngle_acq
        elif geo.iFocalSpotNum ==2:
            pfDetGamma = np.zeros(len(pfdetAngle_acq), dtype=np.float32)
            if geo.FirstFlyDir == 'R':
                pfDetGamma[::2] = pfdetAngle_acq[:geo.iDetColNum]
                pfDetGamma[1::2] = pfdetAngle_acq[geo.iDetColNum:]
            else:
                pfDetGamma[::2] = pfdetAngle_acq[geo.iDetColNum:]
                pfDetGamma[1::2] = pfdetAngle_acq[:geo.iDetColNum]
        return pfDetGamma
    
    def getDetChannelSpace(self, strScanner):
        ## get the channel detector space
        if (strScanner == 'U16'):
            pfDetectorGammaSpace= LoadMat(r'Toolbox/NoiseRecon/Cache/pfU16DetectorGammaSpace.mat')
        elif (strScanner == 'U40'):
            pfDetectorGammaSpace= LoadMat(r'Toolbox/NoiseRecon/Cache/pfU40DetectorGammaSpace.mat')
        elif strScanner == 'U64' or strScanner == 'U80' or strScanner =='U82' or \
            strScanner == 'U86':
            pfDetectorGammaSpace= LoadMat(r'Toolbox/NoiseRecon/Cache/pfDetectorGammaSpace.mat')
        elif (strScanner == 'PCCT_UHR'):
            pfDetectorGammaSpace= LoadMat(r'Toolbox/NoiseRecon/Cache/pfUHRDetectorGammaSpace.mat')
        elif (strScanner == 'PCCT_Macro'):  
            pfDetectorGammaSpace= LoadMat(r'Toolbox/NoiseRecon/Cache/pfMacroDetectorGammaSpace.mat')
            
        return pfDetectorGammaSpace
        # fDetectorChannelSpace = pfDetectorGammaSpace/fSDD# channel space  
        # return   fDetectorChannelSpace
    
    def getDetGammaEqualSpace(self, geo):
        totalDetAngle = abs(geo.pfDetGamma[-1] - geo.pfDetGamma[0]) + \
            (geo.pfDetChannelSpace[-1] + geo.pfDetChannelSpace[0]) / 2 / geo.fSDD
        fDetGammaEqualSpace = totalDetAngle.item()/geo.iDetColNum 
        return fDetGammaEqualSpace

    def getDetRowEqualSpace(self, geo):
        if (geo.strScanner == 'U16'):
            fDetectorRowSpace = (geo.fSliceThickness/0.6-0.085)# row space
        else:
            fDetectorRowSpace = geo.fSliceThickness/(geo.fSID/geo.fSDD)
        return fDetectorRowSpace
