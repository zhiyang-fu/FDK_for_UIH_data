# %%
import os, sys
import numpy as np
import pydicom
import pydicom._storage_sopclass_uids

def write_array_to_dicom(data_dir, image3d, tilt=0,
                    slice_thickness=None, slice_increment=None, 
                    mA=None, kvp=None, fov=None):
    # slice_thickness: acq.SliceThickness.item()
    # slice_increment: user defined or acq.SliceThickness.item()
    # mA: np.floor(np.mean(bbox.mA))
    # kvp: acq.kV.item()
    # fov: user defined
    image3d[image3d<0] = 0
    image3d += 24 #DICOM min is -1024
    image3d = np.uint16(image3d)
    parent_dir = '/home/dpa/mount/D03/Staff/Zhiyang.Fu/DSCT/DICOM_files/Tmp'
    if data_dir[0] != '/':
        data_dir = os.path.join(parent_dir, data_dir) 
    os.makedirs(data_dir, exist_ok=True)
    # Populate required values for file meta information
    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    # meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    meta.FileMetaInformationGroupLength = 190
    meta.FileMetaInformationVersion = b'\x00\x01'
    meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    meta.ImplementationVersionName = 'Python ' + sys.version[0:5]

    ds = Dataset()
    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    # ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    tilt = tilt * np.pi / 180
    ds.ImageOrientationPatient = [0,0,0,0,np.cos(tilt),-np.sin(tilt)]
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.Rows = image3d.shape[-2]
    ds.Columns = image3d.shape[-1]

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1

    ds.RescaleIntercept = -1024
    ds.RescaleType = 'HU'
    ds.RescaleSlope = 1

    ds.PatientName = 'PyTorch FDK'
    # ds.SeriesDescription = ('%s x %s')%(acq.SliceThickness, 0.5)
    if slice_thickness is not None:
        if slice_increment is None:
            slice_increment = slice_thickness
        ds.SeriesDescription = ('%s x %s')%(slice_thickness, slice_increment)
    if kvp is not None:
        ds.KVP = kvp
    if fov is not None:
        ds.ReconstructionDiameter = fov
        ds.PixelSpacing = '{}\{}'.format(fov/ds.Rows,fov/ds.Columns)
    if mA is not None:
        ds.XRayTubeCurrent = mA

    for i in range(len(image3d)):
        ds.file_meta = meta
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = r"0\0\{}".format(i)
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

        ds.PixelData = image3d[i].tobytes()

        ds.save_as(os.path.join(data_dir, '{:02d}.dcm'.format(i)))
    print("saved {} dicoms to {}".format(len(image3d),data_dir))