"""This sub-module implements the I/O for the Zygo metrology data formats.

Supported file formats:
    .datx
    .dat

Credits:
    https://gist.github.com/ccffb1e84df065a690e554f4b40cfd3a.git
    https://github.com/brandondube/prysm.git

"""

import struct
from typing import Dict

import h5py
import numpy as np

ZYGO_INVALID_PHASE = 2147483640
"""int: Value representing the invalid phase."""

ZYGO_ENC = 'utf-8'  # may be ASCII, cp1252...
"""str: Encoding for the byte."""

ZYGO_PHASE_RES_FACTORS = {0: 4096, 1: 32768, 2: 131072}
"""dict: Phase resolution factors for Zygo."""


def read_zygo_dat(file_name: str) -> Dict:
    """Read the Zygo `.dat` (Binary) file.

    Args:
        file_name (str): the file name of the `.dat` file.

    Returns:
        (dict): dict containing phase, intensity, meta

    """
    pass


def _read_zygo_dat_meta(file_contents: bytes) -> Dict:
    """Read the meta data from the Zygo `.dat` (Binary) file.
    Args:
        file_contents (bytes): the file contents read from `fid.read()`

    Returns:
        (dict): dict containing all the meta data

    """
    # convenient single character name
    IB16 = '>H'
    # IL16 = '<H'
    IB32 = '>I'
    # IL32 = '<I'
    FB32 = '>f'
    # FL32 = '<f'
    # C = 'c'
    # uint8 = 'B'
    WASTE_BYTE = '\x00'
    meta = {}

    # begin to read the meta data
    meta['magic_number'] = struct.unpack(IB32, file_contents[:4])[0]
    meta['header_format'] = struct.unpack(IB16, file_contents[4:6])[0]
    meta['header_size'] = struct.unpack(IB32, file_contents[6:10])[0]
    # verify the combination of the above three values
    if not (
        meta['magic_number'] == 0x881B036F
        and meta['header_format'] == 1
        and meta['header_size'] == 834
        or meta['magic_number'] == 0x881B0370
        and meta['header_format'] == 2
        and meta['header_size'] == 834
        or meta['magic_number'] == 0x881B0371
        and meta['header_format'] == 3
        and meta['header_size'] == 4096
    ):
        raise ValueError("Invalid combination of the magic_number, header_format and header_size.")
    meta['swinfo_type'] = struct.unpack(IB16, file_contents[10:12])[0]
    meta['swinfo_date'] = file_contents[12:42].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['swinfo_vers_maj'] = struct.unpack(IB16, file_contents[42:44])[0]
    meta['swinfo_vers_min'] = struct.unpack(IB16, file_contents[44:46])[0]
    meta['swinfo_vers_bug'] = struct.unpack(IB16, file_contents[46:48])[0]
    # read intensity-related meta data
    meta['ac_org_x'] = struct.unpack(IB16, file_contents[48:50])[0]
    meta['ac_org_y'] = struct.unpack(IB16, file_contents[50:52])[0]
    meta['ac_width'] = struct.unpack(IB16, file_contents[52:54])[0]
    meta['ac_height'] = struct.unpack(IB16, file_contents[54:56])[0]
    meta['ac_n_buckets'] = struct.unpack(IB16, file_contents[56:58])[0]
    meta['ac_range'] = struct.unpack(IB16, file_contents[58:60])[0]
    meta['ac_n_bytes'] = struct.unpack(IB32, file_contents[60:64])[0]
    # read phase-related meta data
    meta['cn_org_x'] = struct.unpack(IB16, file_contents[64:66])[0]
    meta['cn_org_y'] = struct.unpack(IB16, file_contents[66:68])[0]
    meta['cn_width'] = struct.unpack(IB16, file_contents[68:70])[0]
    meta['cn_height'] = struct.unpack(IB16, file_contents[70:72])[0]
    meta['cn_n_bytes'] = struct.unpack(IB32, file_contents[72:76])[0]
    # read others
    meta['time_stamp'] = struct.unpack(IB32, file_contents[76:80])[0]
    meta['comment'] = file_contents[80:162].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['source'] = struct.unpack(IB16, file_contents[162:164])[0]
    meta['intf_scale_factor'] = struct.unpack(FB32, file_contents[164:168])[0]
    meta['wavelength_in'] = struct.unpack(FB32, file_contents[168:172])[0]
    meta['num_aperture'] = struct.unpack(FB32, file_contents[172:176])[0]
    meta['obliquity_factor'] = struct.unpack(FB32, file_contents[176:180])[0]
    meta['magnification'] = struct.unpack(FB32, file_contents[180:184])[0]
    meta['lateral_res'] = struct.unpack(FB32, file_contents[184:188])[0]
    meta['acq_type'] = struct.unpack(IB16, file_contents[188:190])[0]
    meta['intens_avg_cnt'] = struct.unpack(IB16, file_contents[190:192])[0]
    meta['ramp_cal'] = struct.unpack(IB16, file_contents[192:194])[0]
    meta['sfac_limit'] = struct.unpack(IB16, file_contents[194:196])[0]
    meta['ramp_gain'] = struct.unpack(IB16, file_contents[196:198])[0]

    return {}


def read_zygo_datx(file_name: str) -> Dict:
    """Read the Zygo `.datx` (HDF5) file.

    Args:
        file_name (str): the file name of the `.datx` file.

    Returns:
        (dict): dict containing phase, intensity, meta

    """
    with h5py.File(file_name, 'r') as f:
        # 1. read intensity (if presented)
        try:
            # get the key of the intensity dataset
            intensity_key = list(f['Data']['Intensity'].keys())[0]
            # read everything in as numpy.ndarray using `[()]`
            intensity = f['Data']['Intensity'][intensity_key][()].astype(np.uint16)
        except (KeyError, OSError):
            intensity = None

        # 2. read phase
        # get the key of the phase dataset
        phase_key = list(f['Data']['Surface'].keys())[0]
        # read the phase object including the meta data
        phase_obj = f['Data']['Surface'][phase_key]

        # read phase-related meta data
        no_data = phase_obj.attrs['No Data'][0]
        wavelength = phase_obj.attrs['Wavelength'][0]
        scale_factor = phase_obj.attrs['Interferometric Scale Factor']
        obliquity = phase_obj.attrs['Obliquity Factor']

        # get the phase and process it as required, clipo nans and convert to meter
        phase = phase_obj[()]
        phase[phase >= no_data] = np.nan
        phase = phase * obliquity * scale_factor * wavelength

        # 3. get attrs
        attrs = f['Attributes']
        key = list(attrs)[-1]
        attrs = attrs[key].attrs

        # read all the attributes
        meta = {}
        for key, value in attrs.items():
            if key.startswith("Data Context.Data Attributes."):
                key = key[len("Data Context.Data Attributes.") :]
            elif key in ['Property Bag List', 'Group Number', 'TextCount']:
                continue
            if value.dtype == 'object':
                value = value[0]
                if isinstance(value, bytes):
                    value = value.decode('UTF-8')
            elif value.dtype in ['uint8', 'int32']:
                value = int(value[0])
            elif value.dtype in ['float64']:
                value = float(value[0])
            else:
                continue  # compound items, h5py objects that do not map nicely to primitives
            # insert the valid key, value into the meta dict
            meta[key] = value
        # save the lateral resolution unit as
        latral_res_unit = meta['Resolution:Unit']
        latral_res_value = meta['Resolution:Value']
        if latral_res_unit == 'MiliMeters':
            meta['lateral_res'] = latral_res_value * 1e-3
        elif latral_res_unit == 'MicroMeters':
            meta['lateral_res'] = latral_res_value * 1e-6
        elif latral_res_unit == 'NanoMeters':
            meta['lateral_res'] = latral_res_value * 1e-9
        elif latral_res_unit == 'Meters':
            meta['lateral_res'] = latral_res_value
        else:
            raise ValueError(f"Unit {latral_res_unit} is not supported yet in monolithic.")

    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }
