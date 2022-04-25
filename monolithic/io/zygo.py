"""This sub-module implements the I/O for the Zygo metrology data formats.

Supported file formats:
    .datx
    .dat

Credits:
    https://gist.github.com/ccffb1e84df065a690e554f4b40cfd3a.git
    https://github.com/brandondube/prysm.git

"""

from typing import Dict

import h5py
import numpy as np


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
        else:
            meta['lateral_res'] = latral_res_value

    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }
