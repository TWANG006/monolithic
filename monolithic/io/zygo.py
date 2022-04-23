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
        # cast intensity down to int16, saves memory and Zygo doesn't use cameras >> 16-bit
        try:
            intens_block = list(f['Data']['Intensity'].keys())[0]
            intensity = f['Data']['Intensity'][intens_block][()].astype(np.uint16)
        except (KeyError, OSError):
            intensity = None

        # load phase
        # find the phase array's H5 group
        phase_key = list(f['Data']['Surface'].keys())[0]
        phase_obj = f['Data']['Surface'][phase_key]

        # get a little metadata
        no_data = phase_obj.attrs['No Data'][0]
        wvl = phase_obj.attrs['Wavelength'][0]
        punit = phase_obj.attrs['Unit'][0]
        if isinstance(punit, bytes):
            punit = punit.decode('UTF-8')
        scale_factor = phase_obj.attrs['Interferometric Scale Factor']
        obliquity = phase_obj.attrs['Obliquity Factor']
        # get the phase and process it as required
        phase = phase_obj[()]
        # step 1, flip (above)
        # step 2, clip the nans
        # step 3, convert punit to nm
        phase[phase >= no_data] = np.nan
        if punit == 'Fringes':
            # the usual conversion per malacara
            phase = phase * obliquity * scale_factor * wvl
        elif punit == 'NanoMeters':
            pass
        else:
            raise ValueError(
                "datx file does not use expected phase unit, contact the prysm author with a sample file to resolve"
            )

        # now get attrs
        attrs = f['Attributes']
        key = list(attrs)[-1]
        attrs = attrs[key].attrs
        meta = {}
        for key, value in attrs.items():
            if key.endswith('Unit'):
                continue  # do not need unit keys, units implicitly understood.

            if key.startswith("Data Context."):
                key = key[len("Data Context.") :]

            if key.startswith("Data Attributes."):
                key = key[len("Data Attributes.") :]
            if key.endswith('Value'):
                key = key[:-5]  # strip value from key
            if key.endswith(':'):
                key = key[:-1]
            if key == 'Resolution':
                key = 'lateral_resolution'
            elif key in ['Property Bag List', 'Group Number', 'TextCount']:
                continue  # h5py particulars
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

            meta[key] = value

    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }
