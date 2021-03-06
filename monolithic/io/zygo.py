"""This sub-module implements the I/O for the Zygo metrology data formats.

Supported file formats:
    .datx
    .dat

Credits:
    https://gist.github.com/ccffb1e84df065a690e554f4b40cfd3a.git
    https://github.com/brandondube/prysm.git

"""

import pathlib
import struct
from typing import Dict, Tuple

import h5py
import numpy as np

ZYGO_INVALID_PHASE = 2147483640
"""int: Value representing the invalid phase."""

ZYGO_ENC = 'utf-8'  # may be ASCII, cp1252...
"""str: Encoding for the byte."""

ZYGO_PHASE_RES_FACTORS = {0: 4096, 1: 32768, 2: 131072}
"""dict: Phase resolution factors for Zygo."""


def read_zygo_binary(file_name: str) -> Tuple:
    """Read the Zygo binary file formates (.dat and .datx).

    Args:
        file_name (str): the file name of the binary file.

    Returns:
        (tuple): tuple containing:
            X (numpy.ndarray): x coordinates of the full aperture.
            Y (numpy.ndarray): y coordinates of the full aperture.
            Z (numpy.ndarray): height in the full aperture.
            X_cropped (numpy.ndarray): x coordinates of the cropped aperture.
            Y_cropped (numpy.ndarray): y coordinates of the cropped aperture.
            Z_cropped (numpy.ndarray): height in the cropped aperture.

    """
    # get the file extension and call the respective read function
    file_extension = pathlib.Path(file_name).suffix
    if file_extension == '.dat':
        data = read_zygo_dat(file_name)
    elif file_extension == '.datx':
        data = read_zygo_datx(file_name)
    else:
        raise ValueError(f'{file_extension} is not a valid Zygo binary file extension.')

    # embed phase into the full aperture, if available
    Z_intensity = data['intensity']
    Z_phase = data['phase']

    # only phase is presented
    if Z_intensity is None:
        Z_cropped = Z_phase
        X_cropped, Y_cropped = np.meshgrid(
            np.arange(0, Z_cropped.shape[1], dtype=float), np.arange(0, Z_cropped.shape[0], dtype=float)
        )
        X_cropped = X_cropped * data['meta']['lateral_res']
        Y_cropped = Y_cropped * data['meta']['lateral_res']
        Y_cropped = np.nanmax(Y_cropped) - Y_cropped + np.nanmin(Y_cropped)
        X, Y, Z = X_cropped, Y_cropped, Z_cropped
    # both intensity & phase are presented
    else:
        # assigne phase
        Z_cropped = Z_phase
        # work with the full aperture
        m, n = Z_intensity[0].shape
        Z = np.full((m, n), fill_value=np.nan)
        X, Y = np.meshgrid(np.arange(0, n, dtype=float), np.arange(0, m, dtype=float))
        X = X * data['meta']['lateral_res']
        Y = Y * data['meta']['lateral_res']
        Y = np.nanmax(Y) - Y + np.nanmin(Y)
        # work with clear aperture
        if Z_intensity.shape == Z_phase.shape:
            X_cropped, Y_cropped, Z_cropped = X, Y, Z
        else:
            p_ys = data['meta']['cn_org_x']
            p_xs = data['meta']['cn_org_y']
            p_height = data['meta']['cn_height']
            p_width = data['meta']['cn_width']
            # feed the phase to the full aperture
            Z[p_ys : p_ys + p_height, p_xs : p_xs + p_width] = Z_cropped
            Y_cropped = Y[p_ys : p_ys + p_height, p_xs : p_xs + p_width]
            X_cropped = X[p_ys : p_ys + p_height, p_xs : p_xs + p_width]

    return (X, Y, Z, X_cropped, Y_cropped, Z_cropped)


def read_zygo_dat(file_name: str) -> Dict:
    """Read the Zygo `.dat` (Binary) file.

    Args:
        file_name (str): the file name of the `.dat` file.

    Returns:
        (dict): dict containing phase, intensity, meta

    """
    # open the binary file
    with open(file_name, 'rb') as fid:
        file_contents = fid.read()

    # 1. obtain the meta data
    meta = _read_zygo_dat_meta(file_contents)

    # 2. read intensity, if presented
    intens_width = meta['ac_width']
    intens_height = meta['ac_height']
    intens_buckets = meta['ac_n_buckets']
    intens_buckets = 1 if intens_buckets == 0 else intens_buckets
    intens_size = intens_width * intens_height * intens_buckets

    intensity = None
    if intens_size > 0:
        intensity = np.frombuffer(
            file_contents, offset=meta['header_size'], count=intens_size, dtype=np.uint16
        ).reshape((intens_buckets, intens_height, intens_width))

    # 3. read phase
    phase_width = meta['cn_width']
    phase_height = meta['cn_height']
    phase_size = phase_width * phase_height

    phase = None
    if phase_size > 0:
        phase_raw = np.frombuffer(
            file_contents, offset=meta['header_size'] + intens_size * 2, count=phase_size, dtype=np.int32
        )
        phase = phase_raw.copy().byteswap(True).astype(float).reshape((phase_height, phase_width))
        phase[phase >= ZYGO_INVALID_PHASE] = np.nan
        phase *= (
            meta['scale_factor']
            * meta['obliquity_factor']
            * meta['wavelength']
            / ZYGO_PHASE_RES_FACTORS[meta['phase_res']]
        )

    return {'phase': phase, 'intensity': intensity, 'meta': meta}


def _read_zygo_dat_meta(file_contents: bytes) -> Dict:
    """Read the meta data from the Zygo `.dat` (Binary) file.

    Args:
        file_contents (bytes): the file contents read from `fid.read()`

    Returns:
        (dict): dict containing all the meta data

    """
    # convenient single character name
    IB16 = '>H'
    IL16 = '<H'
    IB32 = '>I'
    IL32 = '<I'
    FB32 = '>f'
    FL32 = '<f'
    C = 'c'
    uint8 = 'B'
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
    meta['scale_factor'] = struct.unpack(FB32, file_contents[164:168])[0]
    meta['wavelength'] = struct.unpack(FB32, file_contents[168:172])[0]
    meta['num_aperture'] = struct.unpack(FB32, file_contents[172:176])[0]
    meta['obliquity_factor'] = struct.unpack(FB32, file_contents[176:180])[0]
    meta['magnification'] = struct.unpack(FB32, file_contents[180:184])[0]
    meta['lateral_res'] = struct.unpack(FB32, file_contents[184:188])[0]
    meta['acq_type'] = struct.unpack(IB16, file_contents[188:190])[0]
    meta['intens_avg_count'] = struct.unpack(IB16, file_contents[190:192])[0]
    meta['ramp_cal'] = struct.unpack(IB16, file_contents[192:194])[0]
    meta['sfac_limit'] = struct.unpack(IB16, file_contents[194:196])[0]
    meta['ramp_gain'] = struct.unpack(IB16, file_contents[196:198])[0]
    meta['part_thickness'] = struct.unpack(FB32, file_contents[198:202])[0]
    meta['sw_llc'] = struct.unpack(IB16, file_contents[202:204])[0]
    meta['target_range'] = struct.unpack(FB32, file_contents[204:208])[0]
    meta['rad_crv_measure_seq'] = struct.unpack(IL16, file_contents[208:210])[0]
    meta['min_mod'] = struct.unpack(IB32, file_contents[210:214])[0]
    meta['min_mod_count'] = struct.unpack(IB32, file_contents[214:218])[0]
    meta['phase_res'] = struct.unpack(IB16, file_contents[218:220])[0]
    meta['min_area'] = struct.unpack(IB32, file_contents[220:224])[0]
    meta['discon_action'] = struct.unpack(IB16, file_contents[224:226])[0]
    meta['discon_filter'] = struct.unpack(FB32, file_contents[226:230])[0]
    meta['connect_order'] = struct.unpack(IB16, file_contents[230:232])[0]
    meta['sign'] = struct.unpack(IB16, file_contents[232:234])[0]
    meta['camera_width'] = struct.unpack(IB16, file_contents[234:236])[0]
    meta['camera_height'] = struct.unpack(IB16, file_contents[236:238])[0]
    meta['sys_type'] = struct.unpack(IB16, file_contents[238:240])[0]
    meta['sys_board'] = struct.unpack(IB16, file_contents[240:242])[0]
    meta['sys_serial'] = struct.unpack(IB16, file_contents[242:244])[0]
    meta['sys_inst_id'] = struct.unpack(IB16, file_contents[244:246])[0]
    meta['obj_name'] = file_contents[246:258].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['part_name'] = file_contents[258:298].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['codev_type'] = struct.unpack(IB16, file_contents[298:300])[0]
    meta['phase_avg_count'] = struct.unpack(IB16, file_contents[300:302])[0]
    meta['sub_sys_err'] = struct.unpack(IB16, file_contents[302:304])[0]
    # 305-320 unused
    meta['part_sn'] = file_contents[320:360].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['refractive_index'] = struct.unpack(FB32, file_contents[360:364])[0]
    meta['remove_tilt'] = struct.unpack(IB16, file_contents[364:366])[0]
    meta['remove_fringes'] = struct.unpack(IB16, file_contents[366:368])[0]
    meta['max_area'] = struct.unpack(IB32, file_contents[368:372])[0]
    meta['setup_type'] = struct.unpack(IB16, file_contents[372:374])[0]
    meta['wrapped'] = struct.unpack(IB16, file_contents[374:376])[0]
    meta['pre_connect_filter'] = struct.unpack(FB32, file_contents[376:380])[0]
    meta['wavelength_in_2'] = struct.unpack(FB32, file_contents[380:384])[0]
    meta['wavelength_fold'] = struct.unpack(IB16, file_contents[384:386])[0]
    meta['wavelength_in_1'] = struct.unpack(FB32, file_contents[386:390])[0]
    meta['wavelength_in_3'] = struct.unpack(FB32, file_contents[390:394])[0]
    meta['wavelength_in_4'] = struct.unpack(FB32, file_contents[394:398])[0]
    meta['wavelength_select'] = file_contents[398:406].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['fda_res'] = struct.unpack(IB16, file_contents[406:408])[0]
    meta['scan_description'] = file_contents[408:428].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['n_fiducials'] = struct.unpack(IB16, file_contents[428:430])
    meta['fiducials'] = [
        struct.unpack(FB32, file_contents[430:434])[0],
        struct.unpack(FB32, file_contents[434:438])[0],
        struct.unpack(FB32, file_contents[438:442])[0],
        struct.unpack(FB32, file_contents[442:446])[0],
        struct.unpack(FB32, file_contents[446:450])[0],
        struct.unpack(FB32, file_contents[450:454])[0],
        struct.unpack(FB32, file_contents[454:458])[0],
        struct.unpack(FB32, file_contents[458:462])[0],
        struct.unpack(FB32, file_contents[462:466])[0],
        struct.unpack(FB32, file_contents[466:470])[0],
        struct.unpack(FB32, file_contents[470:474])[0],
        struct.unpack(FB32, file_contents[474:478])[0],
        struct.unpack(FB32, file_contents[478:482])[0],
        struct.unpack(FB32, file_contents[482:486])[0],
    ]
    meta['pixel_width'] = struct.unpack(FB32, file_contents[486:490])[0]
    meta['pixel_height'] = struct.unpack(FB32, file_contents[490:494])[0]
    meta['exit_pupil_diameter'] = struct.unpack(FB32, file_contents[494:498])[0]
    meta['light_level_percent'] = struct.unpack(FB32, file_contents[498:502])[0]
    meta['coords_state'] = struct.unpack(IL32, file_contents[502:506])[0]
    meta['coords_x_pos'] = struct.unpack(FL32, file_contents[506:510])[0]
    meta['coords_y_pos'] = struct.unpack(FL32, file_contents[510:514])[0]
    meta['coords_z_pos'] = struct.unpack(FL32, file_contents[514:518])[0]
    meta['coords_x_rot'] = struct.unpack(FL32, file_contents[518:522])[0]
    meta['coords_y_rot'] = struct.unpack(FL32, file_contents[522:526])[0]
    meta['coords_z_rot'] = struct.unpack(FL32, file_contents[526:530])[0]
    meta['coherence_mode'] = struct.unpack(IL16, file_contents[530:532])[0]
    meta['surface_filter'] = struct.unpack(IL16, file_contents[532:534])[0]
    meta['sys_err_file_name'] = file_contents[534:562].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['zoom_descr'] = file_contents[562:570].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['alpha_part'] = struct.unpack(FL32, file_contents[570:574])[0]
    meta['beta_part'] = struct.unpack(FL32, file_contents[574:578])[0]
    meta['dist_part'] = struct.unpack(FL32, file_contents[578:582])[0]
    meta['cam_split_loc_x'] = struct.unpack(IL16, file_contents[582:584])[0]
    meta['cam_split_loc_y'] = struct.unpack(IL16, file_contents[584:586])[0]
    meta['cam_split_trans_x'] = struct.unpack(IL16, file_contents[586:588])[0]
    meta['cam_split_trans_y'] = struct.unpack(IL16, file_contents[588:590])[0]
    meta['material_a'] = file_contents[590:614].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['material_b'] = file_contents[614:638].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    # 639-642 unused
    meta['dmi_ctr_x'] = struct.unpack(FL32, file_contents[642:646])[0]
    meta['dmi_ctr_y'] = struct.unpack(FL32, file_contents[646:650])[0]
    meta['sph_dist_corr'] = struct.unpack(IL16, file_contents[650:652])[0]
    # 653-654 unused
    meta['sph_dist_part_na'] = struct.unpack(FL32, file_contents[654:658])[0]
    meta['sph_dist_part_radius'] = struct.unpack(FL32, file_contents[658:662])[0]
    meta['sph_dist_cal_na'] = struct.unpack(FL32, file_contents[662:666])[0]
    meta['sph_dist_cal_radius'] = struct.unpack(FL32, file_contents[666:670])[0]
    meta['surface_type'] = struct.unpack(IL16, file_contents[670:672])[0]
    meta['ac_surface_type'] = struct.unpack(IL16, file_contents[672:674])[0]
    meta['z_pos'] = struct.unpack(FL32, file_contents[674:678])[0]
    meta['power_multiplier'] = struct.unpack(FL32, file_contents[678:682])[0]
    meta['focus_multiplier'] = struct.unpack(FL32, file_contents[682:686])[0]
    meta['roc_focus_cal_factor'] = struct.unpack(FL32, file_contents[686:690])[0]
    meta['roc_power_cal_factor'] = struct.unpack(FL32, file_contents[690:694])[0]
    meta['ftp_left_pos'] = struct.unpack(FL32, file_contents[694:698])[0]
    meta['ftp_right_pos'] = struct.unpack(FL32, file_contents[698:702])[0]
    meta['ftp_pitch_pos'] = struct.unpack(FL32, file_contents[702:706])[0]
    meta['ftp_roll_pos'] = struct.unpack(FL32, file_contents[706:710])[0]
    meta['min_mod_percent'] = struct.unpack(FL32, file_contents[710:714])[0]
    meta['max_intens'] = struct.unpack(IL32, file_contents[714:718])[0]
    meta['ring_of_fire'] = struct.unpack(IL16, file_contents[718:720])[0]
    # 721 unused
    meta['rc_orientation'] = struct.unpack(C, file_contents[721:722])[0].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    meta['rc_distance'] = struct.unpack(FL32, file_contents[722:726])[0]
    meta['rc_angle'] = struct.unpack(FL32, file_contents[726:730])[0]
    meta['rc_diameter'] = struct.unpack(FL32, file_contents[730:734])[0]
    meta['rem_fringes_mode'] = struct.unpack(IB16, file_contents[734:736])[0]
    # 737 unused
    meta['ftpsi_phase_res'] = struct.unpack(uint8, file_contents[737:738])[0]
    meta['frames_acquired'] = struct.unpack(IL16, file_contents[738:740])[0]
    meta['cavity_type'] = struct.unpack(IL16, file_contents[740:742])[0]
    meta['cam_frame_rate'] = struct.unpack(FL32, file_contents[742:746])[0]
    meta['tune_range'] = struct.unpack(FL32, file_contents[746:750])[0]
    meta['cal_pix_loc_x'] = struct.unpack(IL16, file_contents[750:752])[0]
    meta['cal_pix_loc_y'] = struct.unpack(IL16, file_contents[752:754])[0]
    meta['n_test_cal_pts'] = struct.unpack(IL16, file_contents[754:756])[0]
    meta['n_ref_cal_pts'] = struct.unpack(IL16, file_contents[756:758])[0]
    meta['test_cal_pts'] = [
        struct.unpack(FL32, file_contents[758:762])[0],
        struct.unpack(FL32, file_contents[762:766])[0],
        struct.unpack(FL32, file_contents[766:770])[0],
        struct.unpack(FL32, file_contents[770:774])[0],
    ]
    meta['ref_cal_pts'] = [
        struct.unpack(FL32, file_contents[774:778])[0],
        struct.unpack(FL32, file_contents[778:782])[0],
        struct.unpack(FL32, file_contents[782:786])[0],
        struct.unpack(FL32, file_contents[786:790])[0],
    ]
    meta['test_cal_pix_opd'] = struct.unpack(FL32, file_contents[790:794])[0]
    meta['test_ref_pix_opd'] = struct.unpack(FL32, file_contents[794:798])[0]
    meta['flash_phase_cd_mask'] = struct.unpack(FL32, file_contents[798:802])[0]
    meta['flash_phase_alias_mask'] = struct.unpack(FL32, file_contents[802:806])[0]
    meta['flask_phase_filter'] = struct.unpack(FL32, file_contents[806:810])[0]
    meta['scan_direction'] = struct.unpack(uint8, file_contents[810:811])[0]
    # 812 - 814 unused
    meta['ftpsi_res_factor'] = struct.unpack(IL16, file_contents[814:816])[0]
    # 835 - 900 films, for later
    # 901 - 4096 unused

    return meta


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
