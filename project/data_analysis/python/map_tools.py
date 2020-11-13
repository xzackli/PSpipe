from pixell import enmap
from pixell import reproject
import healpy as hp
import numpy as np
#from taylens import taylor_interpol_iter


def read_map(filename, filetype, *args, **kwargs):
    if filetype == "HP":
        if 'verbose' in kwargs.keys():
            return hp.read_map(filename, *args, **kwargs)
        else:
            return hp.read_map(filename, *args, verbose=False, **kwargs)
    elif filetype == "CAR":
        return enmap.read_map(filename, *args, **kwargs)
    else:
        raise ValueError("filetype %s must be 'HP' or 'CAR'" % filetype)

def pad_map(mapp):
    if (mapp is not None) and (len(mapp.shape) < 2):
        mapp = np.array([mapp])
    return mapp

def hp_rotate(map_hp, coord):
    """Rotate healpix map between coordinate systems

    :param map_hp: A healpix map in RING ordering
    :param coord: A len(2) list of either 'G', 'C', 'E'
    Galactic, equatorial, ecliptic, eg ['G', 'C'] converts
    galactic to equatorial coordinates
    :returns: A rotated healpix map
    """
    if map_hp is None:
        return None
    if coord[0] == coord[1]:
        return map_hp
    r = hp.rotator.Rotator(coord=coord)
    new_map = r.rotate_map_pixel(map_hp)
    return new_map


def hp2car(map_hp, imap, hp_coord=['G', 'G'], do_mask=False):
    """Match Healpix to CAR region, convert to CAR using taylens interpolation

    :param map_hp: Desired map in healpix format
    :param imap:   Mask or other template in equatorial coordinates.
                   This provides the grid of points map_hp is projected onto
    :param hp_coord: len(2) list of 'G', 'C', 'E', coord transform to apply to map_hp
    :param do_mask:  bool, multiplies the new map by imap (assumed a mask)
    :return: map_hp in CAR, possibly rotated / masked
    """
    if map_hp is None:
        return None
    map_hp[map_hp < -1e20] = 0
    map_hp = hp_rotate(map_hp, hp_coord)
    # Get positions on which to calculate the map
    posmap = imap.posmap()
    posmap_shape = posmap.shape
    posmap_flat = posmap.reshape(posmap_shape[0], posmap_shape[1]*posmap_shape[2])
    posmap_flat[0] = np.pi/2 - posmap_flat[0]  # Convert dec to theta

    # Interpolate to new positions
    res = taylor_interpol_iter(map_hp, posmap_flat)
    y = np.array([x for x in res])
    ys = np.reshape(y, (y.shape[0], posmap_shape[1], posmap_shape[2]))
    map_flat = ys[-1]

    # Make CAR
    map_car = enmap.zeros(*imap.geometry)
    map_car[:, :] = map_flat
    if do_mask:
        map_car = map_car * imap
    return map_car


def car2hp(imap, nside=2048, hp_coord=['C', 'C']):
    """ Convert CAR to healpix, change coords"""
    if imap is None:
        return None
    map_hp = imap.to_healpix(nside=nside)
    map_hp = hp_rotate(map_hp, hp_coord)
    return map_hp


def hp2car2(map_hp, imap, hp_coord=['G', 'G'], do_mask=False):
    """Match Planck healpix data to a CAR mask region and convert to CAR"""
    # Get bounding box and resolution
    if map_hp is None:
        return None
    box = imap.box()  # radians
    box_deg = np.rad2deg(box)
    latra, lonra = box_deg[:, 0], box_deg[:, 1]
    npix = imap.geometry[0][1]  # Get same resolution as ACT mask
    # Convert healpix to CAR
    cart = hp.cartview(map_hp, coord=hp_coord, latra=latra,
                       lonra=lonra[::-1], return_projected_map=True, xsize=npix)
    map_car = enmap.zeros(*imap.geometry)
    map_car[:, :] = cart
    if do_mask:
        map_car = map_car * imap
    return map_car

def hp2car3(map_hp, imap, hp_coord=['G','G'], do_mask=False):
    coord_dict = {"G":"gal", "C":"equ"}
    coords = coord_dict[hp_coord[0]]+','+coord_dict[hp_coord[1]]
    shape, wcs = imap.shape, imap.wcs
    map_car = reproject.enmap_from_healpix_interp(map_hp, shape, wcs, coords, interpolate=False)
    if do_mask:
        map_car *= imap
    return map_car

def convert(mapp, in_format, out_format, coord, nside=None, imap=None):
    if in_format == 'HP' and out_format == 'CAR':
        mapp = hp2car3(mapp, imap, coord, False)
    elif in_format == 'HP' and out_format == 'HP':
        mapp = hp_rotate(mapp, coord)
    elif in_format == 'CAR' and out_format == 'HP':
        mapp = car2hp(mapp, nside, coord)
    elif in_format == 'CAR' and out_format == 'CAR':
        pass
    else:
        raise ValueError("in_format %s or out_format %s is invalid. Only 'CAR' and 'HP' are permitted." % (in_format, out_format))
    return mapp
