from ...image_server import get_sftp
from . import add_label_tag
from ..resources import metadata
from tqdm import tqdm


VXMLS_dir = "/med-i_data/exact_prostate_segemnts/vxmls/VXMLs"


# =============== Server Utils ============


def _get_label(core_specifier):
    return metadata().query("core_specifier == @core_specifier")["grade"].iloc[0]


def get_path_to_vxml(core_specifier):
    label = _get_label(core_specifier)
    path = f"{VXMLS_dir}/{label}/{core_specifier}_{label}.vxml"
    return path


def download_vxml(core_specifier, fpath):
    get_sftp().get(get_path_to_vxml(core_specifier), fpath)


# ================ XML Parsing ===========

from xml.etree import ElementTree


def _open_xml_tree(path):
    """
    Returns the root element of the xml tree at the specified file path
    """
    with open(path) as f:
        return ElementTree.fromstring(f.read())


def _flatten_dict(d, sep="."):
    """Flattens a dict recursively"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({f"{k}{sep}{k_sub}": v for k_sub, v in _flatten_dict(v).items()})
        else:
            out[k] = v

    return out


def _parse_settings_from_settings_group(group):
    """
    Parses the settings from one of the 'group' elements at the
    level /settings/roots/root/groups of the xml tree
    """
    out = {}
    for parameter in group.find("parameters"):
        name = parameter.get("name")
        settings = parameter.find("settings")
        if (v := settings.get("value")) is not None:
            value = v
        else:
            value = [v.get("value") for v in settings.find("values")]
        out[name] = value

    return out


def _parse_all_parameters(path_or_root):
    """
    Parses the parameters from the root element of the XML tree
    """
    out = {}
    if isinstance(path_or_root, str):
        root = _open_xml_tree(path_or_root)
    else:
        root = path_or_root

    groups = root.find("settings/roots/root/groups")
    for group in groups:
        name = group.get("name")
        out[name] = _parse_settings_from_settings_group(group)

    for k, v in list(out.items()):
        if not v:
            out.pop(k)

    return out


def remove_columns_with_no_variation(dataframe):
    """Returns a modified version of the given dataframe
    with columns that are the same for every row removed
    """
    num_unique_per_column = dataframe.describe().loc["unique"]
    columns_to_keep = [
        column for column, val in num_unique_per_column.items() if val > 1
    ]
    return dataframe[columns_to_keep]


def get_settings_dataframe(cores):
    """
    given the list of core specifiers,
    parses all the imaging vxml files to create a table whose rows
    are the given imaging settings for that core acquisition.
    """
    all_settings = {}

    for core in tqdm(cores):
        download_vxml(core, "current_xml.xml")
        t = _open_xml_tree("current_xml.xml")

        all_settings[core] = _flatten_dict(_parse_all_parameters(t))

    settings_by_core = {}

    for core, settings in all_settings.items():
        for name, value in settings.items():
            settings_by_core.setdefault(name, {})
            settings_by_core[name].setdefault(core, {})
            settings_by_core[name][core] = value

    import pandas as pd

    settings_by_core = pd.DataFrame(settings_by_core)

    return settings_by_core
