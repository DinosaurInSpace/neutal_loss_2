import os
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from io import StringIO

import numpy as np
import pandas as pd
import requests
import json
from PIL import Image
from metaspace.sm_annotation_utils import GraphQLClient, get_config, SMInstance
from requests.adapters import HTTPAdapter
from scipy.ndimage import zoom, median_filter
from sklearn.metrics.pairwise import cosine_similarity

#%% Parameters

BASE_PATH = os.getcwd()  # '/home/lachlan/dev/notebooks/new_neutral_loss'

# METASPACE credentials (needed for private dataset access)

for key, value in dict(**json.load(open('/Users/dis/.metaspace.json'))).items():
    SM_USER = key
    SM_PASS = value

METASPACE_HOST = 'https://beta.metaspace2020.eu'

DATASETS_WITH_NL = pd.read_csv(StringIO(
"""ds_id,name,polarity,organism,organism_part,analyzer,ionisation_source,maldi_matrix
Literal_to_replace_ds_id
"""))
DATASETS_WITH_NL['ds_id'] = DATASETS_WITH_NL.ds_id.str.strip()

#2019-07-26_11h26m33s,240719_FDA_HELA _NA_W3_DAN_NEG_190725161642,Negative,Hela cells,Cells,Orbitrap,MALDI,DAN
#2019-07-26_11h26m33s,240719_FDA_HELA _NA_W3_DAN_NEG_190725161642,negative,Hela cells,Cells,Orbitrap,MALDI,dan

NEUTRAL_LOSSES = ['-H2O']
#NEUTRAL_LOSSES = ['-H2O', '-CN', '-NH2', '-COH', '-CO2H']
MAX_FDR = 0.2

#%% Code to reprocess datasets with new neutral losses


# NOTE: Only uncomment & run this code when you need to reprocess datasets.
# It takes several hours for all datasets to reprocess, and they cannot be accessed or changed
# while they are reprocessing. Progress can be monitored on the METASPACE website.

reprocess = False

if reprocess == True:
    sm = SMInstance(host=METASPACE_HOST)
    sm.login(**json.load(open('/Users/dis/.metaspace.json')))
    for ds_id in DATASETS_WITH_NL.ds_id:
        query = """mutation updateNeutralLosses($id: String!, $input: DatasetUpdateInput!) {
        updateDataset(id: $id, input: $input, reprocess: true, force: true)
        }"""
        new_params = {
        "neutralLosses": NEUTRAL_LOSSES, "molDBs": 'HMDB-v4'
        }
        sm._gqclient.query(query, {"id": ds_id, "input": new_params})


#%% Helper functions for retrieving data from METASPACE

ANNOTATION_FIELDS = ("sumFormula neutralLoss adduct mz msmScore fdrLevel offSample ion ionFormula "
                     "dataset { id } "
                     "possibleCompounds { information { databaseId } } "
                     "isotopeImages { url maxIntensity totalIntensity } "
                     "isomers { ion } "
                     "isobars { ion msmScore } ")


def get_ion_images_for_analysis(img_ids, hotspot_percentile=99, max_size=None, max_mem_mb=2048):
    """Retrieves ion images, does hot-spot removal and resizing,
    and returns them as numpy array.

    Args:
        img_ids (list[str]):
        hotspot_percentile (float):
        max_size (Union[None, tuple[int, int]]):
            If images are greater than this size, they will be downsampled to fit in this size
        max_mem_mb (Union[None, float]):
            If the output numpy array would require more than this amount of memory,
            images will be downsampled to fit

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int]]
            (value, mask, (h, w))
            value - A float32 numpy array with shape (len(img_ids), h * w)
                where each row is one image
            mask - A float32 numpy array with shape (h, w) containing the ion image mask.
                May contain values that are between 0 and 1 if downsampling
                caused both filled and empty pixels to be merged
            h, w - shape of each image in value such that value[i].reshape(h, w)
                reconstructs the image
    """
    assert all(img_ids)

    zoom_factor = 1
    h, w = None, None
    value, mask = None, None

    def setup_shared_vals(img):
        nonlocal zoom_factor, h, w, value, mask

        img_h, img_w = img.height, img.width
        if max_size:
            size_zoom = min(max_size[0] / img_h, max_size[1] / img_w)
            zoom_factor = min(zoom_factor, size_zoom)
        if max_mem_mb:
            expected_mem = img_h * img_w * len(img_ids) * 4 / 1024 / 1024
            zoom_factor = min(zoom_factor, max_mem_mb / expected_mem)

        raw_mask = np.float32(np.array(img)[:, :, 3] != 0)
        if abs(zoom_factor - 1) < 0.001:
            zoom_factor = 1
            mask = raw_mask
        else:
            mask = zoom(raw_mask, zoom_factor, prefilter=False)

        h, w = mask.shape
        value = np.empty((len(img_ids), h * w), dtype=np.float32)

    def process_img(img_id, idx, do_setup=False):
        img = Image.open(session.get(f'{METASPACE_HOST}/fs/iso_images/{img_id}', stream=True).raw)
        if do_setup:
            setup_shared_vals(img)

        img_arr = np.asarray(img, dtype=np.float32)[:, :, 0]

        # Try to use the hotspot percentile,
        # but fall back to the image's maximum or 1.0 if needed
        # to ensure that there are no divide-by-zero issues
        hotspot_threshold = np.percentile(img_arr, hotspot_percentile) or np.max(img_arr) or 1.0
        np.clip(img_arr, None, hotspot_threshold, out=img_arr)

        if zoom_factor != 1:
            zoomed_img = zoom(img_arr, zoom_factor)
        else:
            zoomed_img = img_arr

        # Note: due to prefiltering & smoothing, zoom can change the min/max of the image,
        # so hotspot_threshold cannot be reused here for scaling
        np.clip(zoomed_img, 0, None, out=zoomed_img)
        zoomed_img /= np.max(zoomed_img) or 1

        value[idx, :] = zoomed_img.ravel()

    session = requests.Session()
    session.mount(METASPACE_HOST, HTTPAdapter(max_retries=5, pool_maxsize=100))
    process_img(img_ids[0], 0, do_setup=True)
    with ThreadPoolExecutor() as executor:
        for _ in executor.map(process_img, img_ids[1:], range(1, len(img_ids))):
            pass

    return value, mask, (h, w)

# DB's: {'HMDB-v4-endogenous', 'HMDB-v4', 'HMDB-ENDO-DETECTED', 'HMDB-SPOTS-KNOWN'}
def get_single_dataset_images(ds_id, fdr=0.5):
    gql = GraphQLClient(get_config(METASPACE_HOST, email=SM_USER, password=SM_PASS))
    gql.ANNOTATION_FIELDS = ANNOTATION_FIELDS # Override selected fields to include neutral losses, HMDB IDs, etc.
    anns = gql.getAnnotations({'database': 'HMDB-v4', 'fdrLevel': fdr, 'hasNeutralLoss': None},
                              {'ids': ds_id})
    img_ids = [ann['isotopeImages'][0]['url'][-32:] for ann in anns]
    if img_ids:
        images, mask, (h, w) = get_ion_images_for_analysis(img_ids, max_mem_mb=2048,
                                                           hotspot_percentile=99)
    else:
        raise Exception(f'{ds_id} not found')

    anns_df = pd.DataFrame([
        dict(
            image=images[i],
            formula=ann['sumFormula'],
            ion=ann['ion'],
            ion_formula=ann['ionFormula'],
            neutral_loss=ann['neutralLoss'],
            adduct=ann['adduct'],
            msm=ann['msmScore'],
            fdr=ann['fdrLevel'],
            off_sample=ann['offSample'],
            hmdb_ids=[c['information'][0]['databaseId'] for c in ann['possibleCompounds']],
            # intensity_max=ann['isotopeImages'][0]['maxIntensity'],
            # intensity_99th=ann['isotopeImages'][0]['totalIntensity'] * np.percentile(images[i][images[i] != 0], 99),
            intensity_avg=ann['isotopeImages'][0]['totalIntensity'] / np.count_nonzero(images[i]),
            isobars=[isobar['ion'] for isobar in ann['isobars'] if isobar['msmScore'] > ann['msmScore']],
            isomers=[isomer['ion'] for isomer in ann['isomers']],
    ) for i, ann in enumerate(anns)])
    return anns_df, images, mask, h, w


def get_median_filtered_coloc(a, b, h, w):
    def preprocess(img):
        img = img.copy().reshape((h, w))
        img[img < np.quantile(img, 0.5)] = 0
        return median_filter(img, (3, 3)).reshape([1, h*w])

    return cosine_similarity(preprocess(a), preprocess(b))[0,0]


#%% Make report

def get_ds_neutral_loss_stats(ds_id, fdr):
    anns_df, images, mask, h, w = get_single_dataset_images(ds_id, fdr)
    suffixes = [nl.replace('-','_') for nl in NEUTRAL_LOSSES]
    print(f'Got {len(anns_df)} annotations for {ds_id} @ {fdr}')

    # Start with a frame of all the annotations without any neutral loss
    df = anns_df[anns_df.neutral_loss == '']

    # Merge in all neutral loss annotations as separate columns
    for nl, suffix in zip(NEUTRAL_LOSSES, suffixes):
        df = pd.merge(
            df,
            anns_df[anns_df.neutral_loss == nl],
            how='outer',
            on=['formula','adduct'],
            suffixes=['',suffix],
        )

    df['ds_id'] = ds_id
    df['has_no_loss'] = df['msm'].notna()

    for nl, suffix in zip(NEUTRAL_LOSSES, suffixes):
        # When no-loss annotations aren't present, copy the HMDB IDs from the with-loss annotation
        df['hmdb_ids'].fillna(df['hmdb_ids' + suffix], inplace=True)
        # Add other stats
        df['has' + suffix] = df['msm' + suffix].notna()
        df['colocalization' + suffix] = pd.Series(dtype=np.float32)
        for idx, r in df.iterrows():
            if r.has_no_loss and r['has' + suffix]:
                coloc = get_median_filtered_coloc(r.image, r['image' + suffix], h, w)
            else:
                coloc = 0
            df.loc[idx, 'colocalization' + suffix] = coloc

        df['loss_intensity_ratio' + suffix] = np.select(
            [df.msm.isna(), df['msm' + suffix].isna()],
            [np.inf, 0],
            df['intensity_avg' + suffix] / df.intensity_avg
        )

        df['loss_intensity_share' + suffix] = (
                df['intensity_avg' + suffix].fillna(0) /
                (df['intensity_avg' + suffix].fillna(0) + df.intensity_avg.fillna(0))
        )

        # Remove columns that are no longer needed
        df.drop(columns=['image' + suffix, 'hmdb_ids' + suffix, 'neutral_loss' + suffix],
                inplace=True)


    df.drop(columns=['image', 'neutral_loss'], inplace=True)

    return df

nl_stats = pd.concat([get_ds_neutral_loss_stats(ds_id, MAX_FDR) for ds_id in DATASETS_WITH_NL.ds_id])
nl_stats.to_pickle(f'{BASE_PATH}/Literal_to_replace_out.pickle')

#%%
#%%