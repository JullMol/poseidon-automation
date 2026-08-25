import os
import sys
import json
import time
import datetime
import zipfile
import joblib
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, MultiPolygon
from supabase import create_client, Client

try:
    from shapely import contains_xy
except ImportError:
    from shapely.vectorized import contains as contains_xy

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

HF_BASE_URL = "https://huggingface.co/JullMol/POSEIDON/resolve/main/POSEIDON_Model"
HF_MODEL_URLS = {
    "length_model": f"{HF_BASE_URL}/model_length_gb.pkl",
    "fishing_model": f"{HF_BASE_URL}/model_fishing_score_xgboost.pkl",
    "isolation_forest": f"{HF_BASE_URL}/GEE/isolation_forest_model.pkl",
    "gee_models": f"{HF_BASE_URL}/GEE/poseidon_models_gee.pkl",
}

GDRIVE_MODEL_IDS = {
    "length_model": "1IPKwME5f-TgfqE5V3f99JAwKmi4y1e8R",
    "fishing_model": "1JfJ7ww1f-zRoCF1kezwSTy7e4_1szsxy",
}

FEATURE_LABELS_MAP = {
    'vv_intensity_db': 'Intensitas Radar VV (dB)',
    'vh_intensity_db': 'Intensitas Radar VH (dB)',
    'radar_intensity_diff': 'Rasio Polarimetri VV/VH (dB)',
    'background_clutter': 'Derau Hamburan Laut (Clutter)',
    'snr_db': 'Rasio Sinyal terhadap Derau (SNR dB)',
    'length_m_new': 'Estimasi Panjang Kapal (m)',
    'dist_to_nearest_mpa_km': 'Jarak ke Kawasan Konservasi (km)',
    'dist_to_eez_boundary_km': 'Jarak ke Batas ZEE (km)',
    'dist_to_nearest_port_km': 'Jarak ke Pelabuhan Terdekat (km)',
    'dist_to_nearest_seizure_km': 'Jarak ke Titik Tangkapan Historis (km)',
    'fishing_score_new': 'Indeks Probabilitas Penangkapan Ikan',
    'historical_hotspot': 'Zona Rawan Pelanggaran Historis',
    'quarter': 'Kuartal Operasional',
}

ALPHA_CONFORMAL = 0.10

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL dan SUPABASE_KEY wajib disetel.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def init_gee():
    import ee
    
    ee_sa = os.environ.get("EE_SERVICE_ACCOUNT")
    ee_pk = os.environ.get("EE_PRIVATE_KEY")
    ee_proj = os.environ.get("EE_PROJECT", "darvin-natuna-2025")
    
    if ee_sa and ee_pk:
        if "\\n" in ee_pk:
            ee_pk = ee_pk.replace("\\n", "\n")
        credentials = ee.ServiceAccountCredentials(ee_sa, key_data=ee_pk)
        ee.Initialize(credentials=credentials, project=ee_proj)
        print("Autentikasi GEE via Environment Variables (GitHub Secrets).")
        return
        
    raise ValueError("Kredensial GEE (EE_SERVICE_ACCOUNT & EE_PRIVATE_KEY) tidak ditemukan di Environment Variables. Dilarang menggunakan kredensial lokal (Hardcoded JSON).")

def load_model_file(model_key: str):
    filename_map = {
        "length_model": "model_length_gb.pkl",
        "fishing_model": "model_fishing_score_xgboost.pkl",
        "isolation_forest": "isolation_forest_model.pkl",
        "gee_models": "poseidon_models_gee.pkl",
    }
    fname = filename_map[model_key]
    cache_dir = os.path.join("automation", "models")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, fname)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return joblib.load(local_path)

    if model_key in GDRIVE_MODEL_IDS:
        try:
            import gdown
            print(f"Mengunduh model {fname} dari Google Drive...")
            gdown.download(id=GDRIVE_MODEL_IDS[model_key], output=local_path, quiet=True)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                return joblib.load(local_path)
        except Exception:
            pass

    hf_url = HF_MODEL_URLS[model_key]
    print(f"Mengunduh model {fname} dari Hugging Face...")
    r = requests.get(hf_url, stream=True, timeout=180)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            f.write(chunk)
    return joblib.load(local_path)

def get_p70_quarter_reference_threshold(target_date: datetime.date, supabase_client=None) -> float:
    ref_year = target_date.year - 1
    curr_q = int((target_date.month - 1) // 3 + 1)
    q_date_ranges = {
        1: (f"{ref_year}-01-01", f"{ref_year}-03-31"),
        2: (f"{ref_year}-04-01", f"{ref_year}-06-30"),
        3: (f"{ref_year}-07-01", f"{ref_year}-09-30"),
        4: (f"{ref_year}-10-01", f"{ref_year}-12-31")
    }
    start_d, end_d = q_date_ranges[curr_q]

    if supabase_client is None:
        supabase_client = get_supabase()

    print(f"Mengambil data referensi raw_risk_score Q{curr_q} {ref_year} ({start_d} s.d. {end_d}) dari Supabase...")
    all_rows = []
    page_size = 1000
    start_idx = 0
    while True:
        resp = supabase_client.table("vessel_detections") \
            .select("raw_risk_score") \
            .gte("pass_date", start_d) \
            .lte("pass_date", end_d) \
            .range(start_idx, start_idx + page_size - 1) \
            .execute()
        batch = resp.data if resp.data else []
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start_idx += page_size

    if all_rows:
        vals = pd.DataFrame(all_rows)['raw_risk_score'].dropna().values
        if len(vals) > 0:
            p70_val = float(np.percentile(vals, 70))
            print(f"Ditemukan {len(vals):,} baris data referensi Q{curr_q} {ref_year}. Ambang P70: {p70_val:.8f}")
            return p70_val

    start_yr, end_yr = f"{ref_year}-01-01", f"{ref_year}-12-31"
    print(f"Kuartal kosong, mengambil data referensi tahun penuh {ref_year} dari Supabase...")
    all_rows_yr = []
    start_idx = 0
    while True:
        resp = supabase_client.table("vessel_detections") \
            .select("raw_risk_score") \
            .gte("pass_date", start_yr) \
            .lte("pass_date", end_yr) \
            .range(start_idx, start_idx + page_size - 1) \
            .execute()
        batch = resp.data if resp.data else []
        if not batch:
            break
        all_rows_yr.extend(batch)
        if len(batch) < page_size:
            break
        start_idx += page_size

    if all_rows_yr:
        vals_yr = pd.DataFrame(all_rows_yr)['raw_risk_score'].dropna().values
        if len(vals_yr) > 0:
            p70_val = float(np.percentile(vals_yr, 70))
            print(f"Ditemukan {len(vals_yr):,} baris data tahun {ref_year}. Ambang P70: {p70_val:.8f}")
            return p70_val

    resp_all = supabase_client.table("vessel_detections").select("raw_risk_score").limit(5000).execute()
    if resp_all.data:
        vals_all = pd.DataFrame(resp_all.data)['raw_risk_score'].dropna().values
        if len(vals_all) > 0:
            p70_val = float(np.percentile(vals_all, 70))
            print(f"Ditemukan {len(vals_all):,} baris sampel Supabase. Ambang P70: {p70_val:.8f}")
            return p70_val

    raise ValueError("Tidak dapat menghitung threshold P70 dari data Supabase vessel_detections.")

def scrape_gee_daily(target_date: datetime.date) -> pd.DataFrame:
    import ee
    init_gee()

    wpp711_geom = ee.Geometry.Rectangle([102.90, -4.16, 111.05, 7.74])
    date_start = target_date.strftime("%Y-%m-%d")
    date_end = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Mengekstrak citra satelit Sentinel-1 GRD IW untuk WPP 711 pada tanggal: {date_start}")

    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(wpp711_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    )

    n_scene = s1.size().getInfo()
    print(f"Total scene Sentinel-1 GRD IW ditemukan pada {date_start}: {n_scene} scene.")
    if n_scene == 0:
        return pd.DataFrame()

    scenes = s1.toList(n_scene)
    all_detections = []

    for i in range(n_scene):
        try:
            t0_scene = time.time()
            img = ee.Image(scenes.get(i))
            scene_id = img.get('system:index').getInfo()
            timestamp_str = img.date().format('YYYY-MM-dd HH:mm:ss').getInfo()

            print(f"[{i+1}/{n_scene}] Memproses scene {scene_id} ({timestamp_str})...")

            vv = img.select('VV')
            mask = vv.gt(-3.0)
            blob = mask.selfMask().connectedComponents(
                connectedness=ee.Kernel.plus(1),
                maxSize=32
            )
            ukuran = blob.select('labels').connectedPixelCount(maxSize=32)
            blob_valid = blob.updateMask(ukuran.gte(2).And(ukuran.lte(20)))

            area_irisan = img.geometry().intersection(wpp711_geom, ee.ErrorMargin(1))
            vektor = blob_valid.select('labels').reduceToVectors(
                geometry=area_irisan,
                scale=40,
                geometryType='centroid',
                maxPixels=1e9,
                bestEffort=True
            )

            feats = vektor.limit(3000).getInfo().get('features', [])
            if not feats:
                print(f"[{i+1}/{n_scene}] Scene {scene_id}: 0 kandidat terdeteksi.")
                continue

            raw_points = []
            for feat in feats:
                coords = feat['geometry']['coordinates']
                raw_points.append({
                    'scene_id': scene_id,
                    'timestamp': timestamp_str,
                    'pass_date': date_start,
                    'lat': coords[1],
                    'lon': coords[0]
                })

            df_pts = pd.DataFrame(raw_points)
            total_pts = len(df_pts)
            print(f"[{i+1}/{n_scene}] Scene {scene_id}: {total_pts} kandidat titik terdeteksi, mengekstrak polarimetri...")

            CHUNK_SIZE = 100
            total_chunks = (total_pts + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx, start_idx in enumerate(range(0, total_pts, CHUNK_SIZE)):
                df_chunk = df_pts.iloc[start_idx:start_idx + CHUNK_SIZE].copy().reset_index(drop=True)

                features_vessel = []
                for r_idx, row in df_chunk.iterrows():
                    pt = ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]).buffer(400), {'row_idx': r_idx})
                    features_vessel.append(pt)
                fc_vessel = ee.FeatureCollection(features_vessel)

                features_bg = []
                for r_idx, row in df_chunk.iterrows():
                    pt = ee.Feature(
                        ee.Geometry.Point([row['lon'], row['lat']]).buffer(800)
                        .difference(ee.Geometry.Point([row['lon'], row['lat']]).buffer(200)),
                        {'row_idx': r_idx}
                    )
                    features_bg.append(pt)
                fc_bg = ee.FeatureCollection(features_bg)

                vessel_stats = img.select(['VV', 'VH']).reduceRegions(collection=fc_vessel, reducer=ee.Reducer.mean(), scale=10)
                bg_stats = img.select('VV').reduceRegions(collection=fc_bg, reducer=ee.Reducer.mean(), scale=10)

                v_list = vessel_stats.toList(len(df_chunk)).getInfo()
                bg_list = bg_stats.toList(len(df_chunk)).getInfo()

                v_map = {f['properties']['row_idx']: f['properties'] for f in v_list}
                bg_map = {f['properties']['row_idx']: f['properties'] for f in bg_list}

                for r_idx, row in df_chunk.iterrows():
                    v = v_map.get(r_idx, {})
                    bg = bg_map.get(r_idx, {})
                    vv_val = v.get('VV')
                    vh_val = v.get('VH')
                    bg_val = bg.get('mean')

                    if vv_val is not None and vh_val is not None and bg_val is not None:
                        vv_vh_ratio = float(vv_val - vh_val)
                        snr = float(vv_val - bg_val)
                        all_detections.append({
                            'scene_id': scene_id,
                            'timestamp': timestamp_str,
                            'pass_date': date_start,
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'vv_intensity_db': float(vv_val),
                            'vh_intensity_db': float(vh_val),
                            'radar_intensity_diff': vv_vh_ratio,
                            'vv_vh_ratio': vv_vh_ratio,
                            'background_clutter': float(bg_val),
                            'snr_db': snr,
                            'is_dark': 1,
                            'ais_flag_country': 'UNKNOWN',
                            'ais_flag_country_encoded': 15
                        })

                print(f"   Chunk {chunk_idx+1}/{total_chunks}: {min(start_idx + CHUNK_SIZE, total_pts)}/{total_pts} titik polarimetri selesai diekstrak.")

            print(f"[{i+1}/{n_scene}] Scene {scene_id} selesai dalam {time.time() - t0_scene:.1f} detik.")

        except Exception as e:
            print(f"[{i+1}/{n_scene}] Melewati scene karena error: {e}")

    df_out = pd.DataFrame(all_detections)
    if df_out.empty:
        return df_out

    df_out = df_out.drop_duplicates(subset=['scene_id', 'lat', 'lon']).reset_index(drop=True)
    print(f"Total kandidat polarimetri sebelum filter: {len(df_out):,} titik.")

    print(f"Menjalankan filter noise Isolation Forest langsung pada hasil ekstraksi radar...")
    iforest = load_model_file("isolation_forest")
    fitur_anomali = ['vv_intensity_db', 'vh_intensity_db', 'vv_vh_ratio', 'background_clutter', 'snr_db']
    
    df_eval = df_out[fitur_anomali].dropna()
    if df_eval.empty:
        print("Seluruh kandidat memiliki nilai polarimetri kosong.")
        return pd.DataFrame()

    preds_anom = iforest.predict(df_eval)
    df_filtered = df_out.loc[df_eval.index].copy()
    df_filtered['is_vessel'] = (preds_anom == 1).astype(int)
    df_clean_vessels = df_filtered[df_filtered['is_vessel'] == 1].reset_index(drop=True)

    n_lolos = len(df_clean_vessels)
    n_dibuang = len(df_filtered) - n_lolos
    pct_lolos = (n_lolos / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0.0
    print(f"Filter Isolation Forest selesai: {n_lolos:,} kapal terkonfirmasi lolos ({pct_lolos:.1f}%), {n_dibuang:,} noise clutter dibuang.")

    return df_clean_vessels

def fill_holes(geom):
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def haversine_km_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0)**2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def compute_dist_to_seizure(lat_arr, lon_arr, seiz_lat, seiz_lon):
    lat_r = np.radians(lat_arr)[:, None]
    lon_r = np.radians(lon_arr)[:, None]
    seiz_lat_r = np.radians(seiz_lat)[None, :]
    seiz_lon_r = np.radians(seiz_lon)[None, :]

    dlat = seiz_lat_r - lat_r
    dlon = seiz_lon_r - lon_r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_r) * np.cos(seiz_lat_r) * np.sin(dlon / 2.0)**2
    dists_matrix = 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))
    dists_sorted = np.sort(dists_matrix, axis=1)
    min_dists = np.where(
        dists_sorted[:, 0] < 0.05,
        np.where(dists_sorted.shape[1] > 1, dists_sorted[:, 1], -1.0),
        dists_sorted[:, 0]
    )
    return np.round(min_dists, 4)

def ensure_spatial_bundle():
    spatial_cache_dir = os.path.join("automation", "spatial")
    os.makedirs(spatial_cache_dir, exist_ok=True)
    
    spatial_base = os.path.join(spatial_cache_dir, "poseidon_spatial_data")
    if not os.path.exists(spatial_base) or not os.path.exists(os.path.join(spatial_base, "shapefiles", "mpa_wpp711.shp")):
        target_zip = os.path.join(spatial_cache_dir, "poseidon_spatial.zip")
        import gdown
        print("Mengunduh bundle shapefile spasial dari Google Drive (ID 1Hi05v9z0o397NGZer0YxcEwISpNzrne9)...")
        gdown.download(id='1Hi05v9z0o397NGZer0YxcEwISpNzrne9', output=target_zip, quiet=True)
        if os.path.exists(target_zip):
            print("Mengekstrak bundle shapefile spasial...")
            with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                zip_ref.extractall(spatial_base)

    return spatial_base

def load_hotspot_grids():
    cache_path = os.path.join("automation", "spatial", "historical_hotspot_grids.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_csv = os.path.join("automation", "temp_hist.csv")
    try:
        import gdown
        print("Mengunduh dataset referensi hotspot dari Google Drive (ID 19L7mhcy_tjy9XYrLm5Gy-uXTKyfEQVMo)...")
        gdown.download(id="19L7mhcy_tjy9XYrLm5Gy-uXTKyfEQVMo", output=temp_csv, quiet=True)
        if os.path.exists(temp_csv):
            df_h = pd.read_csv(temp_csv, usecols=['lat', 'lon', 'historical_hotspot'])
            df_h['grid_lat'] = (df_h['lat'] // 0.1).astype(int)
            df_h['grid_lon'] = (df_h['lon'] // 0.1).astype(int)
            hotspots = df_h[df_h['historical_hotspot'] == 1][['grid_lat', 'grid_lon']].drop_duplicates().values.tolist()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(hotspots, f)
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            return hotspots
    except Exception:
        pass

    raise FileNotFoundError("Gagal memuat historical_hotspot_grids dari remote source.")

def integrate_spatial_and_temporal_gee(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    if df.empty:
        return df

    print(f"Mengintegrasikan fitur spasial GIS & oseanografi untuk {len(df):,} deteksi kapal...")
    spatial_base = ensure_spatial_bundle()
    spatial_cache_dir = os.path.join("automation", "spatial")

    shp_dir = os.path.join(spatial_base, "shapefiles")
    if not os.path.exists(shp_dir):
        shp_dir = spatial_base

    ports_csv = os.path.join(spatial_base, "ports", "fishing_ports_wpp711.csv")
    if not os.path.exists(ports_csv):
        ports_csv = os.path.join(spatial_base, "fishing_ports_wpp711.csv")

    seiz_csv = os.path.join(spatial_base, "labels", "iuu_seizure_records_2023_2025.csv")
    if not os.path.exists(seiz_csv):
        seiz_csv = os.path.join(spatial_base, "iuu_seizure_records_2023_2025.csv")

    mpa_shp = os.path.join(shp_dir, "mpa_wpp711.shp")
    eez_shp = os.path.join(shp_dir, "zee_wpp711.shp")

    gdf_mpa = gpd.read_file(mpa_shp).to_crs('EPSG:4326')
    gdf_eez = gpd.read_file(eez_shp).to_crs('EPSG:4326')
    gdf_eez['geometry'] = gdf_eez.geometry.apply(fill_holes)
    df_ports = pd.read_csv(ports_csv)
    df_seiz = pd.read_csv(seiz_csv)

    print("Menghitung jarak bertanda Kawasan Konservasi (MPA) & Batas ZEE...")
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326'
    )
    gdf_points_proj = gdf_points.to_crs('EPSG:3857')

    mpa_proj = gdf_mpa.to_crs('EPSG:3857')
    eez_proj = gdf_eez.to_crs('EPSG:3857')

    mpa_centroid_coords = np.array([(c.x, c.y) for c in mpa_proj.geometry.centroid if hasattr(c, 'x')])
    mpa_tree = cKDTree(mpa_centroid_coords)

    ports_lat, ports_lon = df_ports['lat'].values, df_ports['lon'].values

    dist_mpa_list = []
    dist_eez_list = []
    dist_port_list = []

    for _, row in gdf_points_proj.iterrows():
        lat_val = float(row['lat'])
        lon_val = float(row['lon'])
        point_geom = row['geometry']
        point_xy = np.array([point_geom.x, point_geom.y])

        _, closest_indices = mpa_tree.query(point_xy, k=min(5, len(mpa_centroid_coords)))
        if isinstance(closest_indices, (int, np.integer)):
            closest_indices = [closest_indices]
        candidates = mpa_proj.iloc[closest_indices]

        distances_mpa_m = candidates.boundary.distance(point_geom)
        min_dist_mpa_m = float(distances_mpa_m.min())
        inside_mpa = bool(candidates.contains(point_geom).any())
        d_mpa = -min_dist_mpa_m / 1000.0 if inside_mpa else min_dist_mpa_m / 1000.0

        dist_eez_m = float(eez_proj.boundary.distance(point_geom).min())
        inside_eez = bool(eez_proj.contains(point_geom).any())
        d_eez = -dist_eez_m / 1000.0 if not inside_eez else dist_eez_m / 1000.0

        d_port = float(haversine_km_vectorized(lat_val, lon_val, ports_lat, ports_lon).min())

        dist_mpa_list.append(round(d_mpa, 2))
        dist_eez_list.append(round(d_eez, 2))
        dist_port_list.append(round(d_port, 2))

    df['dist_to_nearest_mpa_km'] = dist_mpa_list
    df['dist_to_eez_boundary_km'] = dist_eez_list
    df['dist_to_nearest_port_km'] = dist_port_list

    print("Menghitung jarak Haversine ke titik penangkapan IUU historis...")
    seiz_dated = df_seiz.copy()
    if 'date' in seiz_dated.columns:
        seiz_dated['date'] = pd.to_datetime(seiz_dated['date'])
        seiz_dated['year'] = seiz_dated['date'].dt.year
        in_bbox = (
            seiz_dated['lon'].between(102.90, 111.05) &
            seiz_dated['lat'].between(-4.16, 7.74)
        )
        df_seiz_ref = seiz_dated[in_bbox & (seiz_dated['year'] <= 2025)].copy() if 'year' in seiz_dated.columns else seiz_dated[in_bbox].copy()
    else:
        df_seiz_ref = seiz_dated.copy()

    seiz_lat = df_seiz_ref['lat'].values
    seiz_lon = df_seiz_ref['lon'].values
    df['dist_to_nearest_seizure_km'] = compute_dist_to_seizure(df['lat'].values, df['lon'].values, seiz_lat, seiz_lon)

    df['grid_lat'] = np.floor(df['lat'] / 0.1).astype(int)
    df['grid_lon'] = np.floor(df['lon'] / 0.1).astype(int)

    grid_env_path = os.path.join(spatial_cache_dir, "wpp711_grid_environmental_full.csv")
    if not os.path.exists(grid_env_path):
        import gdown
        print("Mengunduh dataset grid parameter oseanografi dari Google Drive (ID 1yreRiRL8w0cJ8NYAZTDs2ILa61dJFhrX)...")
        gdown.download(id='1yreRiRL8w0cJ8NYAZTDs2ILa61dJFhrX', output=grid_env_path, quiet=True)

    grid_env = pd.read_csv(grid_env_path)
    env_cols = ['bathymetry', 'sst_mean', 'sst_std', 'current_speed_mean', 'current_speed_std', 'chlorophyll_mean']
    env_cols_full = ['grid_lat', 'grid_lon'] + [c for c in env_cols if c in grid_env.columns]

    print("Menggabungkan parameter batimetri, SST, arus, dan klorofil-a...")
    df = df.merge(
        grid_env[env_cols_full],
        on=['grid_lat', 'grid_lon'],
        how='left'
    )

    for col in env_cols:
        if col in df.columns and col in grid_env.columns:
            df[col] = df[col].fillna(float(grid_env[col].median()))

    print("Mencocokkan zona rawan pelanggaran historis (850 hotspot grid)...")
    hotspot_grids = load_hotspot_grids()
    hotspot_set = set(tuple(g) for g in hotspot_grids)
    df['historical_hotspot'] = df.apply(
        lambda r: 1 if (int(r['grid_lat']), int(r['grid_lon'])) in hotspot_set else 0,
        axis=1
    )

    df['quarter'] = int((target_date.month - 1) // 3 + 1)
    df['is_dark'] = 1
    df['ais_flag_country'] = 'UNKNOWN'
    df['ais_flag_country_encoded'] = 15

    return df.reset_index(drop=True)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def run_pipeline_scoring_gee(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    if df.empty:
        return df

    print(f"Memulai inferensi model ML untuk {len(df):,} deteksi kapal...")

    print("Mengestimasi panjang fisik kapal (Gradient Boosting Model)...")
    model_len = load_model_file("length_model")
    len_features = ['vv_intensity_db', 'vh_intensity_db', 'vv_vh_ratio', 'radar_intensity_diff', 'snr_db', 'background_clutter']
    df['length_m_new'] = np.clip(model_len.predict(df[len_features]), 5.0, 300.0)

    print("Mengestimasi indeks aktivitas penangkapan ikan (XGBoost Estimator)...")
    model_fish = load_model_file("fishing_model")
    fish_features = [
        'length_m_new', 'vv_intensity_db', 'vh_intensity_db', 'vv_vh_ratio',
        'bathymetry', 'sst_mean', 'sst_std', 'current_speed_mean',
        'current_speed_std', 'chlorophyll_mean', 'dist_to_nearest_port_km'
    ]
    df['fishing_score_new'] = np.clip(model_fish.predict(df[fish_features]), 0.0, 1.0)

    print("Menjalankan inferensi model ensemble PU-Learning GEE (13 Fitur)...")
    pipeline_models = load_model_file("gee_models")
    feature_cols = pipeline_models['features']

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Fitur wajib untuk GEE model tidak lengkap: {missing_features}.")

    X = df[feature_cols].astype(np.float32)
    w_lgb = float(pipeline_models['w_lgb'])

    lgb_preds = np.mean([_sigmoid(m.predict(X, raw_score=True)) for m in pipeline_models['models_lgb']], axis=0)

    import xgboost as xgb
    dtest = xgb.DMatrix(X)
    xgb_preds = np.mean([_sigmoid(m.predict(dtest, output_margin=True)) for m in pipeline_models['models_xgb']], axis=0)

    raw_scores = w_lgb * lgb_preds + (1.0 - w_lgb) * xgb_preds
    df['raw_risk_score'] = raw_scores
    batch_min, batch_max = float(raw_scores.min()), float(raw_scores.max())
    df['risk_score'] = (raw_scores - batch_min) / (batch_max - batch_min + 1e-9)

    q_hat = float(pipeline_models['q_hat'])
    df['conformal_flag'] = (raw_scores >= q_hat).astype(int)

    df = df.sort_values(by='raw_risk_score', ascending=False).reset_index(drop=True)
    df['rank_siklus'] = range(1, len(df) + 1)

    print("Menerapkan skema penentuan status Siaga baru berbasis kuartal referensi...")
    p70_threshold = get_p70_quarter_reference_threshold(target_date)

    status_list = []
    for _, row in df.iterrows():
        rk = int(row['rank_siklus'])
        raw_sc = float(row['raw_risk_score'])
        if rk in [1, 2, 3]:
            if raw_sc >= p70_threshold:
                status_list.append('SIAGA 1 (Prioritas)')
            else:
                status_list.append('SIAGA 1')
        elif rk in [4, 5, 6]:
            status_list.append('SIAGA 2')
        elif rk in [7, 8, 9, 10]:
            status_list.append('SIAGA 3')
        else:
            status_list.append('SIAGA 3 (Pasif)')
    df['status_siaga'] = status_list

    print("Mengekstrak kontribusi faktor risiko SHAP Tree Explainer untuk Top 10 target...")
    df['risk_factors_shap'] = ""
    if len(df) > 0:
        model_lgb_ref = pipeline_models['models_lgb'][0]
        X_sorted = df[feature_cols].astype(np.float32)
        top10_indices = list(range(min(10, len(df))))
        X_top10 = X_sorted.iloc[top10_indices]
        tree_contrib = model_lgb_ref.predict(X_top10, pred_contrib=True)[:, :-1]
        for i, idx in enumerate(top10_indices):
            contrib = tree_contrib[i]
            top_f_idx = np.argsort(-np.abs(contrib))[:3]
            reasons = []
            for fi in top_f_idx:
                fname = feature_cols[fi]
                flbl = FEATURE_LABELS_MAP.get(fname, fname)
                fval = float(df.loc[idx, fname])
                cval = float(contrib[fi])
                arah = "menaikkan risiko" if cval > 0 else "menurunkan risiko"
                reasons.append(f"{flbl} = {fval:.3f} ({arah}, kontribusi {cval:+.4f})")
            df.loc[idx, 'risk_factors_shap'] = " | ".join(reasons)

    df['year'] = target_date.year
    if 'timestamp' not in df.columns or df['timestamp'].isnull().all():
        df['timestamp'] = f"{target_date.strftime('%Y-%m-%d')} 00:00:00+00:00"
    df['is_dark'] = 1
    df['ais_flag_country'] = "UNKNOWN"
    df['ais_flag_country_encoded'] = 15

    return df

def fetch_max_ids(supabase: Client):
    max_id, max_seq, max_top_id = 0, 0, 0
    try:
        res_id = supabase.table("vessel_detections").select("id").order("id", desc=True).limit(1).execute()
        if res_id.data:
            max_id = int(res_id.data[0]['id'])
    except Exception:
        pass

    try:
        res_seq = supabase.table("satellite_passes").select("seq").order("seq", desc=True).limit(1).execute()
        if res_seq.data:
            max_seq = int(res_seq.data[0]['seq'])
    except Exception:
        pass

    try:
        res_top_id = supabase.table("top10_priorities").select("id").order("id", desc=True).limit(1).execute()
        if res_top_id.data:
            max_top_id = int(res_top_id.data[0]['id'])
    except Exception:
        pass

    return max_id, max_seq, max_top_id

def push_results_to_supabase(df: pd.DataFrame, target_date: datetime.date):
    if df.empty:
        print("Data kosong, tidak ada baris yang diunggah.")
        return

    supabase = get_supabase()
    max_id, max_seq, max_top_id = fetch_max_ids(supabase)
    print(f"Supabase Max IDs: vessel_detections={max_id}, satellite_passes={max_seq}, top10_priorities={max_top_id}")

    df['id'] = range(max_id + 1, max_id + 1 + len(df))
    vd_cols = [
        'id', 'pass_date', 'timestamp', 'lat', 'lon', 'risk_score', 'raw_risk_score',
        'status_siaga', 'rank_siklus', 'conformal_flag', 'risk_factors_shap',
        'length_m_new', 'fishing_score_new', 'vv_intensity_db', 'vh_intensity_db',
        'snr_db', 'dist_to_nearest_mpa_km', 'dist_to_eez_boundary_km',
        'dist_to_nearest_port_km', 'dist_to_nearest_seizure_km',
        'is_dark', 'ais_flag_country'
    ]
    df_vd = df[[c for c in vd_cols if c in df.columns]].replace({np.nan: None})
    records_vd = df_vd.to_dict(orient="records")

    print(f"Mengunggah {len(records_vd):,} baris ke vessel_detections...")
    batch_size = 500
    for i in range(0, len(records_vd), batch_size):
        supabase.table("vessel_detections").upsert(records_vd[i:i+batch_size]).execute()
        print(f"   Progress vessel_detections: {min(i+batch_size, len(records_vd))}/{len(records_vd)} baris.")

    df_top10 = df.sort_values(by='raw_risk_score', ascending=False).head(10).copy()
    df_top10['id'] = range(max_top_id + 1, max_top_id + 1 + len(df_top10))
    df_top10_push = df_top10[[c for c in vd_cols if c in df_top10.columns]].replace({np.nan: None})
    records_top10 = df_top10_push.to_dict(orient="records")
    print(f"Mengunggah {len(records_top10)} target prioritas ke top10_priorities...")
    supabase.table("top10_priorities").upsert(records_top10).execute()

    date_str = target_date.strftime("%Y-%m-%d")
    hour_val = str(df['timestamp'].iloc[0])[11:16] if 'timestamp' in df.columns else '00:00'

    pass_record = {
        'seq': max_seq + 1,
        'pass_date': date_str,
        'year': int(target_date.year),
        'month': int(target_date.month),
        'time': hour_val,
        'total_detections': int(len(df)),
        'siaga_1_count': int((df['status_siaga'].isin(['SIAGA 1 (Prioritas)', 'SIAGA 1'])).sum()),
        'siaga_2_count': int((df['status_siaga'] == 'SIAGA 2').sum()),
        'siaga_3_count': int((df['status_siaga'].isin(['SIAGA 3', 'SIAGA 3 (Pasif)'])).sum()),
        'dark_count': int((df['is_dark'] == 1).sum())
    }
    print(f"Mengunggah ringkasan siklus satelit ({date_str}) ke satellite_passes...")
    supabase.table("satellite_passes").upsert([pass_record], on_conflict="pass_date").execute()
    print("Selesai mengunggah seluruh data ke 3 tabel Supabase.")

def run_daily_automation(target_date: datetime.date = None):
    if target_date is not None:
        dates_to_check = [target_date]
    else:
        today = datetime.date.today()
        dates_to_check = [today - datetime.timedelta(days=i) for i in range(3, 0, -1)]

    supabase = get_supabase()
    processed_dates = set()
    
    if not dates_to_check:
        return

    try:
        start_date_str = min(dates_to_check).strftime("%Y-%m-%d")
        end_date_str = max(dates_to_check).strftime("%Y-%m-%d")
        resp = supabase.table("satellite_passes") \
            .select("pass_date") \
            .gte("pass_date", start_date_str) \
            .lte("pass_date", end_date_str) \
            .execute()
        if resp.data:
            processed_dates = {row['pass_date'] for row in resp.data}
    except Exception as e:
        print(f"Gagal mengambil riwayat satellite_passes dari Supabase: {e}")

    processed_any = False
    for d in dates_to_check:
        d_str = d.strftime("%Y-%m-%d")
        if d_str in processed_dates:
            print(f"Tanggal {d_str} sudah pernah diproses dan ada di Supabase, dilewati (Skip).")
            continue

        print(f"\nMemulai eksekusi POSEIDON GEE Pipeline untuk tanggal: {d_str}")
        t_start = time.time()
        df_raw = scrape_gee_daily(d)
        
        if df_raw.empty:
            print(f"Tidak ada deteksi kapal / scene satelit pada tanggal {d_str} di GEE. (Akan dicek ulang esok hari jika masih dalam window 3 hari).")
            continue

        processed_any = True
        df_feat = integrate_spatial_and_temporal_gee(df_raw, d)
        df_scored = run_pipeline_scoring_gee(df_feat, d)
        push_results_to_supabase(df_scored, d)
        print(f"Eksekusi pipeline untuk {d_str} selesai dengan sukses dalam {time.time() - t_start:.1f} detik.")

    if not processed_any and target_date is None:
        print("\nTidak ada siklus satelit baru yang berhasil diproses hari ini (semua sudah up-to-date atau GEE belum update).")

if __name__ == "__main__":
    run_daily_automation()