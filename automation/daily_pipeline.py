import os
import sys
import json
import datetime
import asyncio
import time
import io
import requests
import urllib.request
import joblib
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.vectorized import contains as shapely_contains
from supabase import create_client, Client
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GFW_API_TOKEN = os.environ.get("GFW_API_TOKEN")

HF_BASE_URL = "https://huggingface.co/JullMol/POSEIDON/resolve/main/POSEIDON_Model"
HF_MODEL_URLS = {
    "gfw_models": f"{HF_BASE_URL}/GFW/poseidon_models_gfw.pkl",
    "gee_models": f"{HF_BASE_URL}/GEE/poseidon_models_gee.pkl",
    "isolation_forest": f"{HF_BASE_URL}/GEE/isolation_forest_model.pkl",
    "length_model": f"{HF_BASE_URL}/model_length_gb.pkl",
    "fishing_model": f"{HF_BASE_URL}/model_fishing_score_xgboost.pkl"
}

MODEL_CACHE_DIR = "automation/models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

FEATURE_LABELS_MAP = {
    'vv_intensity_db': 'Intensitas radar VV',
    'vh_intensity_db': 'Intensitas radar VH',
    'radar_intensity_diff': 'Selisih intensitas VV-VH',
    'background_clutter': 'Clutter latar radar',
    'snr_db': 'Signal-to-noise ratio',
    'length_m_new': 'Panjang kapal',
    'dist_to_nearest_mpa_km': 'Jarak ke kawasan konservasi',
    'dist_to_eez_boundary_km': 'Jarak ke batas ZEE',
    'dist_to_nearest_port_km': 'Jarak ke pelabuhan',
    'dist_to_nearest_seizure_km': 'Jarak ke lokasi penyitaan historis',
    'fishing_score_new': 'Skor aktivitas penangkapan ikan',
    'historical_hotspot': 'Riwayat hotspot area',
    'quarter': 'Kuartal musim'
}

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required to connect to Supabase.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_reference_thresholds(target_date: datetime.date, supabase_client=None) -> dict:
    curr_q = int((target_date.month - 1) // 3 + 1)
    ref_year = target_date.year - 1
    date_start = f"{ref_year}-01-01"
    date_end = f"{ref_year}-12-31"

    fallback = {
        1: {'q75': 0.00095800, 'q90': 0.00327450, 'q_hat': 0.00080000},
        2: {'q75': 0.00126175, 'q90': 0.00846870, 'q_hat': 0.00095000},
        3: {'q75': 0.00112400, 'q90': 0.00582100, 'q_hat': 0.00090000},
        4: {'q75': 0.00108600, 'q90': 0.00491300, 'q_hat': 0.00085000},
    }

    try:
        if supabase_client is None:
            supabase_client = get_supabase()
        resp = supabase_client.table("vessel_detections") \
            .select("raw_risk_score, pass_date") \
            .gte("pass_date", date_start) \
            .lte("pass_date", date_end) \
            .execute()
        rows = resp.data if resp.data else []
        if not rows:
            print(f"Tidak ada data referensi {ref_year} di Supabase, pakai fallback thresholds.")
            return fallback[curr_q]

        df_ref = pd.DataFrame(rows)
        df_ref['quarter'] = pd.to_datetime(df_ref['pass_date']).dt.quarter
        sub = df_ref[df_ref['quarter'] == curr_q]['raw_risk_score'].dropna().values

        if len(sub) < 10:
            print(f"Data referensi Q{curr_q} {ref_year} terlalu sedikit ({len(sub)} baris), pakai fallback.")
            return fallback[curr_q]

        q75 = float(np.percentile(sub, 75))
        q90 = float(np.percentile(sub, 90))
        q_hat = float(np.percentile(sub, 69))
        print(f"Thresholds dinamis Q{curr_q} {ref_year}: q75={q75:.8f}, q90={q90:.8f}, q_hat={q_hat:.8f} (n={len(sub)})")
        return {'q75': q75, 'q90': q90, 'q_hat': q_hat}

    except Exception as e:
        print(f"Gagal ambil thresholds dari Supabase: {e}. Pakai fallback thresholds.")
        return fallback[curr_q]

def load_hf_model(model_key: str):
    url = HF_MODEL_URLS[model_key]
    filename = url.split('/')[-1]
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    
    if os.path.exists(local_path):
        print(f"Loading {model_key} dari cache: {local_path}")
        return joblib.load(local_path)
    
    print(f"Downloading {model_key} dari HuggingFace: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open(local_path, 'wb') as f:
                f.write(content)
            return joblib.load(io.BytesIO(content))
    except Exception as e:
        print(f"Gagal download model {model_key} dari HuggingFace: {e}")
        raise e

def apply_phase1_cleaning_filters(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
        
    spatial_dir = "automation/spatial"
    os.makedirs(spatial_dir, exist_ok=True)
    wpp_geojson_path = os.path.join(spatial_dir, "wpp711_big_resmi.geojson")
    local_data_path = os.path.join("data", "wpp711_geojson.json")
    
    wpp711_geom = None
    if os.path.exists(local_data_path):
        try:
            with open(local_data_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if 'features' in d and len(d['features']) > 0:
                wpp711_geom = shape(d['features'][0]['geometry'])
            elif 'coordinates' in d:
                wpp711_geom = shape(d)
        except Exception:
            pass
            
    if wpp711_geom is None:
        if not os.path.exists(wpp_geojson_path):
            print("Mengunduh batas polygon resmi WPP 711 dari BIG Satu Peta...")
            big_url = "https://kspservices.big.go.id/satupeta/rest/services/PUBLIK/SUMBER_DAYA_ALAM_DAN_LINGKUNGAN/MapServer/21/query"
            params = {"where": "namobj = 'WPP-RI 711'", "outFields": "*", "f": "geojson"}
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                res = requests.get(big_url, params=params, headers=headers, timeout=60)
                res.raise_for_status()
                with open(wpp_geojson_path, "w", encoding="utf-8") as f:
                    f.write(res.text)
            except Exception as e:
                print(f"Peringatan: Gagal unduh geojson BIG ({e}), fallback ke boundary shapefile.")
                wpp_geojson_path = os.path.join(spatial_dir, "poseidon_spatial_data", "shapefiles", "zee_wpp711.shp")
                
        if wpp_geojson_path.endswith(".shp"):
            wpp_gdf = gpd.read_file(wpp_geojson_path)
            wpp711_geom = wpp_gdf.geometry.union_all()
        else:
            with open(wpp_geojson_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if 'features' in d and len(d['features']) > 0:
                wpp711_geom = shape(d['features'][0]['geometry'])
            elif 'coordinates' in d:
                wpp711_geom = shape(d)
    
    initial_count = len(df_raw)
    mask_poly = shapely_contains(wpp711_geom, df_raw['lon'].values, df_raw['lat'].values)
    df_clean = df_raw[mask_poly].copy()
    print(f"Filter polygon WPP 711 BIG: {initial_count:,} -> {len(df_clean):,} baris")
    
    if 'presence_score' in df_clean.columns:
        df_clean['presence_score'] = pd.to_numeric(df_clean['presence_score'], errors='coerce')
        count_pre = len(df_clean)
        df_clean = df_clean[df_clean['presence_score'] >= 0.7].copy()
        print(f"Filter presence_score >= 0.7: {count_pre:,} -> {len(df_clean):,} baris")
        
    if 'scene_id' in df_clean.columns:
        dedup_cols = ['scene_id', 'lat', 'lon']
    elif 'pass_date' in df_clean.columns:
        dedup_cols = ['pass_date', 'lat', 'lon']
    else:
        dedup_cols = ['lat', 'lon']
        
    count_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=dedup_cols).copy()
    print(f"Deduplikasi titik overlap ({dedup_cols}): {count_dedup:,} -> {len(df_clean):,} baris")
    
    if 'matched_category' in df_clean.columns:
        df_clean = df_clean[~df_clean['matched_category'].isin(['noisy_vessel', 'discrepancy'])].copy()
        df_clean['matched_category'] = df_clean['matched_category'].replace('seismic_vessel', 'seismicvessel')
        
    return df_clean.reset_index(drop=True)

async def scrape_gfw_async(target_date: datetime.date) -> pd.DataFrame:
    print(f"Mengambil data GFW API untuk tanggal: {target_date}")
    if not GFW_API_TOKEN:
        print("Peringatan: GFW_API_TOKEN tidak ditemukan.")
        return pd.DataFrame()
        
    try:
        import gfwapiclient as gfw
        client = gfw.Client(access_token=GFW_API_TOKEN)
        date_str = target_date.strftime("%Y-%m-%d")
        geojson = {
            "type": "Polygon",
            "coordinates": [[[102.90, -4.16], [111.05, -4.16], [111.05, 7.74], [102.90, 7.74], [102.90, -4.16]]]
        }
        result = await client.fourwings.create_sar_presence_report(
            start_date=date_str, end_date=date_str, spatial_resolution="HIGH", geojson=geojson
        )
        data_list = result.data()
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        if 'latitude' in df.columns: df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
        if 'date' in df.columns: df.rename(columns={'date': 'pass_date'}, inplace=True)
        
        df_cleaned = apply_phase1_cleaning_filters(df)
        return df_cleaned
    except Exception as e:
        print(f"Error GFW API: {e}")
        return pd.DataFrame()

def scrape_gee_fallback(target_date: datetime.date) -> pd.DataFrame:
    print(f"Mengambil data GEE fallback untuk tanggal: {target_date}")
    try:
        import ee
        ee_service_account = os.environ.get("EE_SERVICE_ACCOUNT")
        ee_private_key = os.environ.get("EE_PRIVATE_KEY")
        ee_project = os.environ.get("EE_PROJECT", "darvin-natuna-2025")
        
        if ee_service_account and ee_private_key:
            print("Autentikasi GEE via Service Account.")
            credentials = ee.ServiceAccountCredentials(ee_service_account, key_data=ee_private_key)
            ee.Initialize(credentials, project=ee_project)
        else:
            print("Autentikasi GEE via default credentials.")
            try:
                ee.Initialize(project=ee_project)
            except Exception:
                ee.Authenticate()
                ee.Initialize(project=ee_project)
            
        wpp711_geom = ee.Geometry.Rectangle([102.90, -4.16, 111.05, 7.74])
        date_str = target_date.strftime("%Y-%m-%d")
        next_date_str = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(wpp711_geom)
               .filterDate(date_str, next_date_str)
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
               
        n_scene = col.size().getInfo()
        if n_scene == 0:
            print("Tidak ada scene Sentinel-1 GRD IW di GEE pada tanggal ini.")
            return pd.DataFrame()
            
        print(f"Ditemukan {n_scene} scene Sentinel-1 di GEE.")
        scene_list = col.toList(n_scene)
        raw_candidates = []
        
        for i in range(n_scene):
            img = ee.Image(scene_list.get(i))
            scene_id = img.get('system:index').getInfo()
            tgl_scene = img.date().format('YYYY-MM-dd HH:mm:ss').getInfo()
            
            vv = img.select('VV')
            kandidat_mask = vv.gt(-3.0)
            blob = kandidat_mask.selfMask().connectedComponents(connectedness=ee.Kernel.plus(1), maxSize=32)
            ukuran_blob = blob.select('labels').connectedPixelCount(maxSize=32)
            blob_valid = blob.updateMask(ukuran_blob.gte(2).And(ukuran_blob.lte(20)))
            area_irisan = img.geometry().intersection(wpp711_geom, ee.ErrorMargin(1))
            
            vektor = blob_valid.select('labels').reduceToVectors(
                geometry=area_irisan, scale=40, geometryType='centroid', maxPixels=1e9, bestEffort=True
            )
            fitur_list = vektor.limit(3000).getInfo()['features']
            
            for f in fitur_list:
                lon, lat = f['geometry']['coordinates']
                raw_candidates.append({
                    'scene_id': scene_id,
                    'timestamp': tgl_scene,
                    'pass_date': date_str,
                    'lat': lat,
                    'lon': lon,
                    'img_ref': img
                })
                
        if not raw_candidates:
            return pd.DataFrame()
            
        df_cand = pd.DataFrame(raw_candidates)
        
        VESSEL_BUFFER_M, BG_INNER_M, BG_OUTER_M = 400, 200, 800
        extracted_rows = []
        
        for s_id in df_cand['scene_id'].unique():
            df_sub = df_cand[df_cand['scene_id'] == s_id].reset_index(drop=True)
            img_s1 = df_sub.iloc[0]['img_ref'].select(['VV', 'VH'])
            
            fc_vessel = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]).buffer(VESSEL_BUFFER_M), {'row_idx': idx})
                for idx, row in df_sub.iterrows()
            ])
            fc_bg = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([row['lon'], row['lat']]).buffer(BG_OUTER_M)
                    .difference(ee.Geometry.Point([row['lon'], row['lat']]).buffer(BG_INNER_M)),
                    {'row_idx': idx}
                ) for idx, row in df_sub.iterrows()
            ])
            
            v_stats = img_s1.reduceRegions(collection=fc_vessel, reducer=ee.Reducer.mean(), scale=10)
            bg_stats = img_s1.select('VV').reduceRegions(collection=fc_bg, reducer=ee.Reducer.mean(), scale=10)
            
            v_map = {f['properties']['row_idx']: f['properties'] for f in v_stats.toList(v_stats.size()).getInfo()}
            bg_map = {f['properties']['row_idx']: f['properties'] for f in bg_stats.toList(bg_stats.size()).getInfo()}
            
            for idx, row in df_sub.iterrows():
                v = v_map.get(idx, {})
                bg = bg_map.get(idx, {})
                vv_val = v.get('VV', np.nan)
                vh_val = v.get('VH', np.nan)
                bg_val = bg.get('mean', np.nan)
                vv_vh = (vv_val - vh_val) if (vv_val is not None and vh_val is not None) else 0.0
                snr = (vv_val - bg_val) if (vv_val is not None and bg_val is not None) else 0.0
                
                row_dict = row.to_dict()
                row_dict.pop('img_ref', None)
                row_dict.update({
                    'vv_intensity_db': vv_val,
                    'vh_intensity_db': vh_val,
                    'vv_vh_ratio': vv_vh,
                    'radar_intensity_diff': vv_vh,
                    'background_clutter': bg_val,
                    'snr_db': snr
                })
                extracted_rows.append(row_dict)
                
        return pd.DataFrame(extracted_rows)
    except Exception as e:
        print(f"Error GEE Fallback: {e}")
        return pd.DataFrame()

def load_vessel_data(target_date: datetime.date):
    df_gfw = asyncio.run(scrape_gfw_async(target_date))
    if not df_gfw.empty:
        print(f"Data GFW termuat: {len(df_gfw)} deteksi.")
        return df_gfw, "GFW"
        
    print("Data GFW kosong, beralih ke GEE fallback.")
    df_gee = scrape_gee_fallback(target_date)
    if not df_gee.empty:
        print(f"Data GEE termuat: {len(df_gee)} kandidat kapal.")
        return df_gee, "GEE"
        
    print("Tidak ada deteksi kapal dari GFW maupun GEE.")
    return pd.DataFrame(), "NONE"

def haversine_km_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2_arr, lon2_arr = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2_arr - lat1
    dlon = lon2_arr - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def integrate_spatial_and_temporal(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    if df.empty: return df
    print("Menghitung fitur spasial dan lingkungan.")
    
    spatial_cache_dir = "automation/spatial"
    os.makedirs(spatial_cache_dir, exist_ok=True)
    zip_path = os.path.join(spatial_cache_dir, "poseidon_spatial.zip")
    extract_dir = os.path.join(spatial_cache_dir, "poseidon_spatial_data")
    
    if not os.path.exists(extract_dir):
        print("Mengunduh data referensi spasial.")
        import gdown, zipfile
        gdown.download(id="1Hi05v9z0o397NGZer0YxcEwISpNzrne9", output=zip_path, quiet=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    ports = pd.read_csv(os.path.join(extract_dir, "ports", "fishing_ports_wpp711.csv"))
    seizures = pd.read_csv(os.path.join(extract_dir, "labels", "iuu_seizure_records_2023_2025.csv"))
    mpa = gpd.read_file(os.path.join(extract_dir, "shapefiles", "mpa_wpp711.shp"))
    eez = gpd.read_file(os.path.join(extract_dir, "shapefiles", "zee_wpp711.shp"))
    
    lat_col = 'Latitude' if 'Latitude' in ports.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in ports.columns else 'lon'
    ports_lat, ports_lon = ports[lat_col].values, ports[lon_col].values
    seiz_lat, seiz_lon = seizures['lat'].values, seizures['lon'].values
    
    mpa_union = mpa.geometry.unary_union
    eez_union = eez.geometry.unary_union
    
    dist_mpa, dist_eez, dist_port, dist_seiz = [], [], [], []
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        pt = Point(lon, lat)
        dist_port.append(haversine_km_vectorized(lat, lon, ports_lat, ports_lon).min())
        dist_seiz.append(haversine_km_vectorized(lat, lon, seiz_lat, seiz_lon).min())
        pt_gdf = gpd.GeoSeries([pt], crs="EPSG:4326")
        dist_mpa.append(pt_gdf.distance(mpa_union).min() * 111.0)
        dist_eez.append(pt_gdf.distance(eez_union).min() * 111.0)
        
    df['dist_to_nearest_mpa_km'] = dist_mpa
    df['dist_to_eez_boundary_km'] = dist_eez
    df['dist_to_nearest_port_km'] = dist_port
    df['dist_to_nearest_seizure_km'] = dist_seiz
    
    df['grid_lat'] = np.floor(df['lat'] / 0.25) * 0.25
    df['grid_lon'] = np.floor(df['lon'] / 0.25) * 0.25
    
    grid_env_path = os.path.join(spatial_cache_dir, "wpp711_grid_environmental_full.csv")
    if not os.path.exists(grid_env_path):
        import gdown
        gdown.download(id='1yreRiRL8w0cJ8NYAZTDs2ILa61dJFhrX', output=grid_env_path, quiet=True)
    if os.path.exists(grid_env_path):
        grid_env = pd.read_csv(grid_env_path)
        env_cols = ['grid_lat', 'grid_lon', 'bathymetry', 'sst_mean', 'sst_std', 'current_speed_mean', 'current_speed_std', 'chlorophyll_mean']
        env_cols = [c for c in env_cols if c in grid_env.columns]
        df = df.merge(grid_env[env_cols], on=['grid_lat', 'grid_lon'], how='left')
    
    env_fallbacks = {'bathymetry': -50.0, 'sst_mean': 29.5, 'sst_std': 0.5, 'current_speed_mean': 0.35, 'current_speed_std': 0.1, 'chlorophyll_mean': 0.25}
    for col, val in env_fallbacks.items():
        if col not in df.columns: df[col] = val
        df[col] = df[col].fillna(val)
        
    df['historical_hotspot'] = 0
    df['quarter'] = int((target_date.month - 1) // 3 + 1)
    return df

def run_pipeline_scoring(df: pd.DataFrame, schema: str, target_date: datetime.date) -> pd.DataFrame:
    if df.empty: return df
    print(f"Menjalankan model scoring untuk skema: {schema}")
    
    model_len = load_hf_model("length_model")
    model_fish = load_hf_model("fishing_model")
    
    if 'vv_vh_ratio' not in df.columns:
        df['vv_vh_ratio'] = df['vv_intensity_db'] - df['vh_intensity_db']
    if 'radar_intensity_diff' not in df.columns:
        df['radar_intensity_diff'] = df['vv_vh_ratio']
        
    len_features = ['vv_intensity_db', 'vh_intensity_db', 'vv_vh_ratio', 'radar_intensity_diff', 'snr_db', 'background_clutter']
    for c in len_features:
        if c not in df.columns: df[c] = 0.0
        
    df['length_m_new'] = model_len.predict(df[len_features])
    df['length_m'] = df['length_m_new']
    
    fish_features = [
        'length_m_new', 'vv_intensity_db', 'vh_intensity_db', 'vv_vh_ratio',
        'bathymetry', 'sst_mean', 'sst_std', 'current_speed_mean',
        'current_speed_std', 'chlorophyll_mean', 'dist_to_nearest_port_km'
    ]
    for c in fish_features:
        if c not in df.columns: df[c] = 0.0
        
    df['fishing_score_new'] = np.clip(model_fish.predict(df[fish_features]), 0, 1)
    df['fishing_score'] = df['fishing_score_new']
    
    if schema == "GEE":
        print("Menjalankan filter Isolation Forest untuk kandidat GEE.")
        iforest = load_hf_model("isolation_forest")
        if hasattr(iforest, 'feature_names_in_'):
            fitur_anomali = list(iforest.feature_names_in_)
        else:
            fitur_anomali = ['vv_intensity_db', 'vh_intensity_db', 'radar_intensity_diff', 'background_clutter', 'snr_db']
        for col in fitur_anomali:
            if col not in df.columns: df[col] = 0.0
            
        preds_anom = iforest.predict(df[fitur_anomali])
        df['is_vessel'] = (preds_anom == 1).astype(int)
        df = df[df['is_vessel'] == 1].reset_index(drop=True)
        if df.empty:
            print("Seluruh kandidat tereliminasi sebagai noise oleh Isolation Forest.")
            return df
            
        pipeline_models = load_hf_model("gee_models")
    else:
        if 'matching_score' not in df.columns: df['matching_score'] = 0.0
        if 'ais_flag_country_encoded' not in df.columns: df['ais_flag_country_encoded'] = 0
        pipeline_models = load_hf_model("gfw_models")
        
    feature_cols = pipeline_models['features']
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0
        
    X = df[feature_cols].astype(np.float32)
    w_lgb = float(pipeline_models.get('w_lgb', 0.5))
    
    lgb_preds = np.mean([_sigmoid(m.predict(X, raw_score=True)) for m in pipeline_models['models_lgb']], axis=0)
    
    import xgboost as xgb
    dtest = xgb.DMatrix(X)
    xgb_preds = np.mean([_sigmoid(m.predict(dtest, output_margin=True)) for m in pipeline_models['models_xgb']], axis=0)
    
    raw_scores = w_lgb * lgb_preds + (1.0 - w_lgb) * xgb_preds
    df['raw_risk_score'] = raw_scores
    batch_min, batch_max = float(raw_scores.min()), float(raw_scores.max())
    df['risk_score'] = (raw_scores - batch_min) / (batch_max - batch_min + 1e-9)

    thr = get_reference_thresholds(target_date)
    if thr is None:
        q75_raw, q90_raw = float(np.percentile(raw_scores, 75)), float(np.percentile(raw_scores, 90))
        q_hat_raw = q90_raw
    else:
        q75_raw, q90_raw = float(thr['q75']), float(thr['q90'])
        q_hat_raw = float(thr.get('q_hat', q90_raw))

    df['status_siaga'] = np.select(
        [raw_scores >= q90_raw, (raw_scores >= q75_raw) & (raw_scores < q90_raw)],
        ['SIAGA 1 (Prioritas Verifikasi)', 'SIAGA 2 (Pengamatan Terarah)'],
        default='SIAGA 3 (Pemantauan Pasif)'
    )

    df['conformal_flag'] = (raw_scores >= q_hat_raw).astype(int)
    
    df = df.sort_values(by='risk_score', ascending=False).reset_index(drop=True)
    df['rank_siklus'] = range(1, len(df) + 1)
    
    df['risk_factors_shap'] = ""
    top10_indices = df.head(10).index
    if len(top10_indices) > 0:
        model_lgb_ref = pipeline_models['models_lgb'][0]
        X_top10 = X.loc[top10_indices]
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
    if 'is_dark' not in df.columns:
        df['is_dark'] = 1
    if 'ais_flag_country' not in df.columns:
        df['ais_flag_country'] = "UNKNOWN"
        
    return df

def fetch_max_ids(supabase: Client):
    max_id, max_seq = 0, 0
    try:
        res_id = supabase.table("vessel_detections").select("id").order("id", desc=True).limit(1).execute()
        if res_id.data: max_id = int(res_id.data[0]['id'])
    except Exception: pass
    
    try:
        res_seq = supabase.table("satellite_passes").select("seq").order("seq", desc=True).limit(1).execute()
        if res_seq.data: max_seq = int(res_seq.data[0]['seq'])
    except Exception: pass
    
    return max_id, max_seq

def push_results_to_supabase(df: pd.DataFrame, target_date: datetime.date):
    if df.empty:
        print("Tidak ada data untuk diunggah ke Supabase.")
        return
        
    supabase = get_supabase()
    date_str = target_date.strftime("%Y-%m-%d")

    try:
        supabase.table("vessel_detections").delete().eq("pass_date", date_str).execute()
        supabase.table("top10_priorities").delete().eq("pass_date", date_str).execute()
        supabase.table("satellite_passes").delete().eq("pass_date", date_str).execute()
    except Exception as e:
        print(f"Peringatan saat pembersihan data duplikat untuk {date_str}: {e}")

    max_id, max_seq = fetch_max_ids(supabase)
    print(f"Supabase Max ID: {max_id}, Max Seq: {max_seq}")
    
    df['id'] = range(max_id + 1, max_id + 1 + len(df))
    vd_cols = [
        'id', 'pass_date', 'timestamp', 'lat', 'lon', 'risk_score', 'raw_risk_score',
        'status_siaga', 'rank_siklus', 'conformal_flag', 'risk_factors_shap',
        'length_m_new', 'fishing_score_new', 'vv_intensity_db', 'vh_intensity_db',
        'snr_db', 'dist_to_nearest_mpa_km', 'dist_to_eez_boundary_km',
        'dist_to_nearest_port_km', 'dist_to_nearest_seizure_km', 'is_dark', 'ais_flag_country'
    ]
    df_vd = df[[c for c in vd_cols if c in df.columns]].replace({np.nan: None})
    records_vd = df_vd.to_dict(orient="records")
    
    print(f"Mengunggah {len(records_vd)} baris ke vessel_detections.")
    batch_size = 500
    for i in range(0, len(records_vd), batch_size):
        supabase.table("vessel_detections").upsert(records_vd[i:i+batch_size]).execute()
        
    df_top10 = df.sort_values(by='risk_score', ascending=False).head(10)
    df_top10_push = df_top10[[c for c in vd_cols if c in df_top10.columns]].replace({np.nan: None})
    records_top10 = df_top10_push.to_dict(orient="records")
    print(f"Mengunggah {len(records_top10)} baris prioritas ke top10_priorities.")
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
        'siaga_1_count': int((df['status_siaga'] == 'SIAGA 1 (Prioritas Verifikasi)').sum()),
        'siaga_2_count': int((df['status_siaga'] == 'SIAGA 2 (Pengamatan Terarah)').sum()),
        'siaga_3_count': int((df['status_siaga'] == 'SIAGA 3 (Pemantauan Pasif)').sum()),
        'dark_count': int((df['is_dark'] == 1).sum())
    }
    print("Mengunggah ringkasan lintasan ke satellite_passes.")
    supabase.table("satellite_passes").upsert([pass_record]).execute()
    print("Selesai mengunggah seluruh data ke Supabase.")

def check_date_exists_in_supabase(target_date: datetime.date) -> bool:
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            return False
        supabase = get_supabase()
        date_str = target_date.strftime("%Y-%m-%d")
        resp = supabase.table("satellite_passes").select("pass_date").eq("pass_date", date_str).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            return True
        return False
    except Exception as e:
        print(f"Peringatan: Gagal mengecek status tanggal di Supabase ({e}). Melanjutkan proses.")
        return False

if __name__ == "__main__":
    target_date = datetime.date.today() - datetime.timedelta(days=2)
    print(f"Eksekusi pipeline POSEIDON untuk tanggal: {target_date}")
    
    if check_date_exists_in_supabase(target_date):
        print(f"Data untuk tanggal {target_date} sudah ada di Supabase. Eksekusi dilewati (tidak ada pemrosesan ulang).")
    else:
        df_raw, schema = load_vessel_data(target_date)
        
        if not df_raw.empty:
            df_feat = integrate_spatial_and_temporal(df_raw, target_date)
            df_scored = run_pipeline_scoring(df_feat, schema, target_date)
            push_results_to_supabase(df_scored, target_date)
            print("Pipeline harian selesai diproses.")
        else:
            print("Tidak ada data deteksi yang ditemukan untuk diproses.")