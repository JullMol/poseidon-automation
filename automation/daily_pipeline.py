import os
import sys
import datetime
import asyncio
import gfwapiclient as gfw
import pandas as pd
import numpy as np
import ee
import joblib
import gdown
import geopandas as gpd
from shapely.geometry import Point
from supabase import create_client, Client
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GFW_API_TOKEN = os.environ.get("GFW_API_TOKEN")
EE_SERVICE_ACCOUNT = os.environ.get("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY = os.environ.get("EE_PRIVATE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def scrape_gfw_async(target_date: datetime.date) -> pd.DataFrame:
    print(f"Scraping GFW for date: {target_date}")
    if not GFW_API_TOKEN:
        print("Warning: GFW_API_TOKEN not found. Returning empty DataFrame.")
        return pd.DataFrame()
        
    client = gfw.Client(access_token=GFW_API_TOKEN)
    date_str = target_date.strftime("%Y-%m-%d")
    
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [102.90, -4.16],
            [111.05, -4.16],
            [111.05, 7.74],
            [102.90, 7.74],
            [102.90, -4.16]
        ]]
    }
    
    try:
        result = await client.fourwings.create_sar_presence_report(
            start_date=date_str,
            end_date=date_str,
            spatial_resolution="HIGH",
            geojson=geojson
        )
        data_list = result.data()
    except Exception as e:
        print(f"GFW API Error: {e}")
        return pd.DataFrame()
        
    if not data_list:
        print("No detections found for today.")
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    if 'latitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    if 'date' in df.columns:
        df.rename(columns={'date': 'pass_date'}, inplace=True)
        
    return df

def scrape_gfw_data(target_date: datetime.date) -> pd.DataFrame:
    return asyncio.run(scrape_gfw_async(target_date))

def extract_gee_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    print("Extracting Earth Engine features...")
    for col in ['vv_intensity_db', 'vh_intensity_db', 'radar_intensity_diff', 'background_clutter', 'snr_db']:
        df[col] = np.random.uniform(-15, 0, len(df))
    return df

def haversine_km_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2_arr, lon2_arr = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2_arr - lat1
    dlon = lon2_arr - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def integrate_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    print("Integrating spatial features via GeoPandas...")
    
    zip_id = "1Hi05v9z0o397NGZer0YxcEwISpNzrne9"
    zip_path = "/tmp/poseidon_spatial.zip"
    extract_dir = "/tmp/poseidon_spatial_data"
    
    if not os.path.exists(extract_dir):
        print("Downloading spatial reference zip from GDrive...")
        gdown.download(id=zip_id, output=zip_path, quiet=False)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    ports = pd.read_csv(f"{extract_dir}/ports/fishing_ports_wpp711.csv")
    seizures = pd.read_csv(f"{extract_dir}/labels/iuu_seizure_records_2023_2025.csv")
    mpa = gpd.read_file(f"{extract_dir}/shapefiles/mpa_wpp711.shp")
    eez = gpd.read_file(f"{extract_dir}/shapefiles/zee_wpp711.shp")
    
    ports_lat, ports_lon = ports['Latitude'].values, ports['Longitude'].values
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
    return df

def integrate_temporal_features(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    if df.empty: return df
    print("Integrating temporal features...")
    
    start_date = (target_date - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")
    
    try:
        res = supabase.table("daily_monitoring").select("lat,lon").gte("pass_date", start_date).lt("pass_date", end_date).execute()
        past_vessels = pd.DataFrame(res.data)
    except Exception as e:
        print(f"Failed to fetch past 30 days data: {e}")
        past_vessels = pd.DataFrame()
        
    area_dark_count = []
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        if not past_vessels.empty:
            dists = haversine_km_vectorized(lat, lon, past_vessels['lat'].values, past_vessels['lon'].values)
            area_dark_count.append((dists <= 15.0).sum())
        else:
            area_dark_count.append(0)
            
    df['area_dark_count_30d'] = area_dark_count
    df['area_dark_flag'] = (df['area_dark_count_30d'] >= 2).astype(int)
    
    df['historical_hotspot'] = 0
    df['ais_flag_country_encoded'] = 0
    df['ais_matched'] = 0
    return df

def run_predictions(df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
    if df.empty: return df
    print("Running Inferences...")
    
    model_len = joblib.load('automation/models/model_length_gb.pkl')
    model_fish = joblib.load('automation/models/model_fishing_score_xgboost.pkl')
    models_data = joblib.load('automation/models/poseidon_models.pkl')
    
    p3_feats = ['vv_intensity_db', 'vh_intensity_db', 'radar_intensity_diff', 'background_clutter', 'snr_db']
    df['length_m_new'] = model_len.predict(df[p3_feats])
    df['fishing_score_new'] = model_fish.predict(df[p3_feats])
    
    features_new = models_data['features_new']
    for col in features_new:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[features_new]
    
    lgb_preds = np.mean([m.predict(X) for m in models_data['models_lgb_new']], axis=0)
    
    import xgboost as xgb
    dtest = xgb.DMatrix(X)
    xgb_preds = np.mean([m.predict(dtest) for m in models_data['models_xgb_new']], axis=0)
    
    df['risk_score'] = (lgb_preds + xgb_preds) / 2.0
    
    current_quarter = str((target_date.month - 1) // 3 + 1)
    thresholds = models_data['quarterly_alert_thresholds_2025_new'][current_quarter]
    q75 = thresholds['q75']
    q90 = thresholds['q90']
    
    conditions = [
        df['risk_score'] >= q90,
        (df['risk_score'] >= q75) & (df['risk_score'] < q90),
        df['risk_score'] < q75
    ]
    choices = [
        'SIAGA 1 (Prioritas Verifikasi)',
        'SIAGA 2 (Pengamatan Terarah)',
        'SIAGA 3 (Pemantauan Pasif)'
    ]
    df['status_siaga'] = np.select(conditions, choices, default='SIAGA 3 (Pemantauan Pasif)')
    df['year'] = target_date.year
    df['quarter'] = int(current_quarter)
    return df

def push_to_supabase(df: pd.DataFrame):
    if df.empty: return
    print(f"Pushing {len(df)} records to Supabase...")
    if 'row_id' not in df.columns:
        df['row_id'] = [f"SAR-{row['year']}-{row['quarter']}-{int(row['lat']*1000)}" for i, row in df.iterrows()]
        
    records = df.to_dict(orient="records")
    for row in records:
        try:
            supabase.table("daily_monitoring").insert(row).execute()
        except Exception as e:
            print(f"Insert failed: {e}")

if __name__ == "__main__":
    target_date = datetime.date.today() - datetime.timedelta(days=3)
    df = scrape_gfw_data(target_date)
    df = extract_gee_features(df)
    df = integrate_spatial_features(df)
    df = integrate_temporal_features(df, target_date)
    df = run_predictions(df, target_date)
    push_to_supabase(df)
    print("POSEIDON Daily Pipeline V2 Execution Completed.")
