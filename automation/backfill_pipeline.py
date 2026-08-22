import os
import sys
import datetime
import asyncio
import gfwapiclient as gfw
import pandas as pd
import numpy as np
import ee
import joblib
import geopandas as gpd
from supabase import create_client, Client
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Import functions from daily_pipeline so we don't repeat code
from daily_pipeline import (
    extract_gee_features, 
    integrate_spatial_features, 
    integrate_temporal_features, 
    run_predictions, 
    push_to_supabase,
    SUPABASE_URL,
    SUPABASE_KEY,
    GFW_API_TOKEN
)

async def scrape_gfw_backfill(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    print(f"Scraping GFW from {start_date_str} to {end_date_str}...")
    if not GFW_API_TOKEN:
        print("Warning: GFW_API_TOKEN not found. Returning empty DataFrame.")
        return pd.DataFrame()
        
    client = gfw.Client(access_token=GFW_API_TOKEN)
    
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
            start_date=start_date_str,
            end_date=end_date_str,
            spatial_resolution="HIGH",
            geojson=geojson
        )
        data_list = result.data()
    except Exception as e:
        print(f"GFW API Error: {e}")
        return pd.DataFrame()
        
    if not data_list:
        print("No detections found for this period.")
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    if 'latitude' in df.columns:
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pass_date'] = df['timestamp'].dt.date
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['hour_utc'] = df['timestamp'].dt.hour
    
    # Feature engineering dasar
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(float)
    df['quarter'] = df['timestamp'].dt.quarter.astype(float)
    
    if 'vessel_id' in df.columns:
        df = df.rename(columns={'vessel_id': 'mmsi'})
    else:
        df['mmsi'] = 0.0
        
    if 'length' in df.columns:
        df = df.rename(columns={'length': 'length_m'})
    else:
        df['length_m'] = 0.0
        
    for col in ['vv_intensity_db', 'vh_intensity_db', 'background_clutter', 'snr_db']:
        if col not in df.columns:
            df[col] = 0.0
            
    df['radar_intensity_diff'] = df['vv_intensity_db'] - df['vh_intensity_db']
    df['vv_vh_ratio'] = df['vv_intensity_db'] / (df['vh_intensity_db'].replace(0, 1e-5))
    
    # Fill defaults
    df['presence_score'] = 1.0
    df['scene_id'] = "BACKFILL_SAR"
    df['matched_category'] = "Unknown"
    df['source_yyyymm'] = df['year'] * 100 + df['month']
    df['pipeline_ver'] = "V2_BACKFILL"
    df['ais_flag_country'] = "Unknown"
    df['ais_vessel_type'] = "Unknown"
    df['label'] = "Unknown"
    df['status_siaga'] = "SIAGA 3 (Pemantauan Pasif)"
    df['pu_flag'] = 0.0
    
    return df

def main():
    start_date_str = "2026-07-01"
    # Batas aman API biasanya maksimal sebulan atau rentang waktu tertentu,
    # Kita tarik dari Juli awal sampai 19 Agustus 2026
    end_date_str = "2026-08-19"
    
    df = asyncio.run(scrape_gfw_backfill(start_date_str, end_date_str))
    if df.empty:
        print("Selesai. Tidak ada data di rentang waktu ini.")
        return
        
    print(f"Berhasil menarik {len(df)} kapal dari periode {start_date_str} - {end_date_str}.")
    
    # Eksekusi pipeline (sama persis dengan harian)
    df = extract_gee_features(df)
    df = integrate_spatial_features(df)
    
    # Note: temporal features butuh target_date, kita set ke tanggal terakhir rentang ini saja
    # sebagai pendekatan (karena ini backfill)
    target_date = datetime.date(2026, 8, 19)
    df = integrate_temporal_features(df, target_date)
    df = run_predictions(df, target_date)
    
    # Push ke Supabase
    push_to_supabase(df)
    print("BACKFILL POSEIDON V2 SELESAI!")

if __name__ == "__main__":
    main()
