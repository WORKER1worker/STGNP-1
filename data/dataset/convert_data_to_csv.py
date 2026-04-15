# convert_data_to_csv.py
import pandas as pd
import numpy as np
import os

def convert_stations_to_csv(excel_path, output_csv):
    """
    转换站点信息Excel为CSV
    
    输入: Pali-Stations.xlsx
    输出: Pali-Stations.csv (仅保留PL01-PL24, 去除DNLG)
    """
    print("🔄 转换站点信息...")
    
    # 读取Excel
    df = pd.read_excel(excel_path)
    
    # 清理列名空格
    df.columns = df.columns.str.strip()
    
    # 重命名列 (处理中文列名)
    column_mapping = {
        '站点名称': 'station_id',
        '经度/Longitud(degree)': 'lon',
        '纬度/Latitude(degree)': 'lat',
        '海拔/Elvation': 'elevation'
    }
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    # 过滤只保留PL01-PL24 (去除DNLG)
    df = df[df['station_id'].str.startswith('PL')].reset_index(drop=True)
    
    # 清理数据: 去除海拔中的'm'单位
    if 'elevation' in df.columns:
        df['elevation'] = df['elevation'].astype(str).str.replace('m', '')
    
    # 排序: 确保顺序为PL01, PL02, ..., PL24
    # 注意: PL011实际应该是PL11
    df['station_id'] = df['station_id'].replace('PL011', 'PL11')
    
    # 自定义排序函数
    def sort_key(station_id):
        return int(station_id[2:])
    
    df['sort_order'] = df['station_id'].apply(sort_key)
    df = df.sort_values('sort_order').drop('sort_order', axis=1).reset_index(drop=True)
    
    # 只保留必要列 (d_spatial仅需经纬度)
    df = df[['station_id', 'lon', 'lat']]
    
    # 保存为CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✓ 站点信息转换完成")
    print(f"  输出: {output_csv}")
    print(f"  站点数: {len(df)}")
    print(f"  内容预览:\n{df.head()}\n")
    
    return df


def convert_soil_moisture_to_csv(txt_path, output_csv):
    """
    转换土壤水分TXT为CSV
    
    输入: SM_PL-30 minutes_10cm.txt
    输出: SM_PL-30 minutes_10cm.csv (仅保留时间和PL01-PL24列)
    
    注意: 处理PL11-01, PL11, PL12-01, PL12的重复列问题
    """
    print("🔄 转换土壤水分数据...")
    
    # 读取TXT (按空格/制表符分隔)
    df = pd.read_csv(txt_path, sep='\s+')
    
    print(f"  原始数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}\n")
    
    # 处理列名清理
    df.columns = df.columns.str.strip()
    
    # 重命名列 (PL011应为PL11)
    if 'PL011' in df.columns:
        df = df.rename(columns={'PL011': 'PL11'})
    
    # 处理PL11-01和PL12-01的重复列
    # 策略: 使用标准列(PL11, PL12)，删除-01版本
    cols_to_drop = []
    if 'PL11-01' in df.columns:
        cols_to_drop.append('PL11-01')
    if 'PL12-01' in df.columns:
        cols_to_drop.append('PL12-01')
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  ℹ️  已删除重复列: {cols_to_drop}\n")
    
    # 保留的列: 时间(yyyy,mm,dd,HH,MM,SS) + 传感器(PL01-PL24)
    sensor_cols = [col for col in df.columns if col.startswith('PL') and len(col) == 4]
    sensor_cols = sorted(sensor_cols, key=lambda x: int(x[2:]))
    
    # 删除不需要的列 (DNLG, Ave等)
    keep_cols = ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS'] + sensor_cols
    df = df[[col for col in keep_cols if col in df.columns]]
    
    print(f"  保留列: {df.columns.tolist()}")
    print(f"  处理后数据形状: {df.shape}\n")
    
    # 保存为CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✓ 土壤水分数据转换完成")
    print(f"  输出: {output_csv}")
    print(f"  行数: {len(df)}")
    print(f"  传感器数: {len(sensor_cols)}")
    print(f"  内容预览:\n{df.head()}\n")
    
    return df, sensor_cols


if __name__ == '__main__':
    # 定义路径 (SM文件位于SM子目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'SM')  # SM文件位于SM子目录
    
    stations_xlsx = os.path.join(base_dir, 'Pali-Stations.xlsx')
    stations_csv = os.path.join(base_dir, 'Pali-Stations.csv')
    
    soil_txt = os.path.join(base_dir, 'SM_PL-30 minutes_10cm.txt')
    soil_csv = os.path.join(base_dir, 'SM_PL-30 minutes_10cm.csv')
    
    # 验证输入文件是否存在
    if not os.path.exists(stations_xlsx):
        print(f"❌ 找不到文件: {stations_xlsx}")
        exit(1)
    if not os.path.exists(soil_txt):
        print(f"❌ 找不到文件: {soil_txt}")
        exit(1)
    
    print("="*70)
    print("📊 数据格式转化工具")
    print("="*70 + "\n")
    
    # 1. 转换站点信息
    stations_df = convert_stations_to_csv(stations_xlsx, stations_csv)
    
    # 2. 转换土壤水分数据
    soil_df, sensor_cols = convert_soil_moisture_to_csv(soil_txt, soil_csv)
    
    print("="*70)
    print("✅ 转化完成！")
    print("="*70)
    print(f"\n✓ 已生成CSV文件:")
    print(f"  - {stations_csv}")
    print(f"  - {soil_csv}")
    print(f"\n✓ 数据统计:")
    print(f"  - 站点数: {len(stations_df)}")
    print(f"  - 时间步数: {len(soil_df)}")
    print(f"  - 传感器列: {sensor_cols}")
