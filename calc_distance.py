#coding:utf-8
#緯度経度から距離を算出する関数

import pandas as pd
from math import sin, cos, acos, radians


def latlng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)

def dist_on_sphere(pos0, pos1, radious=6378.137):
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*radious

#最短距離計算する関数
def pt_distance(name,id):
    #station : List型
    #observation : List型
    observation_point=getLL(id)
    station_points=get_listpoint(name)
    min_dist=9999999
    for i in station_points:
        tmp=dist_on_sphere(i,observation_point)
        if tmp<min_dist:
            min_dist=tmp

    return min_dist

#データからlist型緯度経度データを返す関数
def get_listpoint(name):
    data = pd.read_csv('data/points/'+name+'.csv')
    points = data[['longitude','latitude']].as_matrix()
    return [( v , k ) for v,k in points ]

#観測地点IDから緯度経度を返す関数
def getLL(id):
    df=pd.read_csv('data/observation_point.tsv',delimiter='\t')
    longitude=df[df['局ID'].isin([id])]['緯度（10進）'].as_matrix()[0]
    latitude=df[df['局ID'].isin([id])]['経度（10進）'].as_matrix()[0]
    return [longitude,latitude]


if __name__=='__main__':
    df=pd.read_csv('data/points/tyuou.csv')
    stpoint=get_listpoint(df)
    print(pt_distance(stpoint,33140514))


