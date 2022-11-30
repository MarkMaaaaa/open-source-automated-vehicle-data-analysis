clear;
clc;
str = ['D:\Project\AMS\Data\ASta_040719_platoon10.csv'];
data = xlsread(str);
% 1.time 2.speed1 3.lat1 4.lon1 
% 9.speed2 10.lat2 11.lon2
TIme = data(7:end,1);
v_pre = data(7:end,2);
v_fol = data(7:end,9);
lat_pre = data(7:end,3);lon_pre = data(7:end,4);
lat_fol = data(7:end,10);lon_fol = data(7:end,11);
mstruct=defaultm('mercator');%定义椭球体长轴，椭率，坐标原点
mstruct.geoid=[6378137,0.0818191908426215];
mstruct.origin=[0,0,0];
mstruct=defaultm(mstruct);
[x_pre,y_pre] =projfwd(mstruct,lat_pre,lon_pre);
[x_fol,y_fol] =projfwd(mstruct,lat_fol,lon_fol);
dis = sqrt((x_pre-x_fol).^2+(y_pre-y_fol).^2);
v_diff = v_pre(1:end-1)-v_fol(1:end-1);
a_fol = diff(v_fol);







% plot(v_pre,v_fol);
plot(x_pre,y_pre,x_fol,y_fol);




