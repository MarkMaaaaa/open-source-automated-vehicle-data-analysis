clear;
clc;
[~,~,data1_0]=xlsread('D:\Project\AMS\Data\test2.xlsx',1);
[~,~,data1_1]=xlsread('D:\Project\AMS\Data\test2.xlsx',2);
[~,~,data1_2]=xlsread('D:\Project\AMS\Data\test2.xlsx',3);
% data1_0=xlsread('G:\a\test1.xlsx',1);
% data1_1=xlsread('G:\a\test1.xlsx',2);
% data1_2=xlsread('G:\a\test1.xlsx',3);
%%
%参数值（距离单位：m；速度单位：m/s）
value_v=0.1;%启动速度阈值
Hz=10;%采样频率
% data1_0=cell2mat(data1_0);
P_T_0=cell2mat(data1_0(:,2));
Time_1_0=str2num(P_T_0(:,end-7:end));
% start_v_1_0=cell2mat(data1_0(:,5));
% start0=find(start_v_1_0>0.1);
P_T_1=cell2mat(data1_1(:,2));
Time_1_1=str2num(P_T_1(:,end-7:end));
start_v_1_1=cell2mat(data1_1(:,5));
start1=find(start_v_1_1>0.1);
t_end_1=Time_1_1(start1(end));

P_T_2=cell2mat(data1_2(:,2));
Time_1_2=str2num(P_T_2(:,end-7:end));
start_v_1_2=cell2mat(data1_2(:,5));
start2=find(start_v_1_2>0.1);
t_end_2=Time_1_2(start2(end));

t_end=min(t_end_1,t_end_2);
% t_1_0=Time_1_0(start0(1));
t_1_1=Time_1_1(start1(1));
n_1_2=find(Time_1_2==t_1_1);
% t_1_2=Time_1_2(start2(1));
% timeend_1=Time_1_1(start1(end));
N=409;
% T_start=
%初始值
%%
%数据预处理
lat_1_0=cell2mat(data1_0(:,4));lon_1_0=cell2mat(data1_0(:,3));v_1_0=cell2mat(data1_0(:,5));
lat_1_1=cell2mat(data1_1(:,4));lon_1_1=cell2mat(data1_1(:,3));v_1_1=cell2mat(data1_1(:,5));
lat_1_2=cell2mat(data1_2(:,4));lon_1_2=cell2mat(data1_2(:,3));v_1_2=cell2mat(data1_2(:,5));
mstruct=defaultm('mercator');%定义椭球体长轴，椭率，坐标原点
mstruct.geoid=[ 6378137,0.0818191908426215];
mstruct.origin=[0,0,0];
mstruct=defaultm(mstruct);
[x_1_1,y_1_1] =projfwd(mstruct,lat_1_1,lon_1_1);
[x_1_2,y_1_2] =projfwd(mstruct,lat_1_2,lon_1_2);
min_v0=find(v_1_0>value_v);min_v1=find(v_1_1>value_v);
timestart_1=(min_v1(1)-1)/10+t_1_1;
% timeend_1=(min_v1(end)-1)/10+t_1_1;
% locstart_1_0=vpa((timestart_1-t_1_0)*10,3)+1;
% locend_1_0=vpa((timeend_1-t_1_0)*10,3)+1;
min_v2=find(v_1_2>value_v);
% timestart_2=min_v2(1);timeend_2=min_v2(end);
% timestart_1=vpa((min_v2(1)-1)/Hz+t_1_2);
% timeend_1=(min_v2(end)-1)/Hz+t_1_2;
% numstart_1_1=vpa((timestart_1-t_1_1)*10,3)+1;
% numend_1_1=vpa((timeend_1-t_1_1)*10,3)+1;
% numstart_1_2=vpa((timestart_1-t_1_2)*10,3)+1;
% numend_1_2=vpa((timeend_1-t_1_2)*10,3)+1;
% locstart_1_1=vpa((timestart_1-t_1_1)*Hz,3)+1;
% locend_1_1=vpa((timeend_1-t_1_1)*Hz,3)+1;
numstart_1_1=start1(1);
numend_1_1=find(Time_1_1==t_end);
numstart_1_2=n_1_2;
numend_1_2=n_1_2+numend_1_1-numstart_1_1;

L_1_1=x_1_1(numstart_1_1:numend_1_1);%数据1第1辆CAV位置
V_1_1=v_1_1(numstart_1_1:numend_1_1);%数据1第1辆CAV速度
L_1_2=x_1_2(numstart_1_2:numend_1_2);%数据1第2辆CAV位置
V_1_2=v_1_2(numstart_1_2:numend_1_2);%数据1第2辆CAV速度
difl_1=L_1_1-L_1_2;%距离差第一辆减第二辆
difv_1=V_1_1-V_1_2;%速度差第一辆减第二辆

%%
%参数设置
%OV model
num=30;
Td=1.4;d_safe=1;
num_P=3;
b=cell(num_P,1,num);bint=cell(num_P,2,num);r=cell(1482,1,num);rint=cell(1482,2,num);stats=cell(1,4,num);
for i=1:num;%delay步长
gap=difl_1(1:end-i)-V_1_2(1:end-i)*Td-d_safe;
velocity=V_1_2(1:end-i);
velocity_2=velocity.^2;
d_velocity=difv_1(1:end-i);
a1=diff(V_1_2(i:end)).*Hz;%下一个时刻的加速度
X=[gap,d_velocity,velocity_2];
[b1,bint1,r1,rint1,stats1] = regress(a1,X);
b(:,:,i)=num2cell(b1);
bint(:,:,i)=num2cell(bint1);
% r(:,:,i)=num2cell(r1);
% rint(:,:,i)=num2cell(rint1);
stats(:,:,i)=num2cell(stats1);
end
x_time=1:size(a1);

%%
%OVM model
x_sim=L_1_2(1);
dt=0.1;
v_sim=V_1_2(1);
X_sim=[];
V_sim=[];
A_sim=[];
value_state=cell2mat(stats(1,1,:));
num_max=find(value_state==max(value_state));
T_delay=num_max;%总体时间延迟为1.7s
k_g=cell2mat(b(1,1,num_max));
k_dv=cell2mat(b(2,1,num_max));
k_v=cell2mat(b(3,1,num_max));
% aa=k_g.*gap+k_v.*velocity+k_dv.*d_velocity;
aa=k_g.*gap+k_dv.*d_velocity;
vv=velocity+dt.*aa;
% plot(x_time,V_1_1(1:end-i),x_time,vv);figure;
% plot(x_time,V_1_1(1:end-i),x_time,V_1_2(1:end-i));

%仿真开始
a_sim=k_g.*gap(1)+k_v.*velocity_2(1)+k_dv.*d_velocity(1);
% a_sim=k_g.*gap(1)+k_dv.*d_velocity(1);
for ii=1:length(a1)-T_delay
    x_sim=x_sim+v_sim*dt+0.5*a_sim.*dt.^2;
    v_sim=v_sim+dt*a_sim;
    gap_sim=L_1_1(ii)-x_sim-v_sim*Td-d_safe;
    d_velocity_sim=V_1_1(ii)-v_sim;
    a_sim=k_g.*gap_sim+k_dv.*d_velocity_sim++k_v.*v_sim.^2;
    X_sim=[X_sim,x_sim];
    V_sim=[V_sim,v_sim];
    A_sim=[A_sim,a_sim];
end
V_sim=[zeros(1,T_delay),V_sim];
X_sim=[ones(1,T_delay).*X_sim(1),X_sim];
figure;
plot(x_time,V_1_1(1:end-i),x_time,V_1_2(1:end-i),x_time,V_sim);
figure;
plot(x_time,L_1_1(1:end-i),x_time,L_1_2(1:end-i),x_time,X_sim);
