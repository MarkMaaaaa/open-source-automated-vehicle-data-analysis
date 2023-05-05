clear;
clc;

m2ft = 3.28084;
ft2m = 1/m2ft;


str = ['D:\Project\data\ASta_050719_platoon2.csv'];
data = xlsread(str);
% 1.time 2.speed1 3.lat1 4.lon1 
% 9.speed2 10.lat2 11.lon2
for i0=0:3
TIme = data(7:end,1);
v_pre = data(7:end,2+7*i0);
v_fol = data(7:end,9+7*i0);
% lat_pre = data(7:end,3);lon_pre = data(7:end,4);
% lat_fol = data(7:end,10);lon_fol = data(7:end,11);
% mstruct=defaultm('mercator');%定义椭球体长轴，椭率，坐标原点
% mstruct.geoid=[6378137,0.0818191908426215];
% mstruct.origin=[0,0,0];
% mstruct=defaultm(mstruct);
% [x_pre,y_pre] =projfwd(mstruct,lat_pre,lon_pre);
% [x_fol,y_fol] =projfwd(mstruct,lat_fol,lon_fol);
x_pre = data(7:end,6+7*i0);
y_pre = data(7:end,7+7*i0);
x_fol = data(7:end,13+7*i0);
y_fol = data(7:end,14+7*i0);
% [x_fol,y_fol] =projfwd(mstruct,lat_fol,lon_fol);
% dis = sqrt((x_pre-x_fol).^2+(y_pre-y_fol).^2);
d_fol= sqrt(x_fol.^2+y_fol.^2);
d_pre= sqrt(x_pre.^2+y_pre.^2);
dis = d_pre - d_fol;
gap = data(7:end,42+i0);


v_diff = v_pre(1:end-1)-v_fol(1:end-1);
a_fol = diff(v_fol)*10;

num=50;
Td=1.4;d_safe=1;
num_P=4;



X=[ones(22634,1),gap(1:end-1), v_fol(1:end-1), v_diff(1:end)];
[b1,bint1,r1,rint1,stats1] = regress(a_fol(1:end),X);

delay = 13;

X=[ones(22634-delay,1),gap(1:end-1-delay), v_fol(1:end-1-delay), v_diff(1:end-delay)];
[b1,bint1,r1,rint1,stats1] = regress(a_fol(1+delay:end),X);


% d_new_a = 0;
v_new_a = v_fol(1);
D_new_a = [];
V_new_a = [];
A_new_a = [];
gap_new_a = gap(1);


for time_i = 1:22634
    a_new_a = b1(1) + b1(2)*(gap_new_a) + b1(3)*v_new_a + b1(4)*(v_pre(time_i) - v_new_a);
    gap_new_a = gap_new_a + (v_fol(time_i) - (v_new_a + 0.05*a_new_a)) * 0.1;
    v_new_a = v_new_a + a_new_a*0.1;
%     d_new_a = d_new_a + v_new_a*0.1 + 0.5*a_new_a*0.01; 
    D_new_a = [D_new_a;gap_new_a];
    V_new_a = [V_new_a;v_new_a];
    A_new_a = [A_new_a;a_new_a];
    
end
plot(0.1:0.1:2263.4,V_new_a*m2ft);
plot(0.1:0.1:2263.4,v_fol(1:end-1)*m2ft,0.1:0.1:2263.4,V_new_a*m2ft);
legend('Real-SV','Simu-SV','Simu2-SV','Simu3-SV');
xlabel('Time(sec)');
ylabel('Velocity(ft/sec)');

path_1=['C:\Users\Ke\Box\Project\AMS\Figure\models-sv-velocity.fig'];
path_2=['C:\Users\Ke\Box\Project\AMS\Figure\models-sv-velocity.pdf'];
saveas(gcf,path_1);
saveas(gcf,path_2);
rsquare(v_fol(1:end-1),V_new_a);
rsquare(a_fol,A_new_a);

% plot(1:22634,D_new_a,1:22634,gap(1:end-1));
% legend('simu','real');
% xlabel('time(0.1 sec)');
% ylabel('Gap(m)');


D_new_I = [];
V_new_I = [];
A_new_I = [];
params = [1.5226023586787998, 3.0, 0.8723620611448215, 4.004726843181485, 2.172021336082908, 47.66047393864299];
follower_speed = v_fol(1);
gap_new_I = gap(1);
for time_i = 1:22634
    lead_speed = v_pre(time_i);
%     lead_pos = num(i_IDM,2)*0.3048;
%     gap = lead_pos - follower_pos;
    acceleration = IDM(lead_speed, follower_speed, gap_new_I, params);
    gap_new_I = gap_new_I + (v_pre(time_i) - (follower_speed + 0.5*acceleration))*0.1;
    follower_speed = follower_speed + acceleration*0.1;
    
    D_new_I = [D_new_I;gap_new_I];
    V_new_I = [V_new_I;follower_speed];
    A_new_I = [A_new_I;acceleration];
end
rsquare(v_fol(1:end-1),V_new_I);
rsquare(a_fol,A_new_I);

plot(0.1:0.1:2263.4,V_new_I*m2ft);
legend('Simu-SV','Real-SV');
xlabel('Time(sec)');
ylabel('Velocity(ft/sec)');

path_1=['C:\Users\Ke\Box\Project\AMS\Figure\IDM-model-sv-velocity.fig'];
path_2=['C:\Users\Ke\Box\Project\AMS\Figure\IDM-model-sv-velocity.pdf'];
saveas(gcf,path_1);
saveas(gcf,path_2);

% gap(2)-gap(1)
% gap(3)-gap(2)
% gap(4)-gap(3)
% 
% v_pre(1)-v_fol(1)
% v_pre(2)-v_fol(2)
% v_pre(3)-v_fol(3)











b=cell(num_P,1,num);bint=cell(num_P,2,num);r=cell(1482,1,num);rint=cell(1482,2,num);stats=cell(1,4,num);
for i=1:num %delay步长
%     X=[ones(22635-i,1),dis(1:end-i), v_fol(1:end-i)];
    X=[ones(22635-i,1),gap(1:end-i), v_fol(1:end-i), v_diff(1:end+1-i)];
    [b1,bint1,r1,rint1,stats1] = regress(a_fol(i:end),X);
    b(:,:,i)=num2cell(b1);
    bint(:,:,i)=num2cell(bint1);
    % r(:,:,i)=num2cell(r1);
    % rint(:,:,i)=num2cell(rint1);
    stats(:,:,i)=num2cell(stats1);
end
stats_a=[];

for i4= 1:num
    a = cell2mat(stats(:,1,i4));
    stats_a = [stats_a,a];
end
[max_val, max_idx] = max(stats_a);
x=0.1:0.1:num/10;

para_1_delay = cell2mat(b(1,:,max_idx));
para_2_delay = cell2mat(b(2,:,max_idx));
para_3_delay = cell2mat(b(3,:,max_idx));
para_4_delay = cell2mat(b(4,:,max_idx));

d_new_a = 0;
v_new_a = v_fol(1);
D_new_a = [];
V_new_a = [];
A_new_a = [];
gap_new_a = gap(1);

for time_i = 1:22634
    a_new_a = para_1_delay + para_2_delay*(gap_new_a) + para_3_delay*v_new_a + para_4_delay*(v_pre(time_i) - v_new_a);
    gap_new_a = gap_new_a + (v_fol(time_i) - (v_new_a + 0.05*a_new_a)) * 0.1;
    v_new_a = v_new_a + a_new_a*0.1;
%     d_new_a = d_new_a + v_new_a*0.1 + 0.5*a_new_a*0.01; 
    D_new_a = [D_new_a;gap_new_a];
    V_new_a = [V_new_a;v_new_a];
    A_new_a = [A_new_a;a_new_a];
    
end

plot(1:22634,V_new_a,1:22634,v_fol(1:end-1));
legend('simu','real');
xlabel('time(0.1 sec)');
ylabel('velocity(m/s)');

plot(1:22634,D_new_a,1:22634,gap(1:end-1));
legend('simu','real');
xlabel('time(0.1 sec)');
ylabel('Gap(m)');




% plot(x,stats_a);
% hold on
% plot(x(max_idx), max_val, 'r*', 'MarkerSize', 10);
% text(x(max_idx), max_val, sprintf('(%.2f, %.2f)', x(max_idx), max_val), ...
%     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
% xlabel('Time Delay(sec)')
% ylabel('R square')
% title('ICE AV')
% path_1=['C:\Users\Ke\Box\My paper\xp-Empirical study of feedback delay and mobility analyses for production automated vehicles\fig\delay-ICE-AV.fig'];
% path_2=['C:\Users\Ke\Box\My paper\xp-Empirical study of feedback delay and mobility analyses for production automated vehicles\fig\delay-ICE-AV.pdf'];
% saveas(gcf,path_1);
% saveas(gcf,path_2);

k = cell2mat(b(2,:,max_idx));
delta = -(cell2mat(b(1,:,max_idx))/k+3);
tau = -cell2mat(b(3,:,max_idx))/k;
TF = sqrt(4/(4*k*tau^2-k^2*tau^4));

density = 1./dis*1000;
headway = 3600./(dis./v_fol);
X_D =[];
Y_H =[];
for i= 43:1:160
    pos = find(density>i&density<(i+1));
    x_d = mean(density(pos));
    y_h = mean(headway(pos));
    X_D = [X_D,x_d];
    Y_H = [Y_H,y_h];
end
if i0==1
% plot(X_D,Y_H,'.');hold on
xx = [ones(size(X_D')),X_D'];
final_b = regress(Y_H',xx);
x_final = 43:0.1:160;
y_final = final_b(1)+final_b(2)*x_final;
plot(x_final,y_final,'LineWidth',2);
end
hold on
if i0==2
% plot(X_D,Y_H,'*');hold on
xx = [ones(size(X_D')),X_D'];
final_b = regress(Y_H',xx);
x_final = 43:0.1:160;
y_final = final_b(1)+final_b(2)*x_final;
plot(x_final,y_final,'--','LineWidth',2);

end
hold on
if i0==3
% plot(X_D,Y_H,'+');hold on
xx = [ones(size(X_D')),X_D'];
final_b = regress(Y_H',xx);
x_final = 43:0.1:160;
y_final = final_b(1)+final_b(2)*x_final;
plot(x_final,y_final,'-.','LineWidth',2);
end
ylim([0 inf])
end

legend('Hyvrid AV','Eelectric AV','ICE AV')
xlabel('Density(veh/km)')
ylabel('Flow(veh/h)')
path_1=['C:\Users\Ke\Box\My paper\xp-Empirical study of feedback delay and mobility analyses for production automated vehicles\fig\com2.fig'];
path_2=['C:\Users\Ke\Box\My paper\xp-Empirical study of feedback delay and mobility analyses for production automated vehicles\fig\com2.pdf'];
saveas(gcf,path_1);
saveas(gcf,path_2);

% plot(v_pre,v_fol);
plot(x_pre,y_pre,x_fol,y_fol);




