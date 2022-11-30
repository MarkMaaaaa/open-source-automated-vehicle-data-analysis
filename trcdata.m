clear;
clc;
str = ['D:\Project\AMS\Data\Tesla_CAV_2021-09-15-16-50-10_520_120_processed.csv'];
data = xlsread(str);
% 1.ID 2.Time 3.distance_headway 24 closet dis
% AV:  6.pos_x 7.pos_y 8.heading_meter 12.v 13.a
% SV: 16.pos_x 17.pos_y 18.heading_meter 22.v 23.a
% num = 23;
% max(data(:,num))
% min(data(:,num))
% mean(data(:,num))
dis1 = data(:,3); % distance_av (headway)
dis2 = data(:,8); % heading_av_m
dis3 = data(:,18); % heading_sv
dis4 = data(:,24); % closest_distance_longitudinal (gap)

% plot(1:length(dis),data(:,12),1:length(dis),data(:,22))
%%
ID = data(:,1);
test_num = ones(1,max(ID));
for i=2:max(ID)
    [~, B] = max(data(:,1)==i, [], 1);
    test_num(i) = B;
end
num = 2;

test=[test_num(num):test_num(num+1)];
% test=[1:test_num(1)];
av_pos_x = data(test,6);
av_pos_y = data(test,7);
av_heading = data(test,8);
av_v = smooth(data(test,12)); %smooth
av_a = data(test,23);

sv_pos_x = data(test,16);
sv_pos_y = data(test,17);
sv_heading = data(test,18);
sv_v = data(test,22);
sv_a = data(test,23);
plot(1:length(test),av_v,1:length(test),sv_v); legend
legend 
loss_num = 20;
V=[sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v;sv_v];
bar(sort(V),5000)

bar(sort(sv_v))

v_diff = sv_v(loss_num:end-loss_num)-av_v(loss_num:end-loss_num);
a_diff = diff(v_diff);
a_final = av_a(loss_num:end-loss_num);
x_time = 1:(length(test)-2*loss_num+1);
% X = [ones(length(a_final),1),mapminmax(sv_d_final',-1,1)'];
sv_d_final = av_heading(loss_num:end-loss_num);
v_diff_final = v_diff;
sv_v_final = av_v(loss_num:end-loss_num);
[SV_D,PS_D] = mapminmax(sv_d_final',-1,1);
[V_Diff,PS_DV] = mapminmax(v_diff_final',-1,1);
[SV_V,PS_V] = mapminmax(sv_v_final',-1,1);

X =[ones(length(a_final),1),sv_d_final,v_diff_final,sv_v_final];
[b,bint,r,rint,stats] = regress(a_final,X);
stats


X_mapmi = [ones(length(a_final),1),SV_D',V_Diff',SV_V'];
[B,BINT,R,RINT,STATS] = regress(mapminmax(a_final,-1,1),X_mapmi);

% a = -07569 + 0.0009*dis + 0.0614*v_d + 0.0452*v 
% norm a = 0.0188 + 0.0142*dis + 0.0870*v_d + 0.0072*v
% a_prid = 0.188 - 0.0142*sv_d_final + 0.0870*v_diff_final + 0.0072*sv_v_final;
X_d = [SV_D',R];
X_vd = [V_Diff',R];
X_v = [SV_V',R];

XX = [X_mapmi,R];

a1=sortrows(XX,2);
a2=sortrows(XX,3);
a3=sortrows(XX,4);

hold on
plot(a1(:,2),a1(:,5),a1(:,2),zeros(length(a1)));
plot(a2(:,3),a2(:,5),a1(:,3),zeros(length(a1)));
plot(a3(:,4),a3(:,5),a1(:,4),zeros(length(a1)));

for i=1:length(a1)
    a1_diff_r = diff(a1(:,5)>0);
%     a1_neg = find(b==1)+1;
%     a1_pos = find(b==-1)+1;
%     a1_gap = find(b~=0)+1;
%     a1_gap = a1_pos - a1_neg;
    a1_gap = diff(find(a1_diff_r~=0)+1);
    [m_a1,p_a1] = max(a1_gap);
    X_d_11_end = sum(a1_gap(1:p_a1))+1;
    X_d_11_start = sum(a1_gap(1:(p_a1-1)))+2;
    X_d_11 = mean(a1(X_d_11_start:X_d_11_end,2));
    Y_d_11 = mean(a1(X_d_11_start:X_d_11_end,5));

    a2_v = diff(a2(:,5)>0);
    a2_gap = diff(find(a2_v~=0)+1);
    [m_a2,p_a2] = max(a2_gap);
    d_v_11_end = sum(a2_gap(1:p_a2))+1;
    d_v_11_start = sum(a2_gap(1:(p_a2-1)))+2;
    X_dv_11 = mean(a2(d_v_11_start:d_v_11_end,3));
    Y_dv_11 = mean(a2(d_v_11_start:d_v_11_end,5));
    
    a3_v = diff(a3(:,5)>0);
    a3_gap = diff(find(a3_v~=0)+1);
    [m_a3,p_a3] = max(a3_gap);
    v_11_end = sum(a3_gap(1:p_a3))+1;
    v_11_start = sum(a3_gap(1:(p_a3-1)))+2;
    X_v_11 = mean(a2(v_11_start:v_11_end,4));
    Y_v_11 = mean(a2(v_11_start:v_11_end,5));
%     SSE_11_1 = sx_1*x + sy*y + sz*z + con_11; 
%     SSE_11_2 = sx_1*x + sy*y + sz*z + con_11;

end
x_test_1 = find(X_mapmi(:,2)<X_d_11);
[b,bint,r,rint,stats_test1] = regress(a_final(x_test_1),X_mapmi(x_test_1,:));
stats_test1


x_test_3 = find(X_mapmi(:,4)<X_v_11);
[b,bint,r,rint,stats_test3] = regress(a_final(x_test_3),X_mapmi(x_test_3,:));
stats_test3

mapminmax('reverse',X_v_11,PS_V)





mapminmax('reverse',X_d_11,PS_D)


% [model, time, resultsEval] = aresbuild(X,a_diff);





% dis = sqrt((sv_pos_x-av_pos_x).^2+(sv_pos_y-av_pos_y).^2);
% plot3(x_time,av_pos_x,av_pos_y,x_time,sv_pos_x,sv_pos_y);
subplot(1,3,1);
plot(av_pos_x,av_pos_y,sv_pos_x,sv_pos_y);
legend('av','sv');
xlabel('pos-x(m)');ylabel('pos-y(m)');

subplot(1,3,2);
plot(x_time,av_v(x_time),x_time,sv_v(x_time));
xlabel('Time');ylabel('velocity(m/s)');
legend('av','sv');

a_diff(find(abs(a_diff)>0.8))=0;

subplot(1,3,3);
% plot(x_time,dis)
y1=smooth(abs(a_diff(2:end)*10),15);
y2=smooth(abs(av_a(3:end)),15);
plot(x_time(3:end),y1(1:length(x_time(3:end))),x_time(3:end),y2(1:length(x_time(3:end))))
xlabel('Time');ylabel('acceleration(m/s^2)');
legend('velocity-difference','collected acceleration');


%% kmeans

% dis av_v av_a v_diff
X_varible = [av_a;dis;av_v;v_diff];
error_K = [];
for k=1
    [Idx,C,sumD,D] = kmeans(X_varible,K);
%     test_k = zeros(1,k)
%     for ii=1:k    
%         test_k(ii) = length(find(Idx==ii));
%     end
    error_k = sum(sumD(1)/length(x_time));
    error_K = [error_K,error_k];

end


plot(2:k,error_K)



