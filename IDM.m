function [acceleration] = IDM(leader_speed, follower_speed, gap, params)

a = params(1);
b = params(2);
T = params(3);
s0 = params(4);
delta = params(5);
v0 = params(6);

desired_gap = s0 + (follower_speed * T) + (follower_speed * (follower_speed - leader_speed) / (2 * sqrt(a * b)));
acceleration = a * (1 - (follower_speed / v0)^delta - (desired_gap / gap)^2);

end