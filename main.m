% 导入数据
load('ym2.mat')
figure(1); imagesc(im0); colormap gray; hold on; axis equal; axis on;
plot(xim0(1,:), xim0(2,:),'b.');
figure(2); imagesc(im1); colormap gray; hold on; axis equal; axis on;
plot(xim1(1,:), xim1(2,:),'b.');

% 检查数据维度
if size(xim0,2) ~= size(xim1,2) || size(xim0,1) ~= size(xim1,1)
    error('inputs data dimensions do not match!');
end

% 求取坐标点数据的均值
x1 = mean(xim0(1,:),2);
y1 = mean(xim0(2,:),2);
x2 = mean(xim1(1,:),2);
y2 = mean(xim1(2,:),2);

% RMS -> Mat T_scale
M1 = 0;
items1 = size(xim0,2);
for i=1:items1
M1 = M1 + (xim0(1,i) - x1).^2 + (xim0(2,i) - y1).^2;
end
RMS = sqrt(M1 / items1);
T_scale1=[sqrt(2)/RMS 0 0;0 sqrt(2)/RMS 0;0 0 1];
T_trans1 = [1 0 -x1;0 1 -y1;0 0 1];
T_norm1 = T_scale1 * T_trans1;

M2 = 0;
items2 = size(xim1,2);
for i=1:items2
M2 = M2 + (xim1(1,i) - x2).^2 + (xim1(2,i) - y2).^2;
end
RMS=sqrt(M2 / items2);
T_scale2=[sqrt(2)/RMS 0 0;0 sqrt(2)/RMS 0;0 0 1];
T_trans2 = [1 0 -x2;0 1 -y2;0 0 1];
T_norm2 = T_scale2 * T_trans2;

% X_norm = T_norm * X
X_norm1 = T_norm1 * xim0;
X_norm2 = T_norm2 * xim1;

% ((X_norm2)T * F_norm * X_norm1) = 0
%               [[f11 f12 f13],  [[u1],
% [[u2,v2,1]] * [f21 f22 f23], *  [v1], = 0
%               [f31 f32 f33]]    [1]]
h1=[];h2=[];h4=[];h5=[];
for i=1:13
    h1 = [h1, X_norm2(1,i) * X_norm1(1,i)]; % u2 * u1
    h2 = [h2, X_norm2(1,i) * X_norm1(2,i)]; % u2 * v1
    h4 = [h4, X_norm2(2,i) * X_norm1(1,i)]; % v2 * u1
    h5 = [h5, X_norm2(2,i) * X_norm1(2,i)]; % v2 * v1
end
h3 = X_norm2(1,:); % u2
h6 = X_norm2(2,:); % v2
h7 = X_norm1(1,:); % u1
h8 = X_norm1(2,:); % v1
h9 = ones(1,13); % 1

H1 = [h1;h2;h3;h4;h5;h6;h7;h8;h9]';
[U,S,V] = svd(H1);
f_norm = V(:,end);
F_norm = reshape(f_norm,[3,3]);
F_norm = F_norm';

%             [[D1 0 0],
% F_norm = U * [0 D2 0], * VT
%              [0 0 0]]
[U1,S1,V1] = svd(F_norm);
S1(3,3) = 0;
F_norm_ = U1 * S1 * V1'; % rank(F_norm_)=2
% F = (T_norm2)T * F_norm * T_norm1
F = T_norm2' * F_norm_ * T_norm1
% check fundamental matrix F
img = vgg_gui_F(im1,im0,F);

% 求解13个基本点的三维结构
% fundamental matrix E = KT*F*K
E = Khat' * F * Khat
[U_1,S_1,V_1] = svd(E); W = [0 -1 0;1 0 0;0 0 1];
R1 = U_1 * W * V_1';
R2 = U_1 * W' * V_1';
t1 = U_1(:,end);
t2 = -U_1(:,end);
% P'= [R|t]
P_cam1 = [R1,t1];
P_cam2 = [R1,t2];
P_cam3 = [R2,t1];
P_cam4 = [R2,t2];

I = [1 0 0;0 1 0;0 0 1]; O = [0;0;0];
P_cam = [I,O]; % P = [I|O]
Xi1 = inv(Khat) * xim0; % inv(Khat) * xim0
Xi2 = inv(Khat) * xim1; % inv(Khat) * xim1

% P_cam1
a1 = P_cam(3,:) * Xi1(1,1) - P_cam(1,:);
a2 = P_cam(3,:) * Xi1(2,1) - P_cam(2,:);
a3 = P_cam1(3,:) * Xi2(1,1) - P_cam1(1,:);
a4 = P_cam1(3,:) * Xi2(2,1) - P_cam1(2,:);
a1 = a1 / sqrt(a1 * a1');
a2 = a2 / sqrt(a2 * a2');
a3 = a3 / sqrt(a3 * a3');
a4 = a4 / sqrt(a4 * a4');
A = [a1;a2;a3;a4];
[~,~,V_] = svd(A);
XX1 = V_(:,end);
w = P_cam * XX1;
w1 = P_cam1 * XX1;
depth_Pcam = sign(det(I)) * w(3) / (XX1(4) * sqrt(I(3,:) * I(3,:)'));
depth_Pcam1 = sign(det(R1)) * w1(3) / (XX1(4) * sqrt(R1(3,:) * R1(3,:)'));

% P_cam2
a1 = P_cam(3,:) * Xi1(1,1) - P_cam(1,:);
a2 = P_cam(3,:) * Xi1(2,1) - P_cam(2,:);
a3 = P_cam2(3,:) * Xi2(1,1) - P_cam2(1,:);
a4 = P_cam2(3,:) * Xi2(2,1) - P_cam2(2,:);
a1 = a1 / sqrt(a1 * a1');
a2 = a2 / sqrt(a2 * a2');
a3 = a3 / sqrt(a3 * a3');
a4 = a4 / sqrt(a4 * a4');
A = [a1;a2;a3;a4];
[~,~,V_] = svd(A);
XX2 = V_(:,end);
w = P_cam * XX2;
w1 = P_cam2 * XX2;
depth_Pcam = sign(det(I)) * w(3) / (XX2(4) * sqrt(I(3,:) * I(3,:)'));
depth_Pcam2 = sign(det(R1)) * w1(3) / (XX2(4) * sqrt(R1(3,:) * R1(3,:)'));

% P_cam3
a1 = P_cam(3,:) * Xi1(1,1) - P_cam(1,:);
a2 = P_cam(3,:) * Xi1(2,1) - P_cam(2,:);
a3 = P_cam3(3,:) * Xi2(1,1) - P_cam3(1,:);
a4 = P_cam3(3,:) * Xi2(2,1) - P_cam3(2,:);
a1 = a1 / sqrt(a1 * a1');
a2 = a2 / sqrt(a2 * a2');
a3 = a3 / sqrt(a3 * a3');
a4 = a4 / sqrt(a4 * a4');
A = [a1;a2;a3;a4];
[~,~,V_]=svd(A);
XX3 = V_(:,end);
w = P_cam * XX3;
w1 = P_cam3 * XX3;
depth_Pcam = sign(det(I)) * w(3) / (XX3(4) * sqrt(I(3,:) * I(3,:)'));
depth_Pcam3 = sign(det(R2)) * w1(3) / (XX3(4) * sqrt(R2(3,:) * R2(3,:)'));

% P_cam4
a1 = P_cam(3,:) * Xi1(1,1) - P_cam(1,:);
a2 = P_cam(3,:) * Xi1(2,1) - P_cam(2,:);
a3 = P_cam4(3,:) * Xi2(1,1) - P_cam4(1,:);
a4 = P_cam4(3,:) * Xi2(2,1) - P_cam4(2,:);
a1 = a1/sqrt(a1*a1');
a2 = a2/sqrt(a2*a2');
a3 = a3/sqrt(a3*a3');
a4 = a4/sqrt(a4*a4');
A = [a1;a2;a3;a4];
[~,~,V_] = svd(A);
XX4 = V_(:,end);
w = P_cam * XX4;
w1 = P_cam4 * XX4;
depth_Pcam = sign(det(I)) * w(3) / (XX4(4) * sqrt(I(3,:) * I(3,:)'));
depth_Pcam4 = sign(det(R2)) * w1(3) / (XX4(4) * sqrt(R2(3,:) * R2(3,:)'));
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
XX=[];TT=[];
for i=1:13
a1 = P_cam(3,:) * Xi1(1,i) - P_cam(1,:);
a2 = P_cam(3,:) * Xi1(2,i) - P_cam(2,:);
a3 = P_cam2(3,:) * Xi2(1,i) - P_cam2(1,:);
a4 = P_cam2(3,:) * Xi2(2,i) - P_cam2(2,:);
a1 = a1 / sqrt(a1*a1');
a2 = a2 / sqrt(a2*a2');
a3 = a3 / sqrt(a3*a3');
a4 = a4 / sqrt(a4*a4');
A = [a1;a2;a3;a4];
[~,~,V3] = svd(A);
XX = [XX,V3(:,end)];
TT = [TT,XX(1:3,i) / XX(4,i)];
end
TT_scene = TT(1:3,:)

x = TT_scene(1,:); y = TT_scene(2,:); z = TT_scene(3,:);
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y))); %插值
figure,mesh(X,Y,Z);%三维曲面
