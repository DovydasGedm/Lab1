clc;
clear all;

%Naive Bayes classifier

%% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%% Calculate for each image, colour and roundness
%For Apples
%1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
%2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
%3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
%4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
%5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
%6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
%7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
%8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
%9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4];
x2=[metric_A1 metric_A2 metric_A3 metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P1 metric_P2 metric_P3 metric_P4];
T = zeros(1, 13);
p_apple_color = zeros(1, 13);
p_apple_metric = zeros(1, 13);
p_pear_color = zeros(1, 13);
p_pear_metric = zeros(1, 13);
evidence = zeros(1, 13);
posterior_apple = zeros(1, 13);
posterior_pear = zeros(1, 13);

%% Training
Ax1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3];
Ax2 = [metric_A1 metric_A2 metric_A3];
Px1 = [hsv_value_P1 hsv_value_P2 hsv_value_P3];
Px2 = [metric_P1 metric_P2 metric_P3];

A1_mean = mean(Ax1);
A2_mean = mean(Ax2);
A1_variance = var(Ax1);
A2_variance = var(Ax2);

P1_mean = mean(Px1);
P2_mean = mean(Px2);
P1_variance = var(Px1);
P2_variance = var(Px2);

%% Classification
% Determine the probability distribution
P = 0.5;
for i = 1:13
p_apple_color(i) = 1/(sqrt(2*pi*A1_variance))*exp((-(x1(i)-A1_mean)^2)/(2*A1_variance));
p_apple_metric(i) = 1/(sqrt(2*pi*A2_variance))*exp((-(x2(i)-A2_mean)^2)/(2*A2_variance));
p_pear_color(i) = 1/(sqrt(2*pi*P1_variance))*exp((-(x1(i)-P1_mean)^2)/(2*P1_variance));
p_pear_metric(i) = 1/(sqrt(2*pi*P2_variance))*exp((-(x2(i)-P2_mean)^2)/(2*P2_variance));
end

% Determine which posterior is greater, Apple or Pear
for i = 1:13
    evidence(i) = P*p_apple_color(i)*p_apple_metric(i)+P*p_pear_color(i)*p_pear_metric(i);
    posterior_apple(i) = (P*p_apple_color(i)*p_apple_metric(i))/evidence(i);
    posterior_pear(i) = (P*p_pear_color(i)*p_pear_metric(i))/evidence(i);
    if posterior_apple(i) > posterior_pear(i)
	T(i) = 1;
else
	T(i) = -1;
    end
end

display(T);
plot(posterior_apple(1:9), posterior_pear(1:9), 'k*', posterior_apple(10:13), posterior_pear(10:13), 'ro');




