%% Pixels Pattern Recognition

clear;
P1 = [1 -1 -1 -1 -1 1 -1 1 1 1 1 -1 -1 1 1 1 1 -1 -1 1 1 1 1 -1 1 -1 -1 -1 -1 1]';
P2 = [1 1 1 1 1 1 -1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1 1 1 1 1 1 1]';
P3 = [-1 1 1 1 1 1 -1 1 1 -1 -1 -1 -1 1 1 -1 1 -1 1 -1 -1 1 1 -1 1 1 1 1 1 -1]';

T = [P1 P2 P3]; % T=P

%% Reversing 3 random pixels

P1_noisy = P1;
P2_noisy = P2;
P3_noisy = P3;

for pixel = 1:3 % 3 pixels
    r = randi(30); % Length of each vector is 30
    P1_noisy(r) = -1*P1_noisy(r); % At index r in p, reverse sign
    P2_noisy(r) = -1*P2_noisy(r);
    P3_noisy(r) = -1*P3_noisy(r);
end

%% Hebbian Learning 
% Normalizing P1, P2, P3, P1_noisy, P2_noisy, P3_noisy
P1_norm = normc(P1);
P2_norm = normc(P2);
P3_norm = normc(P3);
P1_noisy_norm = normc(P1_noisy);
P2_noisy_norm = normc(P2_noisy);
P3_noisy_norm = normc(P3_noisy);

P = [P1_norm P2_norm P3_norm]; % P = [P1 P2 P3]

% Applying hebbian learning rule W = T*P'
Wh = T*P';

% Testing weight matrix W with noisy patterns W*P_noisy'. Linear
% association a = f(n) = n = W*P(test)
a1h = Wh*P1_noisy_norm;
a2h = Wh*P2_noisy_norm;
a3h = Wh*P3_noisy_norm;

% Reshaping matrix and displaying image to observe recognition performance
figure('Name', 'Output a1 with Hebbian Learning');
imshow(reshape(a1h, 6, 5));
figure('Name', 'Output a2 with Hebbian Learning');
imshow(reshape(a2h, 6, 5));
figure('Name', 'Output a3 with Hebbian Learning');
imshow(reshape(a3h, 6, 5));

% Corelation table in 3x3 matrix

correlation_hebbian = [
      corr2(P1, a1h) corr2(P1, a2h) corr2(P1, a3h); 
      corr2(P2, a1h) corr2(P2, a2h) corr2(P2, a3h);
      corr2(P3, a1h) corr2(P3, a2h) corr2(P3, a3h)
      ]


%% Pesudo Inverse Rule 
Pt = inv(P'*P)*P';
% Applying pseudo inverse rule
Wi = T*Pt;

% Testing noisy patterns W*P
a1i = Wi*P1_noisy;
a2i = Wi*P2_noisy;
a3i = Wi*P3_noisy;

% Reshaping matrix and displaying image to observe recognition performance
figure('Name', 'Output a1 with Pseudo Inverse Rule');
imshow(reshape(a1i, 6, 5));
figure('Name', 'Output a2 with Pseudo Inverse Rule');
imshow(reshape(a2i, 6, 5));
figure('Name', 'Output a3 with Pseudo Inverse Rule');
imshow(reshape(a3i, 6, 5));

% Corelation table in 3x3 matrix

correlation_inverse = [
      corr2(P1, a1i) corr2(P1, a2i) corr2(P1, a3i); 
      corr2(P2, a1i) corr2(P2, a2i) corr2(P2, a3i);
      corr2(P3, a1i) corr2(P3, a2i) corr2(P3, a3i)
      ]
