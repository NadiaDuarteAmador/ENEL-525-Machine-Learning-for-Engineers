clear;

% Reading images and converting them into gray scale and double type
queen = double(rgb2gray(imread('queen.jpg')));
michaeljackson = double(rgb2gray(imread('michael-jackson.jpg')));
marie = double(rgb2gray(imread('marie-curie.jpg')));
beyonce = double(rgb2gray(imread('beyonce.jpg')));
einstein = double(rgb2gray(imread('einstein.jpg')));

% Converting matrix into column vector 
P1 = reshape(queen, numel(queen), 1); % numel = number of elements in matrix. reshape(matrix, rows, columns)
P2 = reshape(michaeljackson, numel(michaeljackson), 1);
P3 = reshape(marie, numel(marie), 1);
P4 = reshape(beyonce, numel(beyonce), 1);
P5 = reshape(einstein, numel(einstein), 1);


% Normalizing vectors
P1_norm = normc(P1);
P2_norm = normc(P2);
P3_norm = normc(P3);
P4_norm = normc(P4);
P5_norm = normc(P5);

% Adding 20dB white Gaussian noise to each image and normalizing. Noisy images.

P1_noisy = normc(awgn(P1_norm, 20, 'measured'));
P2_noisy = normc(awgn(P2_norm, 20, 'measured'));
P3_noisy = normc(awgn(P3_norm, 20, 'measured'));
P4_noisy = normc(awgn(P4_norm, 20, 'measured'));
P5_noisy = normc(awgn(P5_norm, 20, 'measured'));

%% Hebbian Learning Rule

P = [P1_norm P2_norm P3_norm P4_norm P5_norm];

T = [P1 P2 P3 P4 P5]; % T = P

% Applying hebbian learning rule W = T*P'
Wh = T*P';
% Testing with noisy patterns W*P(test)'
a1h = Wh*P1_noisy;
a2h = Wh*P2_noisy;
a3h = Wh*P3_noisy;
a4h = Wh*P4_noisy;
a5h = Wh*P5_noisy;

% Reshaping matrix and displaying image to observe recognition performance
figure('Name', 'Output a1 with Hebbian Learning');
imshow(mat2gray(reshape(a1h, 64, 64)));
figure('Name', 'Output a2 with Hebbian Learning');
imshow(mat2gray(reshape(a2h, 64, 64)));
figure('Name', 'Output a3 with Hebbian Learning');
imshow(mat2gray(reshape(a3h, 64, 64)));
figure('Name', 'Output a4 with Hebbian Learning');
imshow(mat2gray(reshape(a4h, 64, 64)));
figure('Name', 'Output a5 with Hebbian Learning');
imshow(mat2gray(reshape(a5h, 64, 64)));
% Corelation table in 3x3 matrix

correlation = [
      corr2(P1, a1h) corr2(P1, a2h) corr2(P1, a3h) corr2(P1, a4h) corr2(P1, a5h); 
      corr2(P2, a1h) corr2(P2, a2h) corr2(P2, a3h) corr2(P2, a4h) corr2(P2, a5h);
      corr2(P3, a1h) corr2(P3, a2h) corr2(P3, a3h) corr2(P3, a4h) corr2(P3, a5h);
      corr2(P4, a1h) corr2(P4, a2h) corr2(P4, a3h) corr2(P4, a4h) corr2(P4, a5h);
      corr2(P5, a1h) corr2(P5, a2h) corr2(P5, a3h) corr2(P5, a4h) corr2(P5, a5h)
      ]

%% Pesudo Inverse Rule 

Pt = inv(P'*P)*P';
% Applying pseudo inverse rule
Wi = T*Pt;

% Testing noisy patterns W*P
a1i = Wi*P1_noisy;
a2i = Wi*P2_noisy;
a3i = Wi*P3_noisy;
a4i = Wi*P4_noisy;
a5i = Wi*P5_noisy;

% Reshaping matrix and displaying image to observe recognition performance
figure('Name', 'Output a1 with Pseudo Inverse Rule');
imshow(mat2gray(reshape(a1i, 64, 64)));
figure('Name', 'Output a2 with Pseudo Inverse Rule');
imshow(mat2gray(reshape(a2i, 64, 64)));
figure('Name', 'Output a3 with Pseudo Inverse Rule');
imshow(mat2gray(reshape(a3i, 64, 64)));
figure('Name', 'Output a4 with Pseudo Inverse Rule');
imshow(mat2gray(reshape(a4i, 64, 64)));
figure('Name', 'Output a5 with Pseudo Inverse Rule');
imshow(mat2gray(reshape(a5i, 64, 64)));

% Corelation table in 3x3 matrix
correlation_inverse = [
      corr2(P1, a1i) corr2(P1, a2i) corr2(P1, a3i) corr2(P1, a4i) corr2(P1, a5i); 
      corr2(P2, a1i) corr2(P2, a2i) corr2(P2, a3i) corr2(P2, a4i) corr2(P2, a5i);
      corr2(P3, a1i) corr2(P3, a2i) corr2(P3, a3i) corr2(P3, a4i) corr2(P3, a5i);
      corr2(P4, a1i) corr2(P4, a2i) corr2(P4, a3i) corr2(P4, a4i) corr2(P4, a5i);
      corr2(P5, a1i) corr2(P5, a2i) corr2(P5, a3i) corr2(P5, a4i) corr2(P5, a5i)
      ]
