function [x_a, x_p, x_n] = dataset_select(x1, x2, x3, x4, x5)

[x12_a, x12_p, x12_n] = rand_select(x1, x2, 300);
[x13_a, x13_p, x13_n] = rand_select(x1, x3, 300);
[x14_a, x14_p, x14_n] = rand_select(x1, x4, 300);
[x15_a, x15_p, x15_n] = rand_select(x1, x5, 300);

[x21_a, x21_p, x21_n] = rand_select(x2, x1, 300);
[x23_a, x23_p, x23_n] = rand_select(x2, x3, 300);
[x24_a, x24_p, x24_n] = rand_select(x2, x4, 300);
[x25_a, x25_p, x25_n] = rand_select(x2, x5, 300);

[x31_a, x31_p, x31_n] = rand_select(x3, x1, 300);
[x32_a, x32_p, x32_n] = rand_select(x3, x2, 300);
[x34_a, x34_p, x34_n] = rand_select(x3, x4, 300);
[x35_a, x35_p, x35_n] = rand_select(x3, x5, 300);

[x41_a, x41_p, x41_n] = rand_select(x4, x1, 300);
[x42_a, x42_p, x42_n] = rand_select(x4, x2, 300);
[x43_a, x43_p, x43_n] = rand_select(x4, x3, 300);
[x45_a, x45_p, x45_n] = rand_select(x4, x5, 300);

[x51_a, x51_p, x51_n] = rand_select(x5, x1, 300);
[x52_a, x52_p, x52_n] = rand_select(x5, x2, 300);
[x53_a, x53_p, x53_n] = rand_select(x5, x3, 300);
[x54_a, x54_p, x54_n] = rand_select(x5, x4, 300);

% x_a(1:300, :) = x12_a(:, :);
% x_a(1+300:300*2, :) = x13_a(:, :);
% x_a(1+300*2:300*3, :) = x14_a(:, :);
% x_a(1+300*3:300*4, :) = x15_a(:, :);
% 
% x_a(1+300*3:300*4, :) = x15_a(:, :);

x_a = [x12_a;x13_a;x14_a;x15_a;x21_a;x23_a;x24_a;x25_a;x31_a;x32_a;x34_a;x35_a;x41_a;x42_a;x43_a;x45_a;x51_a;x52_a;x53_a;x54_a;];
x_p = [x12_p;x13_p;x14_p;x15_p;x21_p;x23_p;x24_p;x25_p;x31_p;x32_p;x34_p;x35_p;x41_p;x42_p;x43_p;x45_p;x51_p;x52_p;x53_p;x54_p;];
x_n = [x12_n;x13_n;x14_n;x15_n;x21_n;x23_n;x24_n;x25_n;x31_n;x32_n;x34_n;x35_n;x41_n;x42_n;x43_n;x45_n;x51_n;x52_n;x53_n;x54_n;];


end