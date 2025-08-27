function z = remove_bands(data, bs)
% [~, ~, bs] = size(data);

if bs==189
    q1 = linspace(6, 31, (31-6+1));
    q2 = linspace(35, 95, (95-35+1));
    q3 = linspace(97, 105, (105-97+1));
    q4 = linspace(113, 151, (151-113+1));
    q5 = linspace(166, 219, (219-166+1));
    bands = [q1 q2 q3 q4 q5];

elseif bs==175
    q1 = linspace(6, 29, (29-6+1));
    q2 = linspace(35, 92, (92-35+1));
    q3 = linspace(97, 102, (102-97+1));
    q4 = linspace(113, 148, (148-113+1));
    q5 = linspace(166, 216, (216-166+1));
    bands = [q1 q2 q3 q4 q5];

elseif bs==204
    q1 = linspace(3, 31, (31-3+1));
    q2 = linspace(33, 95, (95-33+1));
    q3 = linspace(97, 105, (105-97+1));
    q4 = linspace(109, 151, (151-109+1));
    q5 = linspace(160, 219, (219-160+1));
    bands = [q1 q2 q3 q4 q5];
end

z = data(:, :, bands);


end