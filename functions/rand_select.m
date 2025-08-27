function [z_a, z_p, z_n] = rand_select(x1, x2, n)

for i=1:n
    term = x1(randperm(size(x1, 1),2), :);
    z_a(i, :) = term(1, :);
    z_p(i, :) = term(2, :);
end
z_n = x2(randperm(size(x2, 1),n), :);

end