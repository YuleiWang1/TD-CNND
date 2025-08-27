function y = imitation_sample(data, n)

if n==0
    data = data;
else
    for i=1:n
        data0 = fliplr(data);
        data = [data, data0];

        data0 = flipud(data);
        data = [data;data0];
    end
end

y = data;
end