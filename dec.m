clear;clc;
close all;

addpath('./functions')
addpath('./data')
addpath('./results')

load Beach
load output


[w, h, bs] = size(data);
o = hyperNormalize(o);
imshow(o)

