function [] = mkdirOptional(dirName)
%MKDIROPTIONAL Summary of this function goes here
%   Detailed explanation goes here

if(~exist(dirName,'dir'))
    mkdir(dirName)
end

end