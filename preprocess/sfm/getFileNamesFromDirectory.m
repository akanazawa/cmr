function [nameStruct] = getFileNamesFromDirectory(dirPath,varargin)
%Returns a cell array of names of all files of a specified format in a
%given directory
%   dir is the directory from which image names are required
%   varargin can be used to specify mode (path/name) and filetypes to be
%   read
%   Example Usage - getFileNamesFromDirectory(dir,'mode','path','types',{'.png', '.jpg'})
%   Default mode is 'name' (just returns filenames). Default 'types' is all
%   image types

%% Initializing Variables
nVarargs = length(varargin);
mode = 'name'; % mode can be 'path' or 'name'
types = {'.jpg','.png', '.bmp', '.tiff', '.jpeg'}; % types is a cell array
nameStruct = {};

%% processing varargin
if(nVarargs > 0)
    for i=1:(nVarargs/2)
        if(strcmp(varargin{2*i-1},'mode'))
            mode = varargin{2*i};
        end
        
        if(strcmp(varargin{2*i-1},'types'))
            types = varargin{2*i};
        end
    end
end

%% Getting the names of the files
for i = 1:length(types)
    t = dir([dirPath,'/*',types{i}]);
    if(size(t,1) > 0)
        nameStruct = [nameStruct extractfield(t,'name')];
    end
end

%% adding path if 'mode' == 'path'
if (strcmp(mode,'path'))
    for i=1:length(nameStruct)
        nameStruct{i} = [dirPath,'/',nameStruct{i}];
    end
end

end

