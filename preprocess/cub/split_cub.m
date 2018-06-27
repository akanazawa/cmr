function split_cub()
% Splits the test set of cub into 
% val / test.

cub_cache_dir = fullfile(pwd, '..', '..', 'cachedir', 'cub');

orig_path = fullfile(cub_cache_dir, 'data', 'testval_cub_cleaned.mat');
orig_sfm_path = fullfile(cub_cache_dir, 'sfm', 'anno_testval.mat');

% New mat.
val_path = fullfile(cub_cache_dir, 'data', 'val_cub_cleaned.mat');
test_path = fullfile(cub_cache_dir, 'data', 'test_cub_cleaned.mat');
val_sfm_path = fullfile(cub_cache_dir, 'sfm', 'anno_val.mat');
test_sfm_path = fullfile(cub_cache_dir, 'sfm', 'anno_test.mat');

% Load all data. % This is already cleaned
load(orig_path, 'images');
load(orig_sfm_path, 'sfm_anno', 'S');

num_images = length(images);

half = round(num_images / 2);

rng(100);
inds = randperm(num_images);
test_inds = sort(inds(1:half));
val_inds = sort(inds(half:end));

test_images = images(test_inds);
val_images = images(val_inds);

test_sfm_anno = sfm_anno(test_inds);
val_sfm_anno = sfm_anno(val_inds);

save_mat(test_path, test_images);
save_mat(val_path, val_images);

save_sfm_mat(test_sfm_path, test_sfm_anno);
save_sfm_mat(val_sfm_path, val_sfm_anno);

function save_mat(mat_path, images)
save(mat_path, 'images');

function save_sfm_mat(mat_path, sfm_anno)
save(mat_path, 'sfm_anno');
