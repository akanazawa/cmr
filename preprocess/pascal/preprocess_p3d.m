voc_dir = '/data1/shubhtuls/cachedir/Datasets/VOCdevkit';
p3d_dir = '/data1/shubhtuls/cachedir/PASCAL3D+_release1.1';

basedir = fullfile(pwd, '..', '..');
seg_kp_dir = fullfile(basedir, 'cachedir', 'pascal', 'segkps');

addpath('../sfm');
img_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'data');
sfm_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'sfm');

mkdirOptional(img_anno_dir);
mkdirOptional(sfm_anno_dir);

categories = {'aeroplane', 'car'};

for c = 1:length(categories)
    category = categories{c};

    if exist(fullfile(sfm_anno_dir, [category '_val.mat']))
        continue
    end
    disp(category);
    [class_data, kp_names] = extract_class_data_p3d(category, p3d_dir, voc_dir, 1, seg_kp_dir);
    [class_data_imgnet, ~] = extract_class_data_p3d(category, p3d_dir, voc_dir, 0, seg_kp_dir);
    class_data = [class_data class_data_imgnet];

    % horz_edges should be front to back
    if strcmp(category, 'car')
        horz_edges = [2 4; 6 8];
    elseif strcmp(category, 'aeroplane')
        horz_edges = [8 3];
    else
        disp('Data not available');
        keyboard;
    end
    [sfm_anno, sfm_verts, sfm_faces, kp_perm_inds] = pascal_sfm(class_data, kp_names, horz_edges, []);

    good_inds = ([sfm_anno.err_sfm_reproj] < 0.01);
    class_data = class_data(good_inds);
    sfm_anno = sfm_anno(good_inds);

    train_ids = [class_data.is_train]; train_ids = (train_ids==1);

    all_img_struct = struct('images', class_data);
    train_img_struct = struct('images', class_data(train_ids));
    val_img_struct = struct('images', class_data(~train_ids));

    all_sfm_struct = struct('sfm_anno', sfm_anno, 'S', sfm_verts, 'conv_tri', sfm_faces);
    train_sfm_struct = struct('sfm_anno', sfm_anno(train_ids), 'S', sfm_verts, 'conv_tri', sfm_faces);
    val_sfm_struct = struct('sfm_anno', sfm_anno(~train_ids), 'S', sfm_verts, 'conv_tri', sfm_faces);
    
    save(fullfile(img_anno_dir, [category '_kps']), 'kp_names', 'kp_perm_inds');

    save(fullfile(img_anno_dir, [category '_all']), '-struct', 'all_img_struct');
    save(fullfile(img_anno_dir, [category '_train']), '-struct', 'train_img_struct');
    save(fullfile(img_anno_dir, [category '_val']), '-struct', 'val_img_struct');

    save(fullfile(sfm_anno_dir, [category '_all']), '-struct', 'all_sfm_struct');
    save(fullfile(sfm_anno_dir, [category '_train']), '-struct', 'train_sfm_struct');
    save(fullfile(sfm_anno_dir, [category '_val']), '-struct', 'val_sfm_struct');

end
