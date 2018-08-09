function [class_data, part_names] = extract_class_data_p3d(voc_class, p3d_dir, voc_dir, use_pascal, seg_kp_dir, min_vis_kps)
    if nargin < 6
        min_vis_kps = 3;
    end
    opts = get_voc_opts(voc_dir);
    
    if use_pascal
        [val_ids] = textread(sprintf(opts.imgsetpath, ['val']),'%s');
        [train_ids] = textread(sprintf(opts.imgsetpath, ['train']),'%s');
        anno_dir = fullfile(p3d_dir, 'Annotations', [voc_class '_pascal']);

        seg_kp_anno = load(fullfile(seg_kp_dir, voc_class));
        kps_voc_id = cellfun(@(x,y) [x, '_', y], seg_kp_anno.keypoints.voc_image_id, arrayfun(@num2str, seg_kp_anno.keypoints.voc_rec_id, 'Uniform', false), 'UniformOutput', false);
        segs_voc_id = cellfun(@(x,y) [x, '_', y], seg_kp_anno.segmentations.voc_image_id, arrayfun(@num2str, seg_kp_anno.segmentations.voc_rec_id, 'Uniform', false), 'UniformOutput', false);

    else
        anno_dir = fullfile(p3d_dir, 'Annotations', [voc_class '_imagenet']);
    end
    
    ids = getFileNamesFromDirectory(anno_dir, 'types',{'.mat'});
    for i=1:length(ids)
        ids{i} = ids{i}(1:end-4);
    end

    part_names_assigned = 0;

    class_data = [];
    num_pos = 0;

    for i = 1:length(ids)

        rec = load(fullfile(anno_dir, ids{i})); rec = rec.record;
        clsinds = strmatch(voc_class, {rec.objects(:).class}, 'exact');

        % Delete difficult cases
        diff = [rec.objects(clsinds).difficult];
        trunc = [rec.objects(clsinds).truncated];
        occl = [rec.objects(clsinds).occluded];

        clsinds2del = diff | trunc | occl;
        clsinds(clsinds2del)=[];

        if isempty(clsinds)
          continue;
        end

        if ~(use_pascal)
            det_masks_file = fullfile(p3d_dir, 'masks', [voc_class '_imagenet'], ids{i});
            if ~exist([det_masks_file '.mat'], 'file')
                continue;
            end
            det_masks = load(det_masks_file);
        end

        % Create one entry per bounding box in the pos array
        for j = clsinds(:)'
            j_voc_id = [ids{i} '_' num2str(j)];
            if (part_names_assigned == 0)
                part_names_lookup = fieldnames(rec.objects(j).anchors);
                [part_names_lookup] = sort(part_names_lookup);
                if strcmp(voc_class, 'car')
                    % correct inconsistent naming in pascal3d
                    kps_correction_inds = [1 6 7 8 5 2 3 4 9 12 11 10];
                elseif strcmp(voc_class, 'aeroplane')
                    % correct inconsistent naming in pascal3d
                    kps_correction_inds = [4 5 3 1 2 6 7 8];
                else
                    kps_correction_inds = 1:length(part_names_lookup);
                end
                part_names = part_names_lookup(kps_correction_inds);
                part_names_assigned = 1;
                % cad_models = load(fullfile(p3d_dir, 'CAD', voc_class));
                % keyboard;
            end

            [kps, kps_vis] = get_kps(rec.objects(j).anchors, part_names_lookup);
            kps = [kps kps_vis];
            if (sum(kps_vis) <= min_vis_kps)
                continue;
            end
            
            bbox = rec.objects(j).bbox;
            if use_pascal
                si = find(ismember(segs_voc_id,j_voc_id));
                if isempty(si)
                    continue;
                end
                poly_x = seg_kp_anno.segmentations.poly_x{si};
                poly_y = seg_kp_anno.segmentations.poly_y{si};
                mask = roipoly(zeros([rec.size.height rec.size.width]), poly_x, poly_y);
            else
                [det_iou, det_index] = match_detection(bbox, det_masks.boxes);
                if det_iou < 0.8
                    continue
                end
                mask = det_masks.masks(:, :, det_index);
            end

            num_pos = num_pos + 1;
            class_data(num_pos).width = rec.imgsize(1);
            class_data(num_pos).cad_index = rec.objects(j).cad_index;
            class_data(num_pos).height = rec.imgsize(2);
            class_data(num_pos).voc_image_id = ids{i};
            class_data(num_pos).voc_rec_id = j;
            if use_pascal
                class_data(num_pos).is_train = ismember(class_data(num_pos).voc_image_id, train_ids);
                class_data(num_pos).rel_path = fullfile([voc_class '_pascal'], rec.filename);
            else
                class_data(num_pos).rel_path = fullfile([voc_class '_imagenet'], rec.filename);
                class_data(num_pos).is_train = 1;
            end
 
            class_data(num_pos).bbox = struct('x1', bbox(1), 'y1', bbox(2), 'x2', bbox(3), 'y2', bbox(4));
            class_data(num_pos).parts = kps';
            class_data(num_pos).mask = mask;
            % vis_kps(class_data(num_pos), fullfile(p3d_dir, 'Images'), part_names);
        end
    end

end

function [opts] = get_voc_opts(voc_dir)
    tmp = pwd;
    cd(voc_dir);
    addpath([cd '/VOCcode']);
    VOCinit;
    opts = VOCopts;
    cd(tmp);
end

function [kps, vis] = get_kps(anchor_struct, part_names)
    np = length(part_names);
    kps = zeros(np, 2);
    vis = zeros(np, 1);
    for k = 1:length(part_names)
        kp = getfield(getfield(anchor_struct, part_names{k}), 'location');
        if ~isempty(kp)
           kps(k, :) = kp; vis(k) = 1; 
        end
    end
end

function [iou, index] = match_detection(gt_bbox, pred_bboxes)
    if isempty(pred_bboxes)
        iou = 0; index = 0;
    else
        ious = bbox_overlap(gt_bbox, pred_bboxes);
        [iou, index] = max(ious);
    end
end

function ious = bbox_overlap(bbox1,bboxes)
    bboxes(:,3:4) = [bboxes(:,3)-bboxes(:,1) bboxes(:,4)-bboxes(:,2)] + 1;
    bbox1(3:4) = [bbox1(:,3)-bbox1(:,1) bbox1(:,4)-bbox1(:,2)] + 1;
    intersectionArea=rectint(bbox1,bboxes);
    x_g = bbox1(1); y_g = bbox1(2);
    x_p = bboxes(:,1); y_p = bboxes(:,2);
    width_g = bbox1(3); height_g = bbox1(4);
    width_p = bboxes(:,3); height_p = bboxes(:,4);
    unionCoords=[min(x_g,x_p),min(y_g,y_p),max(x_g+width_g-1,x_p+width_p-1),max(y_g+height_g-1,y_p+height_p-1)];
    unionArea = (unionCoords(:,3)-unionCoords(:,1)+1).*(unionCoords(:,4)-unionCoords(:,2)+1);
    ious = intersectionArea(:)./unionArea(:);
end

function [] = vis_kps(record_struct, imgs_dir, part_names)
    img = imread(fullfile(imgs_dir, record_struct.rel_path));
    imagesc(img); hold on;
    parts = record_struct.parts;
    scatter(parts(1,:), parts(2,:), 'r.');
    text(parts(1,:)+1, parts(2,:)+1, part_names);
    keyboard; close all;
end

function [] = vis_3d_kps(cad_struct, part_names, kps_correction_inds)
    kps = zeros(length(part_names), 3);
    for k = 1:length(part_names)
        kps(k, :) = getfield(cad_struct, part_names{k});
    end
    scatter3(cad_struct.vertices(:,1), cad_struct.vertices(:,2), cad_struct.vertices(:,3), 'b.');
    hold on;
    scatter3(kps(:,1), kps(:,2), kps(:,3), 'r.');
    text(kps(:,1), kps(:,2), kps(:,3), part_names(kps_correction_inds));
    axis equal;
    pause(); close all;
end
