function [out num_feat_ch] = extract_features_only(img, ...
    cos_win, feature_type, w2c, cell_size, use_mex)

im = img;

% hog features
nHogChan = 31;

% compute num. feature channels
num_feat_ch = 0;
feat_gray = false; feat_hog = false; feat_cn = false; feat_color = false; feat_hsv = false;
if sum(strcmp(feature_type, 'hog'))
    num_feat_ch = num_feat_ch + nHogChan;
    feat_hog = true;
end
if sum(strcmp(feature_type, 'gray'))
    num_feat_ch = num_feat_ch + 1;
    feat_gray = true;
end
if sum(strcmp(feature_type, 'cn'))
    num_feat_ch = num_feat_ch + size(w2c,2);
    feat_cn = true;
end

if feat_hog
    out_size = floor([size(im, 1) size(im, 2)] ./ cell_size);
else
    out_size = [size(im, 1) size(im, 2)];
end

out = zeros(out_size(1), out_size(2), num_feat_ch);
channel_id = 1;

% extract features from image
if feat_hog
    % extract HoG features
    nOrients = 9;
	hog_image = fhog(single(im), cell_size, nOrients);
    % put HoG features into output structure
    out(:,:,channel_id:(channel_id + nHogChan - 1)) = hog_image(:,:,1:nHogChan);
    channel_id = channel_id + nHogChan;
end

if feat_gray
    % prepare grayscale patch
	if size(im,3) > 1
		gray_patch = rgb2gray(im);
	else
		gray_patch = im;
    end
    % resize it to out size
	gray_patch = imresize(gray_patch, out_size);
    % put grayscale channel into output structure
    out(:, :, channel_id) = single((single(gray_patch) / 255) - 0.5);
    channel_id = channel_id + 1;
end

if feat_cn
    % extract ColorNames features
    CN = im2c(single(im), w2c, -2);
    if use_mex
        if cell_size > 1
           CN = average_feature_region(CN, cell_size);
        end
    else
        CN = imresize(CN, out_size);
    end
    % put colornames features into output structure
    out(:,:,channel_id:(channel_id + size(w2c, 2) - 1)) = CN;
    channel_id = channel_id + size(w2c,2);
end

% multiply with cosine window
if ~isempty(cos_win)
    out = bsxfun(@times, out, cos_win);
end

end  % endfunction
