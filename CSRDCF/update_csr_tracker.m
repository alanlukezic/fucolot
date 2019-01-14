function tracker = update_csr_tracker(tracker, img)

    if tracker.use_segmentation
        % convert image in desired colorspace
        if strcmp(tracker.seg_colorspace, 'rgb')
            seg_img = img;
        elseif strcmp(tracker.seg_colorspace, 'hsv')
            seg_img = rgb2hsv(img);
            seg_img = seg_img * 255;
        else
            error('Unknown colorspace parameter');
        end

        region = tracker.bb;
        % object rectangle region: subtract 1 because C++ indexing starts with zero
        obj_reg = round([region(1), region(2), region(1)+region(3), region(2)+region(4)]) - [1 1 1 1];

        % extract histograms and update them
        hist_fg = mex_extractforeground(seg_img, obj_reg, tracker.nbins);
        hist_bg = mex_extractbackground(seg_img, obj_reg, tracker.nbins);
        tracker.hist_fg = (1-tracker.hist_lr)*tracker.hist_fg + tracker.hist_lr*hist_fg;
        tracker.hist_bg = (1-tracker.hist_lr)*tracker.hist_bg + tracker.hist_lr*hist_bg;

        % extract masked patch: mask out parts outside image
        [seg_patch, valid_pixels_mask] = get_patch(seg_img, tracker.c, ...
            tracker.currentScaleFactor, tracker.template_size);

        % segmentation
        [fg_p, bg_p] = get_location_prior([1, 1, size(seg_patch,2), size(seg_patch,1)], ...
            tracker.currentScaleFactor*tracker.base_target_sz, [size(seg_patch,2), size(seg_patch, 1)]);
        [~, fg, ~] = mex_segment(seg_patch, tracker.hist_fg, tracker.hist_bg, tracker.nbins, fg_p, bg_p);
        
        % cut out regions outside from image
        mask = single(fg).*single(valid_pixels_mask);
        mask = binarize_softmask(mask);

        % resize to filter size
        mask = imresize(mask, size(tracker.Y), 'nearest');
        
        % check if mask is too small (probably segmentation is not ok then)
        if mask_normal(mask, tracker.target_dummy_area)
            if tracker.mask_diletation_sz > 0
                D = strel(tracker.mask_diletation_type, tracker.mask_diletation_sz);
                mask = imdilate(mask, D);
            end
        else
            mask = tracker.target_dummy_mask;
        end

    else
        
        mask = tracker.target_dummy_mask;

    end

    % extract features from image
    f = get_csr_features(img, tracker.c, tracker.currentScaleFactor, ...
        tracker.template_size, tracker.rescale_template_size, ...
        tracker.window, tracker.feature_type, tracker.w2c, ...
        tracker.cell_size, tracker.use_mex);
    
    if tracker.update_background
        f = bsxfun(@times, f, (1 - tracker.inter_kernel)) + ...
            bsxfun(@times, tracker.f, tracker.inter_kernel);
    end
    
    % calcualte new filter - using segmentation mask
    H_new = create_csr_filter(f, tracker.Y, single(mask));

    % calculate per-channel feature weights
    if tracker.use_channel_weights
        w_lr = tracker.weight_lr;
        response = real(ifft2(fft2(f).*conj(H_new)));
        chann_w = max(reshape(response, ...
            [size(response,1)*size(response,2), ...
            size(response,3)]), [], 1) .* tracker.channel_discr;
        chann_w = chann_w / sum(chann_w);
        tracker.chann_w = (1-w_lr)*tracker.chann_w + w_lr*chann_w;
        tracker.chann_w = tracker.chann_w / sum(tracker.chann_w);
    end
    
    % filter update as weighted average
    lr = tracker.learning_rate;
    tracker.H = (1-lr)*tracker.H + lr*H_new;
    
    tracker.H_new = H_new;
    
    % make a scale search model aswell
    if tracker.estimate_scale
        xs = get_scale_subwindow(img, tracker.c([2,1]), tracker.base_target_sz([2,1]), ...
            tracker.currentScaleFactor * tracker.scaleSizeFactors, ...
            tracker.scale_window, tracker.scale_model_sz([2,1]), []);
        % fft over the scale dim
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, tracker.ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);
        % auto-regressive scale filters update
        slr = tracker.scale_lr;
        tracker.sf_den = (1 - slr) * tracker.sf_den + slr * new_sf_den;
        tracker.sf_num = (1 - slr) * tracker.sf_num + slr * new_sf_num;
    end

end  % endfunction
