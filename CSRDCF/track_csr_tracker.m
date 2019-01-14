function tracker = track_csr_tracker(tracker, img)

    tracker.frame = tracker.frame + 1;

    % extract features
    f = get_csr_features(img, tracker.c, tracker.currentScaleFactor, ...
        tracker.template_size, tracker.rescale_template_size, ...
        tracker.window, tracker.feature_type, tracker.w2c, ...
        tracker.cell_size, tracker.use_mex);

    if ~tracker.use_channel_weights
        response = real(ifft2(sum(fft2(f).*conj(tracker.H), 3)));
    else
        response_chann = real(ifft2(fft2(f).*conj(tracker.H)));
        response = sum(bsxfun(@times, response_chann, reshape(tracker.chann_w, 1, 1, size(response_chann,3))), 3);
    end
    
    tracker.c_prev = tracker.c;
    tracker.response = response;
    
    % calculate detection-based weights
    if tracker.use_channel_weights
        channel_discr = ones(1, size(response_chann, 3));
        for i = 1:size(response_chann, 3)
            norm_response = normalize_img(response_chann(:, :, i));
            local_maxs_sorted = localmax_nonmaxsup2d(squeeze(norm_response(:, :)));

            if local_maxs_sorted(1) == 0, continue; end;
            channel_discr(i) = 1 - (local_maxs_sorted(2) / local_maxs_sorted(1));

            % sanity checks
            if channel_discr(i) < 0.5, channel_discr(i) = 0.5; end;
        end
        tracker.channel_discr = channel_discr;
    end
    
    % new object center
    d = estimate_displacement(response, tracker.currentScaleFactor, ...
        tracker.cell_size, tracker.rescale_ratio);
    c = tracker.c + d;

    if tracker.estimate_scale
        %do a scale space search aswell
        xs = get_scale_subwindow(img, c([2,1]), tracker.base_target_sz([2,1]), ...
            tracker.currentScaleFactor * tracker.scaleSizeFactors, ...
            tracker.scale_window, tracker.scale_model_sz([2,1]), []);
        xsf = fft(xs,[],2);
        % scale correlation response
        scale_response = real(ifft(sum(tracker.sf_num .* xsf, 1) ./ (tracker.sf_den + 1e-2) ));
        recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
        %set the scale
        currentScaleFactor = tracker.currentScaleFactor * tracker.scaleFactors(recovered_scale);

        % check for min/max scale
        if currentScaleFactor < tracker.min_scale_factor
            currentScaleFactor = tracker.min_scale_factor;
        elseif currentScaleFactor > tracker.max_scale_factor
            currentScaleFactor = tracker.max_scale_factor;
        end
        % new tracker scale
        tracker.currentScaleFactor = currentScaleFactor;
    end
    
    % object bounding-box
    region = [c - tracker.currentScaleFactor * tracker.base_target_sz/2, ...
        tracker.currentScaleFactor * tracker.base_target_sz];

    % put new object location into the tracker structure
    tracker.c = c;
    tracker.bb = region;
    
end  % endfunction


function [local_max] = localmax_nonmaxsup2d(response)
    BW = imregionalmax(response);
    CC = bwconncomp(BW);

    local_max = [max(response(:)) 0];
    if length(CC.PixelIdxList) > 1
        local_max = zeros(length(CC.PixelIdxList));
        for i = 1:length(CC.PixelIdxList)
            local_max(i) = response(CC.PixelIdxList{i}(1));
        end
        local_max = sort(local_max, 'descend');
    end
end  % endfunction


function out = normalize_img(img)
    min_val = min(img(:));
    max_val = max(img(:));
    if (max_val - min_val) > 0
        out = (img - min_val)/(max_val - min_val);
    else
        out = zeros(size(img));
    end
end  % endfunction
