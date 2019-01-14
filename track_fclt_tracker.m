function [tracker, region] = track_fclt_tracker(tracker, img)

    tracker.frame = tracker.frame + 1;
    
    tracker.img_prev = img;
    
    c_prev = tracker.c;
    lt_state_prev = tracker.lt_state;
    
    tracker.localizer = track_csr_tracker(tracker.localizer, img);
    response = tracker.localizer.response;
    tracker.response = response;
    tracker.c_prev = c_prev;
    tracker.c = tracker.localizer.c;
    tracker.bb = tracker.localizer.bb;
    
    curr_score = [];
    resp_quality_tracker = Inf;
    
    resp_quality = calculate_resp_quality(response);
    if numel(tracker.resp_budg) >= tracker.skip_check_beginning
        response_budget_mean = mean(tracker.resp_budg);
        curr_quality_norm = resp_quality / tracker.resp_norm;
        resp_quality_tracker = curr_quality_norm;
        curr_score = (response_budget_mean - curr_quality_norm) / curr_quality_norm;
        if curr_score > tracker.detect_failure
            tracker.lt_state = 2;  % target lost
        else
            tracker.lt_state = 1;  % ok
        end
    else
        tracker.lt_state = 1;  % ok
    end
    
    % check for target redetection
    if (tracker.redetect && tracker.lt_state == 2 && lt_state_prev == 2)
        
        % construct larger filter and apply it to the whole image
        % calculate resize ratio for image
        ratio = (tracker.last_scale_factor * tracker.detector.template_size) ./ ...
            tracker.detector.rescale_template_size;
        if ratio(1) ~= ratio(2)
            error('Ratios are not the same sizes');
        end
        ratio = ratio(1);
        img_rescale_sz = round([size(img,1), size(img,2)] / ratio);

        % go with detector over large range of scales
        det_scale = tracker.det_scales(tracker.det_scale_idx);
        % new target size
        sz_ = det_scale * tracker.last_scale_factor * ...
            tracker.detector.base_target_sz;
        % bound so that target size is not too large or too small
        if sz_(1) < 0.5*size(img,1) && sz_(1) > 15 && ...
                sz_(2) < 0.5*size(img,2) && sz_(2) > 15
            img_rescale_sz = det_scale * img_rescale_sz;
        else
            det_scale = 1;
        end

        % resize image and extract features
        if tracker.use_mex
            img_res = mexResize(img, img_rescale_sz, 'auto');
        else
            img_res = imresize(img, img_rescale_sz);
        end
        
        img_f = extract_features_only(img_res, [], ...
            tracker.detector.feature_type, ...
            tracker.detector.w2c, tracker.detector.cell_size, ...
            tracker.use_mex);
        
        if ~tracker.detection_filter_created
            
            H_det = tracker.H_history_budget{tracker.det_filter_idx};

            if tracker.det_scale_idx == numel(tracker.det_scales)
                tracker.det_filter_idx = tracker.det_filter_idx - 1;
                if tracker.det_filter_idx < 1
                    tracker.det_filter_idx = numel(tracker.H_history_budget);
                end
            end

            tracker.det_scale_idx = tracker.det_scale_idx + 1;
            if tracker.det_scale_idx > numel(tracker.det_scales)
                tracker.det_scale_idx = 1;
            end

            h = real(ifft2(H_det));
            tracker.H_det = H_det;
            
            % choose window function for extracted detector features
            if strcmp(tracker.detector_window_type, 'tricube')
                % tricube
                pw = tracker.detector_window_param;
                win_x = ((1 - abs(linspace(-1,1,size(img_f,2))).^pw).^pw);
                win_y = ((1 - abs(linspace(-1,1,size(img_f,1))).^pw).^pw);
                win_img = win_y'*win_x;
            elseif strcmp(tracker.detector_window_type, 'tukey')
                % Tukey
                alpha = tracker.detector_window_param;
                win_x = tukeywin(size(img_f,2), alpha);
                win_y = tukeywin(size(img_f,1), alpha);
                win_img = win_y*win_x';
            elseif strcmp(tracker.detector_window_type, 'hann')
                % Hanning
                win_img = hann(size(img_f,1)) * hann(size(img_f,2))';
            elseif strcmp(tracker.detector_window_type, 'uniform')
                % Uniform (the same as without window)
                win_img = ones(size(img_f,1), size(img_f,2));
            else
                error('Unknown detector window type.');
            end
            
            % apply window on image features and transform to Fourier
            img_f = bsxfun(@times, img_f, win_img);
            
            % create image-sized zero-padded filter
            h_img = zeros(size(img_f,1), size(img_f,2), size(img_f,3));
            % calculate indexes for inserting filter into zero-paded matrix
            if size(h_img,1) > size(h,1)
                y1_img = max(1, floor(size(h_img,1) / 2 - size(h,1) / 2));
                y1_h = 1;
                y2_img = y1_img + size(h,1) - 1;
                y2_h = size(h,1);
            elseif size(h_img,1) < size(h,1)
                y1_img = 1;
                y1_h = max(1, floor(size(h,1) / 2 - size(h_img,1) / 2));
                y2_img = size(h_img,1);
                y2_h = y1_h + size(h_img,1) - 1;
            else
                y1_img = 1;
                y1_h = 1;
                y2_img = size(h_img,1);
                y2_h = size(h,1);
            end
            if size(h_img,2) > size(h,2)
                x1_img = max(1, floor(size(h_img,2) / 2 - size(h,2) / 2));
                x1_h = 1;
                x2_img = x1_img + size(h,2) - 1;
                x2_h = size(h,2);
            elseif size(h_img,2) < size(h,2)
                x1_img = 1;
                x1_h = max(1, floor(size(h,2) / 2 - size(h_img,2) / 2));
                x2_img = size(h_img,2);
                x2_h = x1_h + size(h_img,2) - 1;
            else
                x1_img = 1;
                x1_h = 1;
                x2_img = size(h_img,2);
                x2_h = size(h,2);
            end
            % insert filter and transform it to Fourier
            h_img(y1_img:y2_img, x1_img:x2_img, :) = h(y1_h:y2_h, x1_h:x2_h, :);
            H_img = fft2(h_img);
            
            tracker.H_img = H_img;
            tracker.win_img = win_img;
            
        else
            H_img = tracker.H_img;
            % apply window on image features and transform to Fourier
            img_f = bsxfun(@times, img_f, tracker.win_img);
        end
        
        % features on whole image
        F_img = fft2(img_f);
        
        % calculate correlation response on whole image
        % using channel weights or not
        response_chann_img = real(ifft2(F_img.*conj(H_img)));
        chann_w = ones(size(response_chann_img,3), 1) * ...
            (1.0 / size(response_chann_img,3));
        response_img = sum(bsxfun(@times, response_chann_img, ...
            reshape(chann_w, 1, 1, size(response_chann_img,3))), 3);
        
        response_img = fftshift(response_img);
        
        % Gaussian motion prior
        exponent_idx = tracker.frame - tracker.last_ok_frame - 2;
        size_factor = 1.05 ^ exponent_idx;
        sigma_factor = 0.5;
        % construct Gauss prior for detector
        [Y_,X_] = ndgrid((1:size(img,1)) - tracker.last_c(2), ...
            (1:size(img,2)) - tracker.last_c(1));
        sz_ = size_factor * det_scale * tracker.last_scale_factor * ...
            tracker.detector.base_target_sz;
        gauss_prior = exp(-0.5 * ( ((X_.^2)/(sigma_factor*sz_(1))^2) + ...
            ((Y_.^2)/((sigma_factor*sz_(2))^2)) ) );  % 0.5
        G = mexResize(gauss_prior, size(response_img), 'auto');

        % apply Gaussian motion prior on detector tracking response
        response_img = response_img .* G;
        
        % calculate target position estimated with detector
        [row_img, col_img] = ind2sub(size(response_img), ...
            find(response_img == max(response_img(:)), 1));
        v_neighbors_img = response_img(mod(row_img + [-1, 0, 1] - 1, ...
            size(response_img,1)) + 1, col_img);
        h_neighbors_img = response_img(row_img, ...
            mod(col_img + [-1, 0, 1] - 1, size(response_img,2)) + 1);
        row_img = row_img + subpixel_peak(v_neighbors_img);
        col_img = col_img + subpixel_peak(h_neighbors_img);
        % displacement and new bounding box
        pos_img = 1/det_scale * tracker.last_scale_factor * ...
            tracker.detector.cell_size * ...
            (1/tracker.detector.rescale_ratio) * [col_img - 1, row_img - 1];
        
        bb_img = [pos_img - det_scale * tracker.last_scale_factor * ...
            tracker.detector.base_target_sz/2, ...
            det_scale * tracker.last_scale_factor * tracker.detector.base_target_sz];
        
        % apply short-term CF on new position
        % extract features here
        f_ = get_csr_features(img, pos_img, det_scale*tracker.last_scale_factor, ...
            tracker.detector.template_size, tracker.detector.rescale_template_size, ...
            tracker.detector.window, tracker.detector.feature_type, ...
            tracker.detector.w2c, tracker.detector.cell_size, ...
            tracker.use_mex);
        
        response_ = real(ifft2(mean(fft2(f_).*conj(tracker.H_det), 3)));

        % calculate response quality
        resp_quality_ = calculate_resp_quality(response_);
        response_budget_mean = mean(tracker.resp_budg);
        curr_quality_norm = resp_quality_ / tracker.resp_norm;
        curr_score_ = (response_budget_mean - curr_quality_norm) / curr_quality_norm;
        % check if target is found
        if curr_score_ > tracker.detect_recover
            
            % target has not been found
            tracker.lt_state = 2;  % target lost
            % check if response quality is here better than 
            % on the position of short-term tracker
            if curr_quality_norm > resp_quality_tracker
                % take this correlation response and output position
                resp_quality = resp_quality_;
                tracker.response = response_;
                tracker.c_prev = pos_img;
                % find position of the maximum
                d = estimate_displacement(response_, ...
                    det_scale*tracker.last_scale_factor, ...
                    tracker.detector.cell_size, ...
                    tracker.detector.rescale_ratio);
                % previous tracking center needs to be changed
                % so that current displacement is used correctly
                % to estimate new target position
                tracker.c = pos_img + d;

                if ~around_edge(0, 0, [], 0, 0, 0, tracker.c, ...
                        size(img,2), size(img,1), bb_img(3), bb_img(4))
                    tracker.currentScaleFactor = det_scale * tracker.last_scale_factor;
                end
            end

        else
            % target has been re-detected
            resp_quality = resp_quality_;
            tracker.response = response_;
            tracker.c_prev = pos_img;
            % new target position
            d = estimate_displacement(response_, ...
                det_scale*tracker.last_scale_factor, ...
                tracker.detector.cell_size, ...
                tracker.detector.rescale_ratio);
            tracker.c = pos_img + d;
            tracker.H = tracker.H_det;
            curr_score = curr_score_;
            if ~around_edge(0, 0, [], 0, 0, 0, tracker.c, ...
                    size(img,2), size(img,1), tracker.bb(3), tracker.bb(4))
                tracker.lt_state = 1;
                tracker.currentScaleFactor = det_scale * tracker.last_scale_factor;
            end
        end
        
        % store detector position and bbox only for visualization
        tracker.pos_img = pos_img;
        tracker.bb_img = bb_img;
        tracker.H_det = H_det;
        
        % object bounding-box
        region = [tracker.c - tracker.currentScaleFactor * tracker.detector.base_target_sz/2, ...
            tracker.currentScaleFactor * tracker.detector.base_target_sz];

        % put new object location into the tracker structure
        tracker.bb = region;
        
    else
        
        % tracker is tracking - not necessary to run the detector
        % store output position
        tracker.c = tracker.localizer.c;
        tracker.bb = tracker.localizer.bb;
        region = tracker.bb;
        
    end
    
    % add current correlation response quality to the budget,
    % but do not add it if target is identified as lost
    if tracker.lt_state == 1
        if isempty(tracker.resp_budg)
            % in first localization frame response score needs to be added as
            % response normalization score
            % therefore 1 is added into the budget
            tracker.resp_norm = resp_quality;
            tracker.resp_budg(end+1) = 1;
        else
            % normalize current response score and add it to the budget
            tracker.resp_budg(end+1) = resp_quality / tracker.resp_norm;
            % if budget is reached, remove first (the oldest) element
            if numel(tracker.resp_budg) > tracker.resp_budg_sz
                tracker.resp_budg(1) = [];
            end
        end
        
        % reset detection filter created flag
        tracker.detection_filter_created = false;
        % only to reduce tracker structure size
        tracker.H_img = [];
        tracker.win_img = [];
    end
    
    tracker.q = resp_quality;
    tracker.q_norm = resp_quality / tracker.resp_norm;
    tracker.curr_score = curr_score;

    % set new target center to the localizer 
    % if it was chenged by the detector
    tracker.localizer.c = tracker.c;
    
    % store position and size of last tracker target
    if tracker.lt_state == 1
        tracker.last_ok_frame = tracker.frame;
        tracker.last_c = tracker.c;
    end
    
    % if target is around edge do not estimate scale and do not update
    if around_edge(0, 0, [], 0, 0, 0, tracker.c, ...
            size(img,2), size(img,1), tracker.bb(3), tracker.bb(4)) 
        return;
    end
    
    % estimate scale change
    tracker.scale_estimator.currentScaleFactor = tracker.currentScaleFactor;
    currentScaleFactor = estimate_dsst_scale(tracker.scale_estimator, ...
        img, tracker.c);
    tracker.currentScaleFactor = currentScaleFactor;
    tracker.scale_estimator.currentScaleFactor = currentScaleFactor;
    tracker.localizer.currentScaleFactor = currentScaleFactor;
    
    % store position and size of last tracker target
    if tracker.lt_state == 1
        tracker.last_scale_factor = currentScaleFactor;
    end
    
    % do not learn if target is lost
    if tracker.lt_state == 2
        return;
    end
    
    tracker.det_filter_idx = numel(tracker.H_history_budget);
    tracker.det_scale_idx = 1;
    
    %% ------------------- LEARNING PHASE -------------------
    
    % update localizer
    tracker.localizer = update_csr_tracker(tracker.localizer, img);
    
    H_new = tracker.localizer.H_new;

    lr = tracker.localizer.learning_rate;
    % update detector budget
    for i=1:numel(tracker.detector_temporal)
        if mod(tracker.update_counter, tracker.detector_temporal(i)) == 0
            tracker.H_history_budget{i} = (1-lr) * ...
                tracker.H_history_budget{i} + lr * H_new;
        end
    end
    tracker.update_counter = tracker.update_counter + 1;
    
    % update DSST scale filter
    tracker.scale_estimator = update_dsst_scale(tracker.scale_estimator, ...
        img, tracker.c);
    
end  % endfunction


function delta = subpixel_peak(p)
	%parabola model (2nd order fit)
	delta = 0.5 * (p(3) - p(1)) / (2 * p(2) - p(3) - p(1));
	if ~isfinite(delta), delta = 0; end
end  % endfunction

function s = calculate_resp_quality(R)
% different types of scores:
% 1: maximum response value
% 2: PSR

type = 2;

if type == 1
    s = max(R(:));
elseif type == 2
    response = fftshift(R);
    [row, col] = ind2sub(size(response),find(response == max(response(:)), 1));
    max_val = response(row, col);
    x1 = min(size(R,2), max(1, col - round(0.05 * size(R,2))));
    x2 = min(size(R,2), max(1, col + round(0.05 * size(R,2))));
    y1 = min(size(R,2), max(1, row - round(0.05 * size(R,1))));
    y2 = min(size(R,2), max(1, row + round(0.05 * size(R,1))));
    M = ones(size(R));
    M(y1:y2, x1:x2) = 0;
    sidelobe = response(M==1);
    mu_s = mean(sidelobe(:));
    sigma_s = std(sidelobe(:));
    s = (max_val - mu_s) / (sigma_s + 0.0001);
    % multiply PSR with maximum value to take into account also similarity
    s = s * max_val;
end

end  % endfunction

