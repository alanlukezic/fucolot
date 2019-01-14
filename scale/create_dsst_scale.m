function filter = create_dsst_scale(init_params, img, c, base_target_sz, ...
    template_size)

    % scale adaptation parameters (from DSST)
    currentScaleFactor = init_params.currentScaleFactor;
    n_scales = init_params.n_scales;
    scale_model_factor = init_params.scale_model_factor;
    scale_sigma_factor = init_params.scale_sigma_factor;
    scale_step = init_params.scale_step;
    scale_model_max_area = init_params.scale_model_max_area;
    scale_sigma = sqrt(n_scales) * scale_sigma_factor;
    scale_lr = init_params.scale_lr;     % learning rate parameter
    
    use_mex = init_params.use_mex;

    %label function for the scales
    ss = (1:n_scales) - ceil(n_scales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = single(fft(ys));

    if mod(n_scales,2) == 0
        scale_window = single(hann(n_scales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(n_scales));
    end

    ss = 1:n_scales;
    scaleFactors = scale_step.^(ceil(n_scales/2) - ss);

    template_size_ = template_size;
    if scale_model_factor^2 * prod(template_size_) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(template_size_));
    end

    scale_model_sz = floor(template_size_ * scale_model_factor);
    scaleSizeFactors = scaleFactors;
    min_scale_factor = scale_step ^ ceil(log(max(3 ./ sqrt(template_size_))) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(img,1) size(img,2)] ./ base_target_sz)) / log(scale_step));

    xs = get_scale_subw(img, c([2,1]), base_target_sz([2,1]), ...
    currentScaleFactor * scaleSizeFactors, scale_window, ...
    scale_model_sz([2,1]), use_mex);
    % fft over the scale dim
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);

    filter.n_scales = n_scales;
    filter.scale_lr = scale_lr;
    filter.min_scale_factor = min_scale_factor;
    filter.max_scale_factor = max_scale_factor;
    filter.sf_num = new_sf_num;
    filter.sf_den = new_sf_den;
    
    filter.currentScaleFactor = currentScaleFactor;
    filter.scale_model_sz = scale_model_sz;
    filter.scale_window = scale_window;
    filter.scaleSizeFactors = scaleSizeFactors;
    filter.ysf = ysf;
    
    filter.scale_lr = init_params.scale_lr;     % learning rate parameter
    filter.base_target_sz = base_target_sz;
    filter.use_mex = use_mex;

end  % endfunction
