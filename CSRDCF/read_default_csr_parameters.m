function parameters = read_default_csr_parameters(p)

    parameters.save_output_stats = false;

    % filter parameters
    parameters.padding = 3;
    parameters.learning_rate = 0.02;
    parameters.feature_type = {'hog', 'cn'};
    parameters.y_sigma = 1;
    parameters.channels_weight_lr = parameters.learning_rate;
    parameters.use_channel_weights = true;

    % segmentation parameters
    parameters.hist_lr = 0.04;
    parameters.nbins = 16;  % N bins for segmentation
    parameters.seg_colorspace = 'hsv';     % 'rgb' or 'hsv'
    parameters.use_segmentation = true;  % false to disable use of segmentation
    parameters.mask_diletation_type = 'disk';  % for function strel (square, disk, ...)
    parameters.mask_diletation_sz = 1;

    parameters.use_mex = false;
    
    % scale adaptation parameters (from DSST)
    parameters.estimate_scale = true;
    parameters.currentScaleFactor = 1.0;
    parameters.n_scales = 33;
    parameters.scale_model_factor = 1.0;
    parameters.scale_sigma_factor = 1/4;
    parameters.scale_step = 1.02;
    parameters.scale_model_max_area = 32*16;
    parameters.scale_lr = 0.025;
    
    % background filter update
    parameters.update_background = false;
    parameters.lr_background = 0.95;  % 0.1
    parameters.background_sigma_factor = 0.4;  % 0.5
    
    % windowing function for tracker and detector
    parameters.window_type = 'hann';
    parameters.window_param = 0;
    
    % overwrite parameters that come frome input argument
    if nargin > 0
    	fields = fieldnames(p);

    	for i=1:numel(fields)
            if ~isfield(parameters, fields{i})
                warning('Setting parameter value for: %s. It is not set by default.', fields{i});
            end
    		parameters = setfield(parameters, fields{i}, p.(fields{i}));
    	end
    end

end  % endfunction
