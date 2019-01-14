function tracker = create_fclt_tracker(img, init_bbox, init_params)

    if nargin < 3
        init_params = read_default_fclt_parameters();
    end

    % localizer
    localizer_params = read_default_csr_parameters();
    localizer_params.estimate_scale = false;
    localizer = create_csr_tracker(img, init_bbox, localizer_params);
    
    center = localizer.c;
    
    % scale filter
    scale_params = read_default_scale_params();
    scale_estimator = create_dsst_scale(scale_params, img, center, ...
        localizer.base_target_sz, localizer.template_size);
    
    % detector
    detector_params = read_default_csr_parameters();
    detector_params.estimate_scale = false;
    detector = create_csr_tracker(img, init_bbox, detector_params);
    
    % filter for detector
    H = detector.H;
    
    % tracker structure
    tracker.localizer = localizer;
    tracker.scale_estimator = scale_estimator;
    tracker.c = center;
    tracker.bb = localizer.bb;
    
    tracker.img_prev = img;
    
    % for long-term component
    tracker.resp_budg_sz = init_params.resp_budg_sz;
    tracker.resp_budg = [];
    tracker.resp_norm = 0;
    tracker.lt_state = 1;  % 1-track, 2-lost
    tracker.redetect = init_params.redetect;
    tracker.h_weights = 1;
    tracker.h_detector = [];
    tracker.detect_failure = init_params.detect_failure;
    tracker.detect_recover = init_params.detect_recover;
    tracker.detector_window_type = init_params.detector_window_type;
    tracker.detector_window_param = init_params.detector_window_param;
    tracker.det_weights = init_params.det_weights;
    
    % detector filter properties
    tracker.detector.template_size = detector.template_size;
    tracker.detector.rescale_template_size = detector.rescale_template_size;
    tracker.detector.rescale_ratio = detector.rescale_ratio;
    tracker.detector.base_target_sz = detector.base_target_sz;
    tracker.detector.cell_size = detector.cell_size;
    tracker.detector.feature_type = detector.feature_type;
    tracker.detector.w2c = detector.w2c;
    tracker.detector.window = detector.window;
    
    tracker.skip_check_beginning = init_params.skip_check_beginning;
    
    tracker.currentScaleFactor = scale_estimator.currentScaleFactor;
    tracker.last_scale_factor = scale_estimator.currentScaleFactor;
    
    tracker.frame = 1;
    tracker.last_ok = 1;
    tracker.detection_filter_created = false;
    
    % detection scales
    tracker.det_scales = init_params.det_scales;
    tracker.det_scale_idx = 1;
    
    % detector updates frequencies
    tracker.detector_temporal = init_params.detector_temporal;
    tracker.update_counter = 1;
    tracker.H_history_budget = {};
    for i=1:numel(tracker.detector_temporal)
        tracker.H_history_budget{i} = H;
    end
    tracker.det_filter_idx = numel(tracker.H_history_budget);
    
    tracker.use_mex = init_params.use_mex;
        
end  % endfunction
