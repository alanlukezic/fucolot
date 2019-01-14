function parameters = read_default_fclt_parameters(p)
    
    parameters.localizer_type = 'CSR';

    % quality of correlation responses budget size
    parameters.redetect = true;
    parameters.resp_budg_sz = 100;
    parameters.skip_check_beginning = 5;
    parameters.detect_failure = 1.7;
    parameters.detect_recover = parameters.detect_failure;
    parameters.det_weights = 'uniform';  % uniform / init / learned
    
    parameters.use_mex = false;
    
    % windowing function for tracker and detector
    parameters.tracker_window_type = 'hann';
    parameters.tracker_window_param = 100;
    parameters.detector_window_type = 'tricube';
    parameters.detector_window_param = 7;
    
    % scales cycles
    parameters.det_scales = [1, 0.7, 1.2, 0.5, 1.5, 2];
    
    parameters.detector_temporal = [0,250,50,10,1];
    
%     % detector filter type
%     parameters.detector_type = 'multiple';
%     
%     % detector filter temporal scales
%     if strcmp(parameters.detector_type, 'multiple')
%         parameters.detector_temporal = [0,250,50,10,1];
%     else
%         parameters.detector_temporal = 1;
%     end
    
    % re-detect target every X frames
%     parameters.fixed_redetect = 0;%200;
    
    % update temporal scales for scale filters (multiple DSST filters)
    parameters.scales_temporal = 1;

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
