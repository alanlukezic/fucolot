function new_scale_factor = estimate_dsst_scale(filter, img, c)

    %do a scale space search aswell
    xs = get_scale_subw(img, c([2,1]), filter.base_target_sz([2,1]), ...
        filter.currentScaleFactor * filter.scaleSizeFactors, ...
        filter.scale_window, filter.scale_model_sz([2,1]), filter.use_mex);
    xsf = fft(xs,[],2);
    % scale correlation response
    scale_response = real(ifft(sum(filter.sf_num .* xsf, 1) ./ (filter.sf_den + 1e-2) ));
    recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
    %set the scale
    currentScaleFactor = filter.currentScaleFactor * filter.scaleSizeFactors(recovered_scale);

    % check for min/max scale
    if currentScaleFactor < filter.min_scale_factor
        currentScaleFactor = filter.min_scale_factor;
    elseif currentScaleFactor > filter.max_scale_factor
        currentScaleFactor = filter.max_scale_factor;
    end
    % new tracker scale
    new_scale_factor = currentScaleFactor;

end  % endfunction