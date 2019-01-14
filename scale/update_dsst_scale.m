function filter = update_dsst_scale(filter, img, c)

    xs = get_scale_subw(img, c([2,1]), filter.base_target_sz([2,1]), ...
        filter.currentScaleFactor * filter.scaleSizeFactors, ...
        filter.scale_window, filter.scale_model_sz([2,1]), filter.use_mex);
    % fft over the scale dim
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, filter.ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    % auto-regressive scale filters update
    slr = filter.scale_lr;
    filter.sf_den = (1 - slr) * filter.sf_den + slr * new_sf_den;
    filter.sf_num = (1 - slr) * filter.sf_num + slr * new_sf_num;

end  % endfunction