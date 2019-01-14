function p = read_default_scale_params()

p.currentScaleFactor = 1;
p.n_scales = 33;
p.scale_model_factor = 1.0;
p.scale_sigma_factor = 1/4;
p.scale_step = 1.02;
p.scale_model_max_area = 32 * 16;
p.scale_lr = 0.025;

p.scale_sigma = sqrt(p.n_scales) * p.scale_sigma_factor;
p.use_mex = false;

end  % endfunction
