function around = around_edge(row, col, response, scale_factor, cell_size, ...
    rescale_ratio, c, img_w, img_h, target_w, target_h)

if ~isempty(response)
    v_neighbors = response(mod(row + [-1, 0, 1] - 1, size(response,1)) + 1, col);
    h_neighbors = response(row, mod(col + [-1, 0, 1] - 1, size(response,2)) + 1);

    spv = subpixel_peak(v_neighbors);
    sph = subpixel_peak(h_neighbors);
    spv = sign(spv) * min(1, abs(spv));
    sph = sign(sph) * min(1, abs(sph));
    row = row + spv;
    col = col + sph;

    % wrap around 
    if row > size(response,1) / 2,
        row = row - size(response,1);
    end
    if col > size(response,2) / 2,
        col = col - size(response,2);
    end

    % displacement
    d = scale_factor * cell_size * ...
        (1/rescale_ratio) * [col - 1, row - 1];

    % new object center
    c = c + d;
end

around = false;
thresh_ratio = 0.25;
if c(1) < thresh_ratio * target_w
    around = true;
    return;
end
if c(2) < thresh_ratio * target_h
    around = true;
    return;
end
if c(1) > img_w - thresh_ratio * target_w
    around = true;
    return;
end
if c(2) > img_h - thresh_ratio * target_h
    around = true;
    return;
end

end  % endfunction


function delta = subpixel_peak(p)
	%parabola model (2nd order fit)
	delta = 0.5 * (p(3) - p(1)) / (2 * p(2) - p(3) - p(1));
	if ~isfinite(delta), delta = 0; end
end  % endfunction
