function displacement = estimate_displacement(response, scale_factor, ...
    cell_size, rescale_ratio)

% target position
[row, col] = ind2sub(size(response),find(response == max(response(:)), 1));

% subpixel accuracy: response map is smaller than image patch -
% due to HoG histogram (cell_size > 1)
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
displacement = scale_factor * cell_size * (1/rescale_ratio) * [col - 1, row - 1];


end  % endfunction


function delta = subpixel_peak(p)
	%parabola model (2nd order fit)
	delta = 0.5 * (p(3) - p(1)) / (2 * p(2) - p(3) - p(1));
	if ~isfinite(delta), delta = 0; end
end  % endfunction
