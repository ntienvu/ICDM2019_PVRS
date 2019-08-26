function val=integration(y)
    y
    try
        fprintf('%s ',y);
    catch
        fprintf('BUG');
    end

    val=y - pdfGMM(y).* log(pdfGMM(y));
    %fprintf('%f ',val);

end


function out=pdfGMM(x,means,covs,weights)
    out=GMMpdf(x, means, covs, weights);
    %fprintf('%f ',out);

end 