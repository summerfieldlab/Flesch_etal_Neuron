function betas = fitSigmoid(data)
%% FITSIGMOID(DATA)

betas = nlinfit(data(:,1),data(:,2),@mysigmoid,[1]);

end

function fun = mysigmoid(b,x)

% fun = b(3) + (1-b(3)*2)./(1+exp(-b(1)*(x-b(2))));
% fun = 1./(1+exp(-b(1)*(x-b(2))));
fun = 1./(1+exp(-b(1)*(x)));



end
