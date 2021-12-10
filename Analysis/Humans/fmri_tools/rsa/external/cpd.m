function [cpd_out,RX]=cpd(Y,X)

% cpd_out=cpd(Y,X)
% calculate coefficient of partial demtermination for each regressor in X%
%
% Y is data, size nObservations*nIndependentTests
% X is design matrix, size nObsetvations*nRegressors
%
% cpd_out has dimensions nRegressors*nIndependentTests
%
% author: Laurence Hunt

RX=Y-X*pinv(X)*Y; %residuals from full model
SSE_X=sum(RX.^2); %sum of squared errors
cpd_out=zeros(size(X,2),size(Y,2)); %initialise output

for i=1:size(X,2) %loop over regressors
   %design matrix without regressor i:
   X_i=X;
   X_i(:,i)=[];

   RX_i=Y-X_i*pinv(X_i)*Y; %residuals from reduced model
   SSE_X_i=sum(RX_i.^2); % SSE from reduced model
   cpd_out(i,:)=(SSE_X_i-SSE_X)./SSE_X_i; %coefficient of partial determination for regressor i
end
