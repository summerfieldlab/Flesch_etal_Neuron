function d = compute_cohensD(testType,mu1,sd1,mu2,sd2)
  %% COMPUTE_COHENSD(TESTTYPE,MU1,SD1,MU2,SD2)
  %
  % computes the effect size of a one- or two-sample
  % t-test (Cohen's d)
  %
  % Inputs:
  % testType: t or t2 for one-sample and two-sample tests
  % mu1: mean of group 1
  % sd1: standard deviation of group 1
  % mu2: mean of group 2 (t2) or of H0 (t)
  % sd2: standard deviation of group 2 (t2)
  %
  % Output:
  % d: cohen's d
  %
  % How to interpret:
  % 0.2,0.5,0.8: small,medium,large
  %
  % Timo Flesch, 2017

  switch testType
  case 't'
      if ~exist('mu2')
        mu2 = 0;
      end
      % mu2 is the mean under h0
      d = (mu1-mu2)./sd1;
  case 't2'
      sdPool =  sqrt((sd1^2+sd2^2)/2);
      d = (mu1-mu2)/sdPool;
  end

end
