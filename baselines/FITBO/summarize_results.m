load('min_predict_FITBO.mat')

[T,nRepeat]=size(min_predict_FITBO);

mystd=std(min_predict_FITBO')

sorted_FITBO=min_predict_FITBO;
%T=40;
for ii=1:nRepeat
    for tt=1:T
        sorted_FITBO(tt,ii)=min(min_predict_FITBO(1:tt,ii));
    end
end

myval=mean(sorted_FITBO,2);
%myerr=std(sorted_FITBO,2);
%plot(myval)
errorbar(myval,0.2*mystd');
xlabel('Iteration')
ylabel('Inference Regret f(x)')
%ylim([0,10])

%100*myval(1:40)'


load('next_eval_FITBO.mat')

[T,nRepeat]=size(next_eval_FITBO);

mystd=std(next_eval_FITBO')

sorted_FITBO=next_eval_FITBO;
%T=40;
for ii=1:nRepeat
    for tt=1:T
        sorted_FITBO(tt,ii)=min(next_eval_FITBO(1:tt,ii));
    end
end

myval=mean(sorted_FITBO,2);
%myerr=std(sorted_FITBO,2);
%plot(myval)

figure;
errorbar(myval,0.2*mystd');
xlabel('Iteration')
ylabel('Best Value Sofar f(x)')
%ylim([0,10])

