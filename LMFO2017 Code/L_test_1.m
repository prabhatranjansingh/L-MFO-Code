clear;close all
global initial_flag
ps=50;
me=2000;
% filename1= 'SMO_Final_30D_50NP.xlsx';
header1 = {'Problem','MeanError','standardDev','MinValue','MaxValue','MeanTFE','lb','ub'};
for D=[50,100]
    FES=10000*D;
    for func_num=1:28
        %      if funcnum==5 || func_num==7 || (func_num>=9 && func_num<=21) || (func_num>=23 && func_num<=26)
        %         continue;
        %     end
        if func_num==4 || func_num==5 || func_num==9
            lb=-10;
            ub=10;
            %     elseif func_num==8
            %         lb=-20;
            %         ub=20;
        elseif func_num==7 || func_num==19 || func_num==28
            lb=-50;
            ub=50;
        else
            lb=-100;
            ub=100;
        end
        % for jjj=1:30 %run's number
        initial_flag=0;
        func_num
        %  [SMO_gbest,SMO_gbestval,SMO_fitcount]= DE('CEC2017',me,FES,ps,D,lb,ub,func_num);
        %  [SMO_gbest,SMO_gbestval,SMO_fitcount]= DEJade2('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= DEJade3('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= Shade('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= LShade(fobj,me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= Chimp(fobj,me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= LShadex('CEC2017',me,FES,ps,D,lb,ub,func_num);
        %  [SMO_gbest,SMO_gbestval,SMO_fitcount]= pso(fobj,me,FES,ps,D,lb,ub,func_num);
        %[SMO_gbest,SMO_gbestval,SMO_fitcount]= hs('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= hsorg('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= IHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= SAHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= GDHS(fobj,me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= MIWOA(fobj,me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= LHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        %[SMO_gbest,SMO_gbestval,SMO_fitcount]= hs1('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= abc('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= ranSHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= fitSHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= SelectSHS('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= SelectSHS1('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= SelectSHS2('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= SelectSHS3('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= AHS_HCM('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= GWO(fobj,me,FES,ps,D,lb,ub,func_num);
        [SMO_gbest,SMO_gbestval,SMO_fitcount]= L_MFO_1('CEC2017',me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= MFO(fobj,me,FES,ps,D,lb,ub,func_num);
        % [SMO_gbest,SMO_gbestval,SMO_fitcount]= WF_MFO(fobj,me,FES,ps,D,lb,ub,func_num);
    end
end

% filename = 'testdata30Run50NP30D.xlsx';
% % for func_num=1:28
% % %  xlswrite(filename1, header1, 'SMORESULT');
% % % xlswrite(filename,[SMO_gbestval_res(func_num,:)],[func_num 'FitR']);
% % disp('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
% % func_num
% % disp('SMO'),mean(SMO_gbestval_res(func_num,:)'),std(SMO_gbestval_res(func_num,:)')
% % MeanV(func_num)=mean(SMO_gbestval_res(func_num,:)');
% % StdD(func_num)=std(SMO_gbestval_res(func_num,:)');
% % MinValue(func_num)=min(SMO_gbestval_res(func_num,:)');
% % MaxValue(func_num)=max(SMO_gbestval_res(func_num,:)');
% % TFEval(func_num)=mean(SMO_fitcount_res(func_num,:))
% % probl(func_num)=func_num;
% % end
% xlswrite('SMO_Final_25D_50NP.xlsx', header1, 'SMORESULT');
% xlswrite('SMO_Final_25D_50NP.xlsx',[probl' MeanV' StdD' MinValue'
% MaxValue' TFEval' LowerB' UpperB'],'SMORESULT','A2');