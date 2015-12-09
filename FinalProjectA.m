%%Jason Pang EE 660 Final Project A
pretrainDat=readtable('train.csv');
testDat=readtable('test.csv');
pretrainType=pretrainDat.(1);%%TripType
%%4,5,6,7,8,9,12,14,15,18,19,20,21,22,23,24,25
%%26,27,28,29,30,31,32,33,34,35,36,37,37,39,40
%%41,42,43,44,999
pretrainVNum=pretrainDat.(2);%%VisitNumber
%%95674visits
pretrainDay=pretrainDat.(3);%%Weekday
%%Fri,Mon,Sat,Sun,Thur,Tue,Wed
pretrainUPC=pretrainDat.(4);%%Upc
%%101843...
pretrainSCT=pretrainDat.(5);%%ScanCount
%%-12,-10,-9,-7,-6,-5,-4,-3,-2,-1,1
%%2:20,22,23,24,25,30,31,46,51,71
pretrainDept=pretrainDat.(6);%%Department, 39 diff
%%
pretrainFNum=pretrainDat.(7);%%FinelineNumber,9324

%%VisitNumber sorts data by the shopping grp
%%so we want to assign trip types to each vist #

%%Preprocessing
%%Some values are missing (did not know this before)
%%so we have to fill them. No UPC value is 0 so any
%%missing UPC value will be assigned 0. The max
%%fineline number is 9998 so any missing fineline#
%%will be assigned the value 9999. Categorical data
%%such as weekday will be assigned numerical
%%representation (e.g. 1-7 for Sun->Sat)
%%or 1000000->0000001 for Sun->Sat
pretrainUPC(find(isnan(pretrainUPC)))=0;
pretrainFNum(find(isnan(pretrainFNum)))=9999;
scramble=randperm(647054);
prey=pretrainType(scramble);
preVNum=pretrainVNum(scramble);
preday=pretrainDay(scramble);
preUPC=pretrainUPC(scramble);
preSCT=pretrainSCT(scramble);
preDept=pretrainDept(scramble);
preFNum=pretrainFNum(scramble);
% biny=zeros(length(prey),length(unique(prey)));
% binVNum=zeros(length(preVNum),length(unique(preVNum)));
% binday=zeros(length(preday),length(unique(preday)));
% binUPC=zeros(length(preUPC),length(unique(preUPC)));
% binSCT=zeros(length(preSCT),length(unique(preSCT)));
% binDept=zeros(length(preDept),length(unique(preDept)));
% binFNum=zeros(length(preFNum),length(unique(preFNum)));
%%size restriction: matrices >10000 are a struggle
%%%Training
dept1=preDept(1:100000);
FNum1=preFNum(1:100000);
prey1=prey(1:100000);
argd=unique(pretrainDept);
argf=unique(pretrainFNum);
argy1=unique(prey);
argy2=eye(38);
dept1g=cell(1,10);
fnum1g=cell(1,10);
y1=cell(1,10);
y2=cell(1,10);
w_mled1=cell(1,10);
w_mlef1=cell(1,10);
w_mled2=cell(1,10);
w_mlef2=cell(1,10);
for j=1:10 %%can take a while
    subd=dept1((10000*(j-1)+1):10000*j);
    subf=FNum1((10000*(j-1)+1):10000*j);
    suby=prey1((10000*(j-1)+1):10000*j);
    dept1g{j}=zeros(length(argd),10000);
    fnum1g{j}=zeros(length(argf),10000);
    for i=1:10000
        arg1=find(ismember(argd,intersect(subd(i),argd)));
        arg2=find(ismember(argf,intersect(subf(i),argf)));
        %%find index
        dept1g{j}(arg1,i)=1;
        fnum1g{j}(arg2,i)=1;
        %%binarize
        arg3=find(ismember(argy1,intersect(suby(i),argy1)));
        y2{j}(arg3,i)=1;
    end
    y1{j}=prey1(10000*(j-1)+1:10000*j);%%nonbinary y
    w_mled1{j} = dept1g{j}'\y1{j};
    w_mlef1{j} = fnum1g{j}'\y1{j}; %%slow
    w_mled2{j} = dept1g{j}'\y2{j}';
    w_mlef2{j} = fnum1g{j}'\y2{j}'; %%slow

%%zeros cause rank deficient warning
end

% plot(w_mled1{1},'kx')
% xlabel('Index');
% ylabel('Weights');
% title('Department Influence');
%%MLE, department
w_mled1bar=zeros(1,69);
w_mlef1bar=zeros(1,5196);
w_mled2bar=zeros(38,69);
w_mlef2bar=zeros(38,5196);
for i=1:69
    count=0;
    for j=1:10 %%weight averaging for nonzeros
        w_mled1bar(i)=w_mled1bar(i)+w_mled1{j}(i);
        if w_mled1{j}(i)~=0
            count=count+1;
        end
    end
    if w_mled1bar(i)~=0
        w_mled1bar(i)=w_mled1bar(i)/count;
    end
end
%%incredible spikes at 15,50
%%for concept stores, and other departments
for i=1:5196
    count=0;
    for j=1:10 %%weight averaging for nonzeros
        w_mlef1bar(i)=w_mlef1bar(i)+w_mlef1{j}(i);
        if w_mlef1{j}(i)~=0
            count=count+1;
        end
    end
    if w_mlef1bar(i)~=0
        w_mlef1bar(i)=w_mlef1bar(i)/count;
    end
end
clear w_mled1;
clear w_mlef1;
for k=1:38
    for i=1:69
        count=0;
        for j=1:10 %%weight averaging for nonzeros
            w_mled2bar(k,i)=w_mled2bar(k,i)+w_mled2{j}(i,k);
            if w_mled2{j}(i,k)~=0
                count=count+1;
            end
        end
        if w_mled2bar(k,i)~=0
            w_mled2bar(k,i)=w_mled2bar(k,i)/count;
        end
    end
end
for k=1:38
    for i=1:5196
        count=0;
        for j=1:10 %%weight averaging for nonzeros
            w_mlef2bar(k,i)=w_mlef2bar(k,i)+w_mlef2{j}(i,k);
            if w_mlef2{j}(i,k)~=0
                count=count+1;
            end
        end
        if w_mlef2bar(k,i)~=0
            w_mlef2bar(k,i)=w_mlef2bar(k,i)/count;
        end
    end
end
clear w_mled2;
clear w_mlef2;
y1re=prey1;
y2re=horzcat(y2{:});%%for insample error
x1d=horzcat(dept1g{:});
clear dept1g;
y1hatd=w_mled1bar*x1d;
y2hatd=w_mled2bar*x1d;
clear x1d;

y1hatf=zeros(1,100000);
y2hatf=zeros(38,100000);
for j=1:10
    %%x1f=horzcat(fnum1g{:});%%too large
    x1f=fnum1g{j};
    prey1hatf=w_mlef1bar*x1f;
    prey2hatf=w_mlef2bar*x1f;
    y1hatf(10000*(j-1)+1:10000*j)=prey1hatf;
    y2hatf(:,10000*(j-1)+1:10000*j)=prey2hatf;
end

clear fnum1g;
clear x1f;%%even with this we are low on memory
%%absolute in sample error, not classified
rougherry1d=sum(((y1re-y1hatd')./y1re).^2)/100000;
rougherry1f=sum(((y1re-y1hatf')./y1re).^2)/100000;
%%^isn't normalized, so bad measure
rougherry2d=sum(sum((y2re-y2hatd).^2))/100000;
rougherry2f=sum(sum((y2re-y2hatf).^2))/100000;
%%rougherry1d=54.5230, rougherry1f=32.7617
%%rougherry2d=.9136, rougherry2f=1.1995

%%%%Classifier
for i=1:100000
    arg1db=10000;%%arbitrary
    arg1fb=10000;
    arg2db=10000;
    arg2fb=10000;
    arg3d1=38;
    arg3f1=5196;
    arg3d2=38;
    arg3f2=5196;
    for j=1:38
        arg1da=abs(argy1(j)-abs(y1hatd(i)));
        if arg1da<arg1db
            arg1db=arg1da;
            arg3d1=j;
        end
        y1hatdc(i)=argy1(arg3d1);
        arg2da=sum(abs(argy2(:,j)-abs(y2hatd(:,i))));
        if arg2da<arg2db
            arg2db=arg2da;
            arg3d2=j;
        end
        y2hatdc(:,i)=argy2(:,arg3d2);
        arg1fa=sum(abs(argy1(j)-abs(y1hatf(i))));
        if arg1fa<arg1fb
            arg1fb=arg1fa;
            arg3f1=j;
        end
        y1hatfc(i)=argy1(arg3f1);
        arg2fa=abs(argy2(:,j)-abs(y2hatf(:,i)));
        if arg2fa<arg2fb
            arg2fb=arg2fa;
            arg3f2=j;
        end
        y2hatfc(:,i)=argy2(:,arg3f2);
    end
end
sum1d=0;
sum1f=0;
sum2d=0;
sum2f=0;
for i=1:100000
    arg1d=1-isequal(y1re(i),y1hatdc(i));
    arg1f=1-isequal(y1re(i),y1hatfc(i));
    sum1d=sum1d+arg1d;
    sum1f=sum1f+arg1f;
    arg2d=1-isequal(y2re(:,i),y2hatdc(:,i));
    arg2f=1-isequal(y2re(:,i),y2hatfc(:,i));
    sum2d=sum2d+arg2d;
    sum2f=sum2f+arg2f;
end
%%classified in sample error
erry1d=sum1d/100000;
erry1f=sum1f/100000;
%%^isn't normalized, so bad measure
erry2d=sum2d/100000;
erry2f=sum2f/100000;
%%actually this performance is terrible

%%%Validation
dept2=preDept(100001:200000);
FNum2=preFNum(100001:200000);
prey2=prey(100001:200000);
dept2g=cell(1,10);
fnum2g=cell(1,10);
y1=cell(1,10);
y2=cell(1,10);

for j=1:10 %%can take a while
    subd=dept2((10000*(j-1)+1):10000*j);
    subf=FNum2((10000*(j-1)+1):10000*j);
    suby=prey2((10000*(j-1)+1):10000*j);
    dept2g{j}=zeros(length(argd),10000);
    fnum2g{j}=zeros(length(argf),10000);
    for i=1:10000
        arg1=find(ismember(argd,intersect(subd(i),argd)));
        arg2=find(ismember(argf,intersect(subf(i),argf)));
        %%find index
        dept2g{j}(arg1,i)=1;
        fnum2g{j}(arg2,i)=1;
        %%binarize
        arg3=find(ismember(argy1,intersect(suby(i),argy1)));
        y2{j}(arg3,i)=1;
    end
    y1{j}=prey2(10000*(j-1)+1:10000*j);%%nonbinary y

end
x2d=horzcat(dept2g{:});
y1hatd=w_mled1bar*x2d;
y2hatd=w_mled2bar*x2d;
y1hatf=zeros(1,100000);
y2hatf=zeros(38,100000);
for j=1:10
    %%x1f=horzcat(fnum1g{:});%%too large
    x2f=fnum2g{j};
    prey1hatf=w_mlef1bar*x2f;
    prey2hatf=w_mlef2bar*x2f;
    y1hatf(10000*(j-1)+1:10000*j)=prey1hatf;
    y2hatf(:,10000*(j-1)+1:10000*j)=prey2hatf;
end
%%%%Classifier
for i=1:100000
    arg1db=10000;%%arbitrary
    arg1fb=10000;
    arg2db=10000;
    arg2fb=10000;
    arg3d1=38;
    arg3f1=5196;
    arg3d2=38;
    arg3f2=5196;
    for j=1:38
        arg1da=abs(argy1(j)-abs(y1hatd(i)));
        if arg1da<arg1db
            arg1db=arg1da;
            arg3d1=j;
        end
        y1hatdc(i)=argy1(arg3d1);
        arg2da=sum(abs(argy2(:,j)-abs(y2hatd(:,i))));
        if arg2da<arg2db
            arg2db=arg2da;
            arg3d2=j;
        end
        y2hatdc(:,i)=argy2(:,arg3d2);
        arg1fa=sum(abs(argy1(j)-abs(y1hatf(i))));
        if arg1fa<arg1fb
            arg1fb=arg1fa;
            arg3f1=j;
        end
        y1hatfc(i)=argy1(arg3f1);
        arg2fa=abs(argy2(:,j)-abs(y2hatf(:,i)));
        if arg2fa<arg2fb
            arg2fb=arg2fa;
            arg3f2=j;
        end
        y2hatfc(:,i)=argy2(:,arg3f2);
    end
end
sum1d=0;
sum1f=0;
sum2d=0;
sum2f=0;
for i=1:100000
    arg1d=1-isequal(y1re(i),y1hatdc(i));
    arg1f=1-isequal(y1re(i),y1hatfc(i));
    sum1d=sum1d+arg1d;
    sum1f=sum1f+arg1f;
    arg2d=1-isequal(y2re(:,i),y2hatdc(:,i));
    arg2f=1-isequal(y2re(:,i),y2hatfc(:,i));
    sum2d=sum2d+arg2d;
    sum2f=sum2f+arg2f;
end
%%classified in sample error
erryval1d=sum1d/100000;
erryval1f=sum1f/100000;
%%^isn't normalized, so bad measure
erryval2d=sum2d/100000;
erryval2f=sum2f/100000;
