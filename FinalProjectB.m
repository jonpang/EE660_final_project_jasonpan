%%Jason Pang EE 660 Final Project B
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
dept1=preDept(1:100000);
FNum1=preFNum(1:100000);
prey1=prey(1:100000);
dept2=preDept(100000:200000);
FNum2=preFNum(100000:200000);
prey2=prey(100000:200000);
argd=unique(pretrainDept);
argf=unique(pretrainFNum);
argy1=unique(prey);
argy2=eye(38);
dept1g=cell(1,10);
fnum1g=cell(1,10);
y1=cell(1,10);
y2=cell(1,10);

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
    subd=dept1((10000*(j-1)+1):10000*j);
    subf=FNum1((10000*(j-1)+1):10000*j);
    suby=prey1((10000*(j-1)+1):10000*j);
    dept2g{j}=zeros(length(argd),10000);
    fnum2g{j}=zeros(length(argf),10000);
    for i=1:10000
        arg1=find(ismember(argd,intersect(subd(i),argd)));
        arg2=find(ismember(argf,intersect(subf(i),argf)));
        %%find index
        dept1g{j}(arg1,i)=1;
        fnum1g{j}(arg2,i)=1;
        %%binarize
        arg3=find(ismember(argy1,intersect(suby(i),argy1)));
        ytest2{j}(arg3,i)=1;
    end
    ytest1{j}=prey2(10000*(j-1)+1:10000*j);
    ntree = 40;
    Ytrain=y1{j};
    ytestd=ytest1{j};
    ytestf=ytest1{j};
    Xtraind=dept1g{j};
    Xtrainf=fnum1g{j};
    Xtestd=dept2g{j};
    Xtestf=fnum2g{j};
    forestd = fitForest(Xtraind',Ytrain,'randomFeatures',30,'bagSize',1/3,'ntrees',ntree);
    %forestf = fitForest(Xtrainf',Ytrain,'randomFeatures',20,'bagSize',1/3,'ntrees',ntree);
    %%^have to reimplement as 2 dimension input rather than cell
    yhatd{j} = predictForest(forestd,Xtestd');
    %yhatf{j} = predictForest(forestf,Xtestf');
    errord(j) = sum(((ytestd-yhatd{j})./ytestd).^2)/10000;
    %errorf(j) = sum(((ytestf-yhatf{j})./ytestf).^2)/10000;
end
errd=sum(errord(:))/10;
%errf=sum(errorf(:))/10;