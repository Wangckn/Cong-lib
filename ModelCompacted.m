clear all
close all 
clc
Layer=[];
NeuronNums=[784,64,32,32,10];
Layer = Initialization(Layer, NeuronNums, 'normrnd');
Layer(1).LayerNums=length(NeuronNums);
ActiveFunList={'Lrelu','Lrelu','Lrelu','Lrelu'};
for ii=1:length(NeuronNums)-1    
Layer(ii).ActiveFunName=ActiveFunList{ii};
end
Beta=1;
load('Mnisttrain60000test10000.mat');
Index = randperm(length(xtrain));
EpochNum=40;
BatchSize=100;
BatchNum=floor(length(xtrain)/BatchSize);
for ii=1:BatchNum
    FindOut=find(Index>=(ii-1)*BatchSize+1 & Index<=ii*BatchSize);
    Xtraincell{ii}=xtrain(FindOut,:);
    Ytraincell{ii}=ytrain(FindOut,:);
end

% ytrain=ytrain*2-1;
% ytest=ytest*2-1;
% ytrain=ytrain*2-1;

for Epochii=1:EpochNum
Epochii
    for Interationii=1:BatchNum
Layer(1).ytrain=Ytraincell{Interationii};
tempxtrain=Xtraincell{Interationii};
Layer=ForwardsUpdate(Layer,tempxtrain);
Layer=ErrorBackpropagation(Layer);
Layer= UpdateWeight(Layer, Beta);
    end 
Layer=LossFun(Layer);
Loss(Epochii)=Layer(1).Loss;
ACC(Epochii) = Acc(Layer,xtest,ytestlocation);
figure(1)
semilogy(Loss)
figure(2)
plot(ACC)
drawnow
end

function Layer = LossFun(Layer)
[tempa,tempb]=size(Layer(1).ytrain);
Layer(1).Loss=sum(sum((Layer(Layer(1).LayerNums).X-Layer(1).ytrain).^2))/tempa/tempb;
end

function Layer = Initialization(Layer, inputArg1, InitScheme)
switch InitScheme
    case 'random';
for ii = 1: length(inputArg1)-1
Layer(ii).weight=0.01*(rand(inputArg1(ii)+1, inputArg1(ii+1))-0.5)/(inputArg1(ii)+1+inputArg1(ii+1));
end

    case 'zeros';
for ii = 1: length(inputArg1)-1
Layer(ii).weight=zeros(inputArg1(ii)+1, inputArg1(ii+1));
end
    case 'normrnd';
for ii = 1: length(inputArg1)-1
Layer(ii).weight=normrnd(0,2/(inputArg1(ii)+1+inputArg1(ii+1)),[inputArg1(ii)+1, inputArg1(ii+1)]);
end
    otherwise
        warning(['聪哥警告: 无定义初始化类型: ',InitScheme])
end
end

function Layer = ForwardsUpdate(Layer, input)
Layer(1).X=input;
for ii=1:(Layer(1).LayerNums-1);
    tempX=Layer(ii).X;
    tempX(:, end+1)=1;
    Layer(ii).h=tempX*Layer(ii).weight;
    [Layer(ii).f,Layer(ii).fDS]= ActiveFun(Layer(ii), Layer(ii).h);
    Layer(ii+1).X=Layer(ii).f;
end
end


function Layer = ErrorBackpropagation(Layer)
[tempa,tempb]=size(Layer(1).ytrain);
Layer(Layer(1).LayerNums).ErrorX=(Layer(Layer(1).LayerNums-1).f-Layer(1).ytrain)/tempa/tempb;

for ii=(Layer(1).LayerNums-1) : -1:1
    Layer(ii).ErrorF=Layer(ii+1).ErrorX;
    Layer(ii).ErrorH=Layer(ii).ErrorF.*Layer(ii).fDS;
    tempErrorX=Layer(ii).ErrorH*Layer(ii).weight(1:end-1,:)';
     Layer(ii).ErrorX=tempErrorX;
end
end

function [output, outputDS] = ActiveFun(ActiveFunStruct,h)
switch ActiveFunStruct.ActiveFunName 
    case 'sigmoid';
[output, outputDS] = sigmoid(h);
    case 'tanh';
[output, outputDS] = tanh(h);
    case 'relu';
[output, outputDS] = relu(h);
    case 'Lrelu';
[output, outputDS] = Lrelu(h);
    otherwise
        warning(['聪哥警告: 无定义激活函数:', ActiveFunStruct.ActiveFunName])
end
end

function [output, outputDS]  = sigmoid(h)
output=1./(1+exp(-h));
outputDS=output.*(1-output)*8;
end

function [output, outputDS]  = tanh(h)
output=(exp(h)-exp(-h))./(exp(h)+exp(-h));
outputDS=1-output.^2;
end

function [output, outputDS] = relu(h)
output=max(h, 0);
outputDS=h>=0;
end

function [output, outputDS] = Lrelu(h)
output=max(h, 0.1*h);
tempa=h>=0;
tempb=h<0;
outputDS=tempa+0.1*tempb;
end

function [ACC] = Acc(tempLayer,xtest,ytestlocation)
tempLayer=ForwardsUpdate(tempLayer, xtest);
[tempvalue,templocation]=max(tempLayer(tempLayer(1).LayerNums).X');
counts=templocation==ytestlocation;
ACC=sum(counts)/length(counts);
end

function Layer= UpdateWeight(Layer, Beta)
for ii=(Layer(1).LayerNums-1) : -1:1
tempLayerX=Layer(ii).X;
tempLayerX(:, end+1)=1;
Layer(ii).ErrorW=tempLayerX'*Layer(ii).ErrorH; 
[tempa,tempb]=size(Layer(ii).ErrorW);
Layer(ii).weight=Layer(ii).weight-Beta*Layer(ii).ErrorW;
end
end
