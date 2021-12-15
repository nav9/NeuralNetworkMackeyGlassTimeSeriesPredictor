%This project was created for a college assignment.
%Due to IDE usability reasons, the variable names had to be kept short and un-intuitive.

clc;clear;close all;

%NNLayers has to have a minimum of two layers. Even if should be a single
%layer, the second layer here is an indicator of the number of outputs. The
%input to the first layer has to be a minimum of two values. The second
%value being a bias.
%NNLayers = [5 15 3 1];%number of nodes in each layer
NNLayers = [5 5 1];%number of nodes in each layer
NNLen = size(NNLayers, 2);
numTrainings = 5000;
load mgdata.dat;
td = 6;
readStartPos = 3 * td + 1;
%time = mgdata(1:round(size(mgdata, 1)/2), 1);
cuttingPoint = round(size(mgdata, 1)/2);
trainExemp = mgdata(1:cuttingPoint, 2);
testExemp = mgdata(cuttingPoint:size(mgdata,1), 2);
bias = 1;
activationFn = 'sigmoid';

disp('===== C R E A T E =====');
NN = [];%framework of nodes. The Neural Net
for i = 1:NNLen
    if i == NNLen, L = Layer(NNLayers(i), NNLayers(i), i, activationFn); 
    else L = Layer(NNLayers(i), NNLayers(i+1), i, activationFn); end
    NN = [NN L];
end
if length(NNLayers) > 2, 
    NN(length(NNLayers)-1).setAsSecondLastLayer();
    disp('Only linear combiner in final layer');
end
for i = 1:NNLen%create a doubly linked list
    if i == 1, NN(i).setNeighbours(NN(i), NN(i+1));
    else if i == NNLen, NN(i).setNeighbours(NN(i-1), NN(i));
        else NN(i).setNeighbours(NN(i-1), NN(i+1));end
    end
end

tic;
desired = zeros(size(numTrainings));
trainingOutput = desired;
trainingMSE = [];
disp('===== T R A I N =====');
for epo = 1:numTrainings
    if mod(epo,100)==0, fprintf('Epoch %d\n', epo);end
    for i = readStartPos:length(trainExemp),
        x = [trainExemp(i);trainExemp(i-td);trainExemp(i-2*td);trainExemp(i-3*td); bias];% Define Input Vector
        desired(i) = mgdata(i+6, 2);
        NN(NNLen).setDesiredOutput(desired(i));%set in last layer
        NN(1).Train(x);%invoke from first layer 
        trainingOutput(i) = NN(NNLen).getOutput();        
    end
    trainingMSE = [trainingMSE mean((desired-trainingOutput).^2)];
    NN(1).UpdateWeights();
    %NN(1).showWeights();
end  

fprintf('Minutes taken for training %d epochs = %f\n', numTrainings, toc/60);

disp('===== P R E D I C T =====');
outputStore = zeros(size(testExemp));
known = [];pred=[];
predError = [];
tic;
for i = readStartPos:length(testExemp)-td
    x = [testExemp(i);testExemp(i-td);testExemp(i-2*td);testExemp(i-3*td); bias];% Define Input Vector    
    NN(1).Predict(x);
    outputStore(i) = NN(NNLen).getOutput();%last layer stores output
    des = mgdata(cuttingPoint+i-1+td, 2);%desired value
    known = [known; des];pred = [pred; outputStore(i)]; 
    predError = [predError; (des-outputStore(i)).^2];
end
fprintf('Minutes taken for predicting %f\n', toc/60);
PredMSE = mean(predError);
fprintf('Prediction MSE for training of %d epochs = %f\n', numTrainings, PredMSE);
disp('Layers');disp(NNLayers);
disp('Number of epochs trained');disp(numTrainings);

figure(1);
xax = linspace(1,length(known),length(known));
plot(xax, known, '.b', xax, pred, 'or');
legend('ground truth', 'trained predictions');
xlabel('time');ylabel('white blood cell production');title('outputs');

figure(2);
plot(linspace(1,numTrainings,numTrainings), trainingMSE, 'b');
legend('Training MSE');
ylabel('MSE');xlabel('epochs');title('Mean Squared Error (MSE)');

figure(3);
plot(linspace(1,length(predError),length(predError)), predError, 'b');
legend('Prediction error');
ylabel('Error');xlabel('prediction samples');title('Prediction error');
