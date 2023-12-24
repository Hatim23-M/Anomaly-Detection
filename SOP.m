%{Training a GDN is a computationally intensive task. To make the example run quicker, 
% this example skips the training step and loads a pretrained model. To instead train the model, set the doTraining variable to true. %}

doTraining = false;

% Load the human activity data. The data contains the variable feat, which is a numTimeSteps-by-numChannels array containing the time series data.
load humanactivity

% View the number of time steps and number of channels in feat.
[numTimeSteps, numChannels] = size(feat);

% Randomly select and visualize four channels.
idx = randperm(numChannels,4);
figure
stackedplot(feat(:,idx),DisplayLabels="Channel " + idx);
xlabel("Time Step")

% Partition the data using the first 40% time steps for training.
numTimeStepsTrain = floor(0.4*numTimeSteps);
numTimeStepsValidation = floor(0.2*numTimeSteps);

featuresTrain = feat(1:numTimeStepsTrain,:);

% Normalize the training data.
[featuresTrain,muData,sigmaData] = normalize(featuresTrain);

windowSize = 10;

[XTrain,TTrain] = processData(featuresTrain,windowSize);

size(XTrain)

size(TTrain)

dsXTrain = arrayDatastore(XTrain,IterationDimension=3);
dsTTrain = arrayDatastore(TTrain,IterationDimension=2);
dsTrain = combine(dsXTrain,dsTTrain);

parameters = struct;

topKNum = 15;
embeddingDimension = 96;
numHiddenUnits = 1024;
inputSize = numChannels+1;

sz = [embeddingDimension inputSize];
mu = 0;
sigma = 0.01;
parameters.embed.weights = initializeGaussian(sz,mu,sigma);

sz = [embeddingDimension windowSize];
numOut = embeddingDimension;
numIn = windowSize;

parameters.graphattn.weights.linear = initializeGlorot(sz,numOut,numIn);
attentionValueWeights = initializeGlorot([2 numOut],1,2*numOut);
attentionEmbedWeights = initializeZeros([2 numOut]);
parameters.graphattn.weights.attention = cat(2,attentionEmbedWeights,attentionValueWeights);

sz = [numHiddenUnits embeddingDimension*numChannels];
numOut = numHiddenUnits;
numIn = embeddingDimension*numChannels;
parameters.fc1.weights = initializeGlorot(sz,numOut,numIn);
parameters.fc1.bias = initializeZeros([numOut,1]);

sz = [numChannels,numHiddenUnits];
numOut = numChannels;
numIn = numHiddenUnits;
parameters.fc2.weights = initializeGlorot(sz,numOut,numIn);
parameters.fc2.bias = initializeZeros([numOut,1]);

numEpochs = 80;
miniBatchSize = 200;
learnRate = 0.001;

mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    OutputAsDlarray=[1 0],...
    OutputEnvironment = ["auto" "cpu"]);

trailingAvg = [];
trailingAvgSq = [];

if doTraining
    numObservationsTrain = size(XTrain,3);
    numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
    numIterations = numIterationsPerEpoch*numEpochs;
    
    % Create a training progress monitor
    monitor = trainingProgressMonitor(...
        Metrics="Loss",...
        Info="Epoch",...
        XLabel="Iteration");
    
    epoch = 0;
    iteration = 0;

    % Loop over epochs
    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch+1;

        % Shuffle data
        shuffle(mbq)
            
        % Loop over mini-batches
        while hasdata(mbq) && ~monitor.Stop

            iteration = iteration+1;
    
            % Read mini-batches of data
            [X,T] = next(mbq);
        
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients] = dlfeval(@modelLoss,parameters,X,T,topKNum);
        
            % Update the network parameters using the Adam optimizer
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate);

            % Update training progress monitor
            recordMetrics(monitor,iteration,Loss=loss);
            updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
            monitor.Progress = 100*(iteration/numIterations);
        end
    end
else
    % Download and unzip the folder containing the pretrained parameters
    zipFile = matlab.internal.examples.downloadSupportFile("nnet","data/parametersHumanActivity_GDN.zip");
    dataFolder = fileparts(zipFile);
    unzip(zipFile,dataFolder);

    % Load the pretrained parameters
    load(fullfile(dataFolder,"parametersHumanActivity_GDN","parameters.mat"))
end

YTrain = modelPredictions(parameters,dsXTrain,topKNum);
scoreTrain = deviationScore(YTrain,TTrain,windowSize);

featuresValidation = feat(numTimeStepsTrain+(1:numTimeStepsValidation),:);


featuresValidation = normalize(featuresValidation,center=muData,scale=sigmaData);
[XValidation,TValidation] = processData(featuresValidation,windowSize);
dsXValidation = arrayDatastore(XValidation,IterationDimension=3);
YValidation = modelPredictions(parameters,dsXValidation,topKNum);
scoreValidation = deviationScore(YValidation,TValidation,windowSize);
threshold = max(scoreValidation);

featuresTest = feat(numTimeStepsTrain+numTimeStepsValidation+1:end,:);

featuresTest = normalize(featuresTest,center=muData,scale=sigmaData);
[XTest,TTest] = processData(featuresTest,windowSize);
dsXTest = arrayDatastore(XTest,IterationDimension=3);
YTest = modelPredictions(parameters,dsXTest,topKNum);
scoreTest = deviationScore(YTest,TTest,windowSize);

numObservationsTrain = numel(scoreTrain);
numObservationsValidation = numel(scoreValidation);
numObservationsTest = numel(scoreTest);
trainTimeIdx = windowSize+(1:numObservationsTrain);
validationTimeIdx = windowSize+trainTimeIdx(end)+(1:numObservationsValidation);
testTimeIdx = windowSize+validationTimeIdx(end)+(1:numObservationsTest);

figure
plot(...
    trainTimeIdx,scoreTrain,'b',...
    validationTimeIdx,scoreValidation,'g',...
    testTimeIdx,scoreTest,'r',...
    'linewidth',1.5)
hold on
yline(threshold,"k-",join(["Threshold = " threshold]),...
    LabelHorizontalAlignment="left");
hold off
xlabel("Time Step")
ylabel("Anomaly Score")
legend("Training","Validation","Test",Location="northwest")
grid on

featuresNew = feat(numTimeStepsTrain+numTimeStepsValidation+1:end,:);

featuresNewNormalized = normalize(featuresNew,center=muData,scale=sigmaData);
[XNew,TNew] = processData(featuresNewNormalized,windowSize);
dsXNew = arrayDatastore(XNew,IterationDimension=3);
YNew = modelPredictions(parameters,dsXNew,topKNum);
[scoreNew,channelMaxScores] = deviationScore(YNew,TNew,windowSize);

numObservationsNew = numel(scoreNew);
anomalyFraction = sum(scoreNew>threshold)/numObservationsNew;

anomalousChannels = channelMaxScores(scoreNew>threshold);
for i = 1:numChannels
    frequency(i) = sum(anomalousChannels==i);
end

figure
bar(frequency)
xlabel("Channel")
ylabel("Frequency")
title("Anomalous Node Count")

[~, channelHighestFrequency] = max(frequency)

figure
plot(featuresNew(:,channelHighestFrequency),'r')
xlabel("Time Step")
ylabel("Value")
title("Test Time Series Data - Channel " + num2str(channelHighestFrequency))

anomalousTimeSteps = find(scoreNew>threshold);
channelHighestFrequencyTimeSteps = anomalousTimeSteps(anomalousChannels==channelHighestFrequency);

figure
tiledlayout(2,1);
nexttile
plot(1:numObservationsNew,TNew(channelHighestFrequency,:),'r',...
    1:numObservationsNew,YNew(channelHighestFrequency,:),'g')
xlim([1 numObservationsNew])
legend("Targets","Predictions",Location="northwest")
xlabel("Time Step")
ylabel("Normalized Value")
title("Test Data: Channel " + num2str(channelHighestFrequency))
nexttile
plot(channelHighestFrequencyTimeSteps,TNew(channelHighestFrequency,channelHighestFrequencyTimeSteps),'xk')
xlim([1 numObservationsNew])
legend("Anomalous points",Location="northwest")
xlabel("Time Step")
ylabel("Normalized Value")

[~,idxMaxScore] = max(scoreNew);
channelHighestAnomalyScore = channelMaxScores(idxMaxScore)

timeHighestAnomalyScore = idxMaxScore;

figure
plot(featuresNew(:,channelHighestAnomalyScore),'r')
hold on
plot(timeHighestAnomalyScore,0,'s',MarkerSize=10,MarkerEdgeColor='g',MarkerFaceColor='g')
hold off
legend("","Highest anomaly point")
xlabel("Time Step")
ylabel("Value")
title("Time Series Data: Channel " + num2str(channelHighestAnomalyScore))

function [XData,TData] = processData(features, windowSize)
numObs = size(features,1) - windowSize;
XData = zeros(windowSize,size(features,2),numObs);
for startIndex = 1:numObs
    endIndex = (startIndex-1)+windowSize;
    XData(:,:,startIndex) = features(startIndex:endIndex,:);
end
TData = features(windowSize+1:end,:);
TData = permute(TData,[2 1]);
end

function [Y,attentionScores] = model(parameters,X,topKNum)
% Embedding
weights = parameters.embed.weights;
numNodes = size(weights,2) - 1;
embeddingOutput = embed(1:numNodes,weights,DataFormat="CU");

% Graph Structure
adjacency = graphStructure(embeddingOutput,topKNum,numNodes);

% Add self-loop to graph structure
adjacency = adjacency + eye(size(adjacency));

% Graph Attention
embeddingOutput = repmat(embeddingOutput,1,1,size(X,3));
weights = parameters.graphattn.weights;
[outputFeatures,attentionScores] = graphAttention(X,embeddingOutput,adjacency,weights);

% Relu
outputFeatures = relu(outputFeatures);

% Multiply
outputFeatures = embeddingOutput .* outputFeatures;

% Fully Connect
weights = parameters.fc1.weights;
bias = parameters.fc1.bias;
Y = fullyconnect(outputFeatures,weights,bias,DataFormat="UCB");

% Relu
Y = relu(Y);

% Fully Connect
weights = parameters.fc2.weights;
bias = parameters.fc2.bias;
Y = fullyconnect(Y,weights,bias,DataFormat="CB");
end

function [loss,gradients] = modelLoss(parameters,X,T,topKNum)
Y = model(parameters,X,topKNum);
loss = l2loss(Y,T,DataFormat="CB");
gradients = dlgradient(loss,parameters);
end

function Y = modelPredictions(parameters,ds,topKNum,minibatchSize)
arguments
    parameters
    ds
    topKNum
    minibatchSize = 500
end

ds.ReadSize = minibatchSize;
Y = [];

reset(ds)
while hasdata(ds)
    data = read(ds);
    data= cat(3,data{:});
    if canUseGPU
        X = gpuArray(dlarray(data));
    else
        X = dlarray(data);
    end
    miniBatchPred = model(parameters,X,topKNum);
    Y = cat(2,Y,miniBatchPred);
end
end

function adjacency = graphStructure(embedding,topKNum,numChannels)
% Similarity score
normY = sqrt(sum(embedding.*embedding));
normalizedY = embedding./normY;
score = embedding.' * normalizedY;

% Channel relations
adjacency = zeros(numChannels,numChannels);
for i = 1:numChannels
    topkInd = zeros(1,topKNum);
    scoreNodeI = score(i,:);
    % Make sure that channel i is not in its own candidate set
    scoreNodeI(i) = NaN;
    for j = 1:topKNum
        [~, ind] = max(scoreNodeI);
        topkInd(j) = ind;
        scoreNodeI(ind) = NaN;
    end
    adjacency(i,topkInd) = 1;
end
end

function [outputFeatures,attentionScore] = graphAttention(inputFeatures,embedding,adjacency,weights)
linearWeights = weights.linear;
attentionWeights = weights.attention;

% Compute linear transformation of input features
value = pagemtimes(linearWeights, inputFeatures);

% Concatenate linear transformation with channel embedding
gate = cat(1,embedding,value);

% Compute attention coefficients
query = pagemtimes(attentionWeights(1, :), gate);
key = pagemtimes(attentionWeights(2, :), gate);

attentionCoefficients = query + permute(key,[2, 1, 3]);
attentionCoefficients = leakyrelu(attentionCoefficients,0.2);

% Compute masked attention coefficients
mask = -10e9 * (1 - adjacency);
attentionCoefficients = (attentionCoefficients .* adjacency) + mask;

% Compute normalized masked attention coefficients
attentionScore = softmax(attentionCoefficients,DataFormat = "UCB");

% Normalize features using normalized masked attention coefficients
outputFeatures = pagemtimes(value, attentionScore);
end

function [smoothedScore,channel] = deviationScore(prediction,target,windowSize)
error = l1loss(prediction,target,DataFormat="CB",Reduction="none");
error = gather(double(extractdata(error)));

epsilon=0.01;
normalizedError = (error - median(error,2))./(abs(iqr(error,2)) + epsilon);
[scorePerTime,channel] = max(normalizedError);
smoothedScore = movmean(scorePerTime,windowSize);
end