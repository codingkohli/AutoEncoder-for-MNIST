%loading the images
imgFile = 't10k-images-idx3-ubyte';
labelFile = 't10k-labels-idx1-ubyte';
[imgs labels] = readMNIST(imgFile,labelFile,10000,0);


for i =1:20
	subplot(4,5,i);
	imshow(imgs(:,:,i));
end

%% resizing the labels
tTrain = zeros(10,10000);
for i = 1:10000
	temp = [0 0 0 0 0 0 0 0 0 0];
	if labels(i) == 0 
		labels(i) = 10;
	end
	temp(labels(i)) = 1;
	tTrain(:,i) = temp;
	%tTrain{i} = temp;
end

%traing the images
xTrainImages = {};
for i = 1:10000
	xTrainImages{i} = imgs(:,:,i);
end

%setting the weights of a nueral net to default instead of random
rng('default')

%training in the first layer
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1,'MaxEpochs',400,'L2WeightRegularization',0.004,'SparsityRegularization',4,'SparsityProportion',0.15,'ScaleData',false);

%viewing the autoenc1
%view(autoenc1)

%{plotting the weights 
figure()
plotWeights(autoenc1);
%}

%getting the features from the first encoder
feat1 = encode(autoenc1,xTrainImages);


% training the 2nd autoencoder
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',100,'L2WeightRegularization',0.002,'SparsityRegularization',4,'SparsityProportion',0.1,'ScaleData',false);

%viewing the autoenc2
%view(autoenc2)

%getting the features of the second layer 
feat2 = encode(autoenc2,feat1);

%training the final softmax layer
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);

%viewing the softmax layer
view(softnet)

%stacking to form a deepnet
deepnet = stack(autoenc1,autoenc2,softnet);

%viewing the deepnet
view(deepnet)
