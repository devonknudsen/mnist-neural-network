/*
    Name: Devon Knudsen
    Student #: 10260170
    Date: 29 October 2020
    Assignment #: 2 - MNIST Handwritten Digit Recognizer Neural Network in Java
    Description: 3 layer neural network that was constructed to train on and recognize the MNIST handwritten digit set.
                 There are 784 nodes in the input layer, 30 in the hidden and 10 in the output.
*/

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Network {
    private static int NumOfPixels = 28 * 28;
    private static int MiniBatchSize = 10;
    public static int InputSize = 784;
    public static int HiddenSize = 30;
    public static int OutputSize = 10;
    private static double LearningRate = 3.0;
    private double[][] hiddenBias = new double[HiddenSize][1];
    private double[][] outputBias = new double[OutputSize][1];
    private double[][] inputWeight = new double[HiddenSize][InputSize];
    private double[][] hiddenWeight = new double[OutputSize][HiddenSize];
    private static double[][] hiddenBiasGradients = new double[MiniBatchSize][HiddenSize];
    private static double[][] outputBiasGradients = new double[MiniBatchSize][OutputSize]; 
    private static double[][][] inputWeightGradients = new double[MiniBatchSize][HiddenSize][InputSize];
    private static double[][][] hiddenWeightGradients = new double[MiniBatchSize][OutputSize][HiddenSize];

    private static int[] classificationTotals = {0,0,0,0,0,0,0,0,0,0};
    private static int[] correctClassifcations = {0,0,0,0,0,0,0,0,0,0};
    
    public Network(String mode, double[][] inputWeight, double[][] hiddenWeight){
        if(mode == "-TRAINING"){
            for(int i = 0; i < HiddenSize; i++){
                this.inputWeight[i] = randomDoubleArrayFill(this.inputWeight[i]);
                this.hiddenBias[i] = randomDoubleArrayFill(this.hiddenBias[i]);
            }
            for(int i = 0; i < OutputSize; i++){
                this.hiddenWeight[i] = randomDoubleArrayFill(this.hiddenWeight[i]);
                this.outputBias[i] = randomDoubleArrayFill(this.outputBias[i]);
            }
        }
        else if(mode == "-LOAD"){
            this.inputWeight = inputWeight;
            this.hiddenWeight = hiddenWeight;
        }

    }

    // generates an array of doubles between a defined range
    private double[] randomDoubleArrayFill(double[] array){
        Random doubleGenerator = new Random();
        double min = -1.0;
        double max = 1.0;

        for(int i = 0; i < array.length; i++){
            // equation comes up with a random double between -1 and 1
            array[i] = min + (max - min) * doubleGenerator.nextDouble();
        }

        return array;
    }

    public void stoGradientDescent(double[][] dataSet, int epochs, String mode){

        // zero out classification totals for additional gradient descent calls
        for(int j = 0; j < classificationTotals.length; j++){
            classificationTotals[j] = 0;
        }

        for(int j = 0; j < epochs; j++){
            // Step 1: Randomize the order of the items in the training set
            Collections.shuffle(Arrays.asList(dataSet));
            
            // Step 2: Divide the training set into equal sized mini-batches
            double[][][] miniBatches = miniBatchGenerator(dataSet);

            for(int i = 0; i < miniBatches.length; i++){
                double[][] currentMiniBatch = miniBatches[i];
                for(int k = 0; k < currentMiniBatch.length; k++){
                    double[] currentLineOfPixels = currentMiniBatch[k];

                    // convert label -- correct classified value of the current pixel data -- into one hot (column) vector
                    int label = (int)currentLineOfPixels[0];
                    double[] oneHotVector = oneHotConversion(label, j);

                    // label removed from pixel data and all pixels are now scaled (and a column vector)
                    double[][] imagePixels = scalePixels(currentLineOfPixels);

                    // Step 3: Using backpropagation compute the weight gradients and bias gradients over the first [next] mini-batch
                    backPropogation(imagePixels, oneHotVector, k);
                }

                // Step 4: After completing the mini-batch update the weights and biases
                if(mode == "-TRAIN"){
                    changeWeights();
                    changeBiases();
                }
            }

            // Display results
            for(int n = 0; n < 10; n++){
                System.out.println(n + " = " + correctClassifcations[n] + "/" + classificationTotals[n]);
            }
            int totalCorrect = Arrays.stream(correctClassifcations).sum();
            System.out.println("Accuracy = " + totalCorrect + "/" + dataSet.length + " = " + ((double)totalCorrect/dataSet.length) * 100.0 + "%");

            // zero out correct classifications for next epoch
            for(int h = 0; h < correctClassifcations.length; h++){
                correctClassifcations[h] = 0;
            }
        }
    }

    private static double[] oneHotConversion(int classifiedValue, int currEpoch){
        double[] oneHotVector = new double[OutputSize];
        for(int i = 0; i < 10; i++){
            if(i != classifiedValue){
                oneHotVector[i] = 0.0;
            }
            else{
                oneHotVector[i] = 1.0;

                // count up classifications at the first epoch
                if(currEpoch == 0){
                    classificationTotals[i]++;
                }

            }  
        }

        return oneHotVector;
    }

    private static double[][][] miniBatchGenerator(double[][] dataSet){
        Random indexGenerator = new Random();
        int numOfMiniBatches = dataSet.length/MiniBatchSize;
        double[][][] miniBatches = new double[numOfMiniBatches][MiniBatchSize][NumOfPixels];
        int randomIndex = -1;
        ArrayList<Integer> takenIndexes = new ArrayList<Integer>();

        for(int i = 0; i < numOfMiniBatches; i++){
            for(int j = 0; j < MiniBatchSize; j++){
                randomIndex = indexGenerator.nextInt(dataSet.length);

                // if the data is taken, try again
                if(takenIndexes.contains(randomIndex)){
                    j--;
                }
                else{
                    miniBatches[i][j]= dataSet[randomIndex];
                    takenIndexes.add(randomIndex);
                }
            }
        }

        return miniBatches;
    }

    // removes label from pixel array data and divides each pixel by 255 to scale them to be fraction between 0 and 1
    private static double[][] scalePixels(double[] currentLineOfPixels){
        double[][] scaledPixels = new double[currentLineOfPixels.length - 1][1];
        for(int k = 1; k < currentLineOfPixels.length; k++){
            scaledPixels[k - 1][0] = currentLineOfPixels[k]/255;
        }

        return scaledPixels;
    }

    private void backPropogation(double[][] inputX, double[] inputY, int indexWithinMiniBatch){
        // Step 1: Using the current weights and biases [which are initially random] along with an input vector X, compute the activations (outputs) of all neurons at all layers of the network. This is the "feed forward" pass.
        double[][] hiddenLayerActivations = feedForward(inputX, 1, inputY);
        double[][] outputLayerActivations = feedForward(hiddenLayerActivations, 2, inputY);

        // // Step 2: Using the computed output of the final layer together with the desired output vector Y, Compute the gradient of the error at the final level of the network and then move "backwards" through the network computing the error at each level, one level at a time. This is the "backwards pass".
        backwardsPass(outputLayerActivations, 2, inputY, indexWithinMiniBatch);
        backwardsPass(hiddenLayerActivations, 1, inputY, indexWithinMiniBatch);
        backwardsPass(inputX, 0, inputY, indexWithinMiniBatch);
    }

    // feed forward step for back propogation -- calculates the activations for all layers
    private double[][] feedForward(double[][] previousLayerActivationsMatrix, int currentLayer, double[] oneHotVector){
        int rows1;
        int columns1;
        double[][] weightMatrix;

        // determine which weight matrix to use based on which layer you're currently on
        if(currentLayer == 1){
            rows1 = inputWeight.length;
            columns1 = inputWeight[0].length;
            weightMatrix = inputWeight;
        }
        else{
            rows1 = hiddenWeight.length;
            columns1 = hiddenWeight[0].length;
            weightMatrix = hiddenWeight;
        }

        int columns2 = previousLayerActivationsMatrix[0].length;
        
        // dot product of the weight matrix and activation matrix of the previous layer
        double[][] beforeAddingBias = new double[rows1][columns2];
        for (int i = 0; i < rows1; i++){
            for (int j = 0; j < columns2; j++){
                for (int k = 0; k < columns1; k++){
                    beforeAddingBias[i][j] += weightMatrix[i][k] * previousLayerActivationsMatrix[k][j];
                }
            }
        }

        // determine which bias matrix to use based on which layer you're currently on
        double[][] currentLayerBias;
        if(currentLayer == 1){
            currentLayerBias = hiddenBias;
        }
        else{
            currentLayerBias = outputBias;
        }
        
        // adding the bias matrix to dot product matrix
        double[][] currentLayerZ = new double[rows1][columns2];
        for(int p = 0; p < rows1; p++){
            currentLayerZ[p][0] = beforeAddingBias[p][0] + currentLayerBias[p][0];
        }

        // determine the activations using the sigma function
        double[][] currentLayerActivations = new double[currentLayerZ.length][currentLayerZ[0].length];
        double currentActivation;
        for(int i = 0; i < currentLayerZ.length; i++){
            currentActivation = 1/(1 + Math.exp((-1) * currentLayerZ[i][0]));
            currentLayerActivations[i][0] = currentActivation;
            
            if(currentLayer == 2){
                if(currentActivation > 0.5 & oneHotVector[i] == 1.0){
                    correctClassifcations[i]++;
                }
            }
        }

        return currentLayerActivations;

    }

    // back pass step for back propogation -- calculates the weight and bias gradients for all layers
    private void backwardsPass(double[][] layerActivations, int currentLayer, double[] oneHotVector, int indexWithinMiniBatch){
        for(int i = 0; i < layerActivations.length; i++){

            // if at the final layer -- calculate only the bias gradient
            if(currentLayer == 2){
                outputBiasGradients[indexWithinMiniBatch][i] = (layerActivations[i][0] - oneHotVector[i]) * layerActivations[i][0] * (1 - layerActivations[i][0]);
            }

            // if at the hidden layer -- calculate the weight and bias gradients
            else if(currentLayer == 1){
                double weightBiasSummation = 0.0;

                // compute the summation of weights times bias gradient of the current node by traversing column-wise
                for(int j = 0; j < hiddenWeight.length; j++){
                    weightBiasSummation += hiddenWeight[j][i] * outputBiasGradients[indexWithinMiniBatch][j];
                }

                hiddenBiasGradients[indexWithinMiniBatch][i] = weightBiasSummation * layerActivations[i][0] * (1 - layerActivations[i][0]);

                // weight gradient calculation: activation of current node * bias gradient of current node
                for(int k = 0; k < outputBiasGradients[indexWithinMiniBatch].length; k++){
                    hiddenWeightGradients[indexWithinMiniBatch][k][i] = layerActivations[i][0] * outputBiasGradients[indexWithinMiniBatch][k];
                }
            }

            // if at the first layer -- calculate only the weight gradient
            else{
                for(int m = 0; m < hiddenBiasGradients[indexWithinMiniBatch].length; m++){
                    inputWeightGradients[indexWithinMiniBatch][m][i] = layerActivations[i][0] * hiddenBiasGradients[indexWithinMiniBatch][m];
                }
            }
        }
    }

    // change the weights at both hidden and input layers after back propogation
    // both layers first for loop moves columns while the second loop goes down the columns
    private void changeWeights(){
        for(int j = 0; j < HiddenSize; j++){
            for(int i = 0; i < OutputSize; i++){
                double hiddenWeightGradientSummation = 0.0;

                // compute the summation of the hidden layer's weight gradients for the current node by traversing column-wise
                for(int m = 0; m < MiniBatchSize; m++){
                    hiddenWeightGradientSummation += hiddenWeightGradients[m][i][j];
                }

                hiddenWeight[i][j] = hiddenWeight[i][j] - (LearningRate/MiniBatchSize) * hiddenWeightGradientSummation;
            }
        }
        
        for(int k = 0; k < InputSize; k++){
            for(int n = 0; n < HiddenSize; n++){
                double inputWeightGradientSummation = 0.0;

                // compute the summation of the input layer's weight gradients for the current node by traversing column-wise
                for(int z = 0; z < MiniBatchSize; z++){
                    inputWeightGradientSummation += inputWeightGradients[z][n][k];
                }

                inputWeight[n][k] = inputWeight[n][k] - (LearningRate/MiniBatchSize) * inputWeightGradientSummation;
            }
        }
    }

    // change the biases at both output and hidden layers after back propogation
    private void changeBiases(){
        for(int i = 0; i < OutputSize; i++){
            double outputBiasGradientSummation = 0.0;

            // compute the summation of the output layer's bias gradients for the current node by traversing column-wise
            for(int k = 0; k < MiniBatchSize; k++){
                outputBiasGradientSummation += outputBiasGradients[k][i];
            }

            outputBias[i][0] = outputBias[i][0] - (LearningRate/MiniBatchSize) * outputBiasGradientSummation;
        }

        for(int j = 0; j < HiddenSize; j++){
            double hiddenBiasGradientSummation = 0.0;

            // compute the summation of the hidden layer's bias gradients for the current node by traversing column-wise
            for(int v = 0; v < MiniBatchSize; v++){
                hiddenBiasGradientSummation += hiddenBiasGradients[v][j];
            }

            hiddenBias[j][0] = hiddenBias[j][0] - (LearningRate/MiniBatchSize) * hiddenBiasGradientSummation;
        }
    }

    public void saveCurrentWeights() throws IOException{
        FileWriter fileWriter = new FileWriter("../savedWeightSet.csv");
        for(double[] weightRow : inputWeight){
            for(int i = 0; i < weightRow.length; i++){
                fileWriter.append(Double.valueOf(weightRow[i]).toString());
                if(i < weightRow.length - 1){
                    fileWriter.append(",");
                }
            }
            fileWriter.append("\n");
        }
        for(double[] weightRow : hiddenWeight){
            for(int i = 0; i < weightRow.length; i++){
                fileWriter.append(Double.valueOf(weightRow[i]).toString());
                if(i < weightRow.length - 1){
                    fileWriter.append(",");
                }
            }
            fileWriter.append("\n");
        }

        fileWriter.close();
    }
}