/*
    Name: Devon Knudsen
    Student #: 10260170
    Date: 29 October 2020
    Assignment #: 2 - MNIST Handwritten Digit Recognizer Neural Network in Java
    Description: 3 layer neural network that was constructed to train on and recognize the MNIST handwritten digit set.
                 There are 784 nodes in the input layer, 30 in the hidden and 10 in the output.
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    private static Scanner input = new Scanner(System.in);

    public static void main(String[] args) throws IOException {
        // Figure out how large the data set is
        FileReader fileOpener1 = new FileReader("../mnist_train.csv");
        BufferedReader rowSizeAssesser = new BufferedReader(fileOpener1);
        int i = 0;
        while(rowSizeAssesser.readLine() != null){
            i++;
            if(i == 20000){
                break;
            }
        }
        fileOpener1.close();
        rowSizeAssesser.close();
        

        // Read in training data from CSV file
        FileReader fileOpener2 = new FileReader("../mnist_train.csv");
        BufferedReader dataCopier = new BufferedReader(fileOpener2);

        // iterate through lines of csv file and append them to an array to be shuffled
        double[][] trainingDataSet = new double[i][785];
        String row = "";
        String[] splitRow;
        i = 0;
        while((row = dataCopier.readLine()) != null){
            splitRow = row.split(",");
            trainingDataSet[i] = stringArrayToDoubleArray(splitRow, "-TRAIN");
            i++;
            if(i == 20000){
                break;
            }
        }
        fileOpener2.close();
        dataCopier.close();

        while(true){
            mainMenu(trainingDataSet);
        }
    }

    public static void mainMenu(double[][] trainingDataSet) throws IOException{
        System.out.println(
            "[1] Train the network\n" +
            "[2] Load a pre-trained network\n" +
            "[0] Exit" 
        );
        int selection = input.nextInt();
        input.nextLine();

        switch(selection){
            case 1:
                System.out.println("Initiating Network Training...");
                System.out.print("Input the amount of epochs: ");
                int epochs = input.nextInt();
                input.nextLine();

                // create a new network and begin stocastic gradient descent
                Network net = new Network("-TRAINING", null, null);
                net.stoGradientDescent(trainingDataSet, epochs, "-TRAIN");
                secondaryMenu(net, trainingDataSet);       
            
            case 2:
                System.out.print("Input path to file you would like to load: ");
                Scanner filePathScan = new Scanner(System.in);
                String filePath = filePathScan.nextLine();
               
                // iterate through lines of csv file
                FileReader fileOpener = new FileReader(filePath);
                BufferedReader dataCopier = new BufferedReader(fileOpener);
                double[][] inputWeights = new double[Network.HiddenSize][Network.InputSize];
                double[][] hiddenWeights = new double[Network.OutputSize][Network.HiddenSize];
                String currRow = "";
                String[] splitRow2;
                int i = 0;
                while((currRow = dataCopier.readLine()) != null){
                    splitRow2 = currRow.split(",");
                    if(i < inputWeights.length){
                        inputWeights[i] = stringArrayToDoubleArray(splitRow2, "-LOAD");
                    }
                    else{
                        hiddenWeights[i - inputWeights.length] = stringArrayToDoubleArray(splitRow2, "-LOAD"); 
                    }
                    
                    i++;
                }

                Network preloadNet = new Network("-LOAD", inputWeights, hiddenWeights);
                System.out.println("Weights Successfully Loaded!");
                secondaryMenu(preloadNet, trainingDataSet);
            
            case 0:
                System.out.println("Exiting Program...");
                System.exit(0);
        }
    }

    public static void secondaryMenu(Network currentNetwork, double[][] trainingDataSet) throws IOException{
        System.out.println(
            "[3] Display network accuracy on TRAINING data\n" +
            "[4] Display network accuracy on TESTING data\n" +
            "[5] Save the network state to file\n" +
            "[6] Go Back\n" +
            "[0] Exit" 
        );
        int selection = input.nextInt();
        input.nextLine();

        switch(selection){
            case 3:
                currentNetwork.stoGradientDescent(trainingDataSet, 1, "-DISPLAY");
                secondaryMenu(currentNetwork, trainingDataSet);          
            case 4:
                double[][] testingDataSet = null;
                if(testingDataSet == null){
                    // figure out how large the data set is
                    FileReader fileOpener1 = new FileReader("../mnist_test.csv");
                    BufferedReader rowSizeAssesser = new BufferedReader(fileOpener1);
                    int i = 0;
                    while(rowSizeAssesser.readLine() != null){
                        i++;
                    }
                    fileOpener1.close();
                    rowSizeAssesser.close();
                    
    
                    // read in testing data from CSV file
                    FileReader fileOpener2 = new FileReader("../mnist_test.csv");
                    BufferedReader dataCopier = new BufferedReader(fileOpener2);
    
                    // iterate through lines of csv file and append them to an array to be shuffled
                    testingDataSet = new double[i][785];
                    String row = "";
                    String[] splitRow;
                    i = 0;
                    while((row = dataCopier.readLine()) != null){
                        splitRow = row.split(",");
                        testingDataSet[i] = stringArrayToDoubleArray(splitRow, "-TEST");
                        i++;
                    }
                    fileOpener2.close();
                    dataCopier.close();
                }
 
                currentNetwork.stoGradientDescent(testingDataSet, 1, "-DISPLAY");
                secondaryMenu(currentNetwork, trainingDataSet);
            case 5:
                currentNetwork.saveCurrentWeights();
                System.out.println("Current Weights Saved!");
                secondaryMenu(currentNetwork, trainingDataSet);
            case 6:
                mainMenu(trainingDataSet);
            case 0:
                System.out.println("Exiting Program...");
                System.exit(0);
        }
    }

    // changes a string array into a double array
    private static double[] stringArrayToDoubleArray(String[] array, String mode){
        double[] doubleArr = new double[array.length];
        for(int i = 0; i < array.length; i++){
            if(mode == "-TRAIN" | mode == "-TEST"){
                doubleArr[i] = (double)(Integer.valueOf(array[i]));
            }
            else if(mode == "-LOAD"){
                doubleArr[i] = (double)(Double.parseDouble(array[i]));
            }
        }

        return doubleArr;
    }
}  