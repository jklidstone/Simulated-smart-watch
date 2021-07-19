
import processing.serial.*;
import grafica.*;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.lazy.IBk;
//import weka.core.converters.ConverterUtils.DataSource;

Serial myPort;  // Create object from Serial class

//Visluzation Related Variables
int locationPlotIMU_X = 100;// The position of the plot in the window
int locationPlotIMU_Y = 100;
int widthPlotIMU = 800;// size of the plot 
int heigthPlotIMU = 600;
int absRangeofY = 15;// The value range of the Y Axis in the plot
int numAxis = 6; 
int winSize = 50;// How many data points are saved in the data array and draw on the screen
GPlot plotIMU[] = new GPlot[numAxis];
long plotIMUIndex[] = new long[numAxis]; // Save the index for GPoints in GPlot 
String myString = null;
long lastTimeTriggered = 0;
String gesture = "";

//Weka ML related Variables
static public FastVector atts;
public static FastVector attsResult;
public Classifier myclassifier ;
BufferedWriter trainingfileWriter = null;
static Instances mInstances;  // Save the training instances
String[] classLabels= {"Up", "Down", "Left", "Punch", "Snap", "Tap"}; // The names of the class 
int numfeatures = 9;
double[] featurelist = new double[numfeatures+1];// The last one is for the lables
int numofTrainingSamples =60;
int samplecounter = 0;
String savingpath = "C:/Users/jklid/Desktop/";

//Save the data of the current window in multiple axis
ArrayList<ArrayList> IMUDataArray = new ArrayList();

void setup() {
  size(1200, 900);
  background(255);
  printArray(Serial.list());
  String portName = Serial.list()[1];
  myPort = new Serial(this, portName, 115200);// 
  
  // Initialize Plot Setting 
  plotInitialization();
  
  for(int i=0; i<numAxis; ++i){
   IMUDataArray.add(new ArrayList()); 
  }
  setupARFF(savingpath,classLabels);
}

void draw() {
  //background(red, green, blue);
  updataSerial();
  draw_plot();
  
}
void updataSerial() {
  while(myPort.available() > 0){
    myString = myPort.readStringUntil(10);    // '\n'(ASCII=10) every number end flag
    //print(myString);
    if(myString!=null){
      analysisData(myString);
    }
  }
}

void analysisData(String myString){
  String[] list = split(myString.substring(0, myString.length()-2), ',');
  if(list.length == 6){
    float[] imuValue = new float[numAxis]; // imuValue 0-6 : acclx, y, z, gyro x, y, z;
    for(int i = 0; i<numAxis; i++){
      imuValue[i] = Float.parseFloat(list[i]);
    }
    
    // If the size is more than the windowsize, remove a data before adding new values in
    while(IMUDataArray.get(0).size() >= winSize ){
       for (int i= 0; i < numAxis; ++i){
        // Maintain the lenght of the array, If the size of array is larger than winSize, remove the oldest data.
         IMUDataArray.get(i).remove(0);
         plotIMU[i].removePoint(0);
        }
    }
    
    // Add data into dataArray and PlotPoints at the same time
    for (int i= 0; i < numAxis; ++i){
     //System.out.println(IMUDataArray.get(0).size());
     plotIMU[i].addPoint(new GPoint(plotIMUIndex[i]++, imuValue[i]));
     IMUDataArray.get(i).add(imuValue[i]);
    }
    
    // Write your data processing algorithm here , miminum 1000 ms between two gestures
    long currenttime = millis();
    //If the gyroscope x [3] is larger than 2, then save the feature vector generated from the curent window
    //Gesture segmentation, detect when a gesture is happening
    
    //punch
    if(getABSMax(IMUDataArray.get(0))>4 && (currenttime-lastTimeTriggered) >3000){
      
      lastTimeTriggered = currenttime;
      featurelist = new double[numfeatures+1];
      featurelist[0] = getMean(IMUDataArray.get(2)); 
      featurelist[1] = getMax(IMUDataArray.get(2)); ; 
      featurelist[2] = getMin(IMUDataArray.get(2)); ;
      featurelist[3] = getMean(IMUDataArray.get(1)); 
      featurelist[4] = getMax(IMUDataArray.get(1)); ; 
      featurelist[5] = getMin(IMUDataArray.get(1)); ;
      featurelist[6] = getMean(IMUDataArray.get(3)); ;
      featurelist[7] = getMax(IMUDataArray.get(3)); ; 
      featurelist[8] = getMin(IMUDataArray.get(3)); ;
      featurelist[9] = 0.0; 

     try {
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        addinstance.setDataset(mInstances); // Specify the instance family this instance belongs to
        int resultindex = (int)myclassifier.classifyInstance(addinstance);
        System.out.println("Gesture Detected: "+classLabels[resultindex]);
        gesture = classLabels[resultindex];
     } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
     }
    }
    
    //gyro
    if(getABSMax(IMUDataArray.get(3))>2 && (currenttime-lastTimeTriggered) >3000){
      
      lastTimeTriggered = currenttime;
      featurelist = new double[numfeatures+1];
      featurelist[0] = 0.0; ;
      featurelist[1] = 0.0; ; 
      featurelist[2] = 0.0; ;
      featurelist[3] = 0.0; ;
      featurelist[4] = 0.0; ;
      featurelist[5] = 0.0; ;
      featurelist[6] = getMean(IMUDataArray.get(3)); ;
      featurelist[7] = getMax(IMUDataArray.get(3)); ; 
      featurelist[8] = getMin(IMUDataArray.get(3)); ;
      featurelist[9] = 0.0; 

     try {
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        addinstance.setDataset(mInstances); // Specify the instance family this instance belongs to
        int resultindex = (int)myclassifier.classifyInstance(addinstance);
        System.out.println("Gesture Detected: "+classLabels[resultindex]);
        gesture = classLabels[resultindex];
     } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
     }
    }
    
    if(getMax(IMUDataArray.get(1)) > 7 && (currenttime-lastTimeTriggered) >3000){
      
      lastTimeTriggered = currenttime;
      featurelist = new double[numfeatures+1];
      featurelist[0] = getMean(IMUDataArray.get(2)); 
      featurelist[1] = getMax(IMUDataArray.get(2)); ; 
      featurelist[2] = getMin(IMUDataArray.get(2)); ;
      featurelist[3] = getMean(IMUDataArray.get(1)); 
      featurelist[4] = getMax(IMUDataArray.get(1)); ; 
      featurelist[5] = getMin(IMUDataArray.get(1)); ;
      featurelist[6] = getMean(IMUDataArray.get(3)); ;
      featurelist[7] = getMax(IMUDataArray.get(3)); ; 
      featurelist[8] = getMin(IMUDataArray.get(3)); ;
      featurelist[9] = 0.0; 

     try {
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        addinstance.setDataset(mInstances); // Specify the instance family this instance belongs to
        int resultindex = (int)myclassifier.classifyInstance(addinstance);
        System.out.println("Gesture Detected: "+classLabels[resultindex]);
        gesture = classLabels[resultindex];
     } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
     }
    }
    
    //left swipe
    if(getMin(IMUDataArray.get(1))<-4 && (currenttime-lastTimeTriggered) >3000){
      
      lastTimeTriggered = currenttime;
      featurelist = new double[numfeatures+1];
      featurelist[0] = getMean(IMUDataArray.get(2)); 
      featurelist[1] = getMax(IMUDataArray.get(2)); ; 
      featurelist[2] = getMin(IMUDataArray.get(2)); ;
      featurelist[3] = getMean(IMUDataArray.get(1)); 
      featurelist[4] = getMax(IMUDataArray.get(1)); ; 
      featurelist[5] = getMin(IMUDataArray.get(1)); ;
      featurelist[6] = getMean(IMUDataArray.get(3)); ;
      featurelist[7] = getMax(IMUDataArray.get(3)); ; 
      featurelist[8] = getMin(IMUDataArray.get(3)); ;
      featurelist[9] = 0.0; 

     try {
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        addinstance.setDataset(mInstances); // Specify the instance family this instance belongs to
        int resultindex = (int)myclassifier.classifyInstance(addinstance);
        System.out.println("Gesture Detected: "+classLabels[resultindex]);
        gesture = classLabels[resultindex];
     } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
     }
    }
    
    //up down
    if(getABSMax(IMUDataArray.get(2))>14 && (currenttime-lastTimeTriggered) >3000){
      
      lastTimeTriggered = currenttime;
      featurelist = new double[numfeatures+1];
      featurelist[0] = getMean(IMUDataArray.get(2)); 
      featurelist[1] = getMax(IMUDataArray.get(2)); ; 
      featurelist[2] = getMin(IMUDataArray.get(2)); ;
      featurelist[3] = getMean(IMUDataArray.get(1)); 
      featurelist[4] = getMax(IMUDataArray.get(1)); ; 
      featurelist[5] = getMin(IMUDataArray.get(1)); ;
      featurelist[6] = getMean(IMUDataArray.get(3)); ;
      featurelist[7] = getMax(IMUDataArray.get(3)); ; 
      featurelist[8] = getMin(IMUDataArray.get(3)); ;
      featurelist[9] = 0.0; 

     try {
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        addinstance.setDataset(mInstances); // Specify the instance family this instance belongs to
        int resultindex = (int)myclassifier.classifyInstance(addinstance);
        System.out.println("Gesture Detected: "+classLabels[resultindex]);
        gesture = classLabels[resultindex];
     } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
     }
    }
       
  }
}

// Initialization data display
void plotInitialization(){
    // initialization of plotIMU 
    for (int i= 0; i < numAxis; ++i){
        plotIMU[i]= new GPlot(this);
        plotIMU[i].setPos(locationPlotIMU_X, locationPlotIMU_Y);
        plotIMU[i].setDim(widthPlotIMU, heigthPlotIMU);
    }
    plotIMU[0].setTitleText("IMU Data");
    plotIMU[0].getXAxis().setAxisLabelText("Time (t)");
    plotIMU[0].getYAxis().setAxisLabelText("y axis");
}

void draw_plot(){
  // cover the interface
    background(color(150));
    text(gesture, 50, 850);
    
    // plot background and axis 
    plotIMU[0].beginDraw();
    plotIMU[0].drawBackground();
    plotIMU[0].drawXAxis();
    plotIMU[0].drawYAxis();
    plotIMU[0].drawTitle();
    plotIMU[0].endDraw();
    
    // plot lines
    for (int i= 0; i < numAxis; ++i){   
      plotIMU[i].beginDraw();
      plotIMU[i].setXLim(plotIMUIndex[i]-winSize, plotIMUIndex[i]);
      plotIMU[i].setYLim(-absRangeofY, absRangeofY);
      plotIMU[i].drawLines();
      plotIMU[i].endDraw();
    }
  
}

/**
   * Set up Arff files for later retrieving data out from here
   * 
   * @param folder
   */
  private void setupARFF(String folder, String[] mylabels) {
    atts = new FastVector(); // Save the feature namse
    attsResult = new FastVector(); // Save the label names
    
    //Set up the folder , in case the folder dose not exist
    File writeFolder = new File(folder);
      if (!writeFolder.exists()) {
        writeFolder.mkdirs();
      }
    
    for (int i=0; i<mylabels.length;++i) {
      attsResult.addElement(mylabels[i]);
    }
    
    atts.add(new Attribute("Mean_AcclZ"));
    atts.add(new Attribute("Max_AcclZ"));
    atts.add(new Attribute("Min_AcclZ"));
    atts.add(new Attribute("Mean_AcclY"));
    atts.add(new Attribute("Max_AcclY"));
    atts.add(new Attribute("Min_AcclY"));
    atts.add(new Attribute("Mean_Gyro"));
    atts.add(new Attribute("Max_Gyro"));
    atts.add(new Attribute("Min_Gyro"));
    atts.add(new Attribute("result", attsResult));
    mInstances = new Instances("Gestures", atts, 0);

    try {
      //Load training file and train classifier     
      BufferedReader reader = new BufferedReader(new FileReader(savingpath+"2019_12_04_19_33_47_Training.arff"));
      mInstances  = new Instances(reader);
      reader.close();
      mInstances.setClassIndex(mInstances.numAttributes() - 1);
      // Use KNN with a K = 3 
      myclassifier = new IBk(3);
      myclassifier.buildClassifier(mInstances);
      
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
  

float getABSMax(ArrayList data){
  float max = Math.abs((float)data.get(0));
  for(int i=1;i<data.size(); ++i){
    if(max< Math.abs((float)data.get(i))) max = Math.abs((float)data.get(i));
  }
  return max; 
}

float getMax(ArrayList data){
  float max = (float)data.get(0);
  for(int i=1;i<data.size(); ++i){
    if(max< (float)data.get(i)) max = (float)data.get(i);
  }
  return max; 
}

float getMin(ArrayList data){
  float min = (float)data.get(0);
  for(int i=1;i<data.size(); ++i){
    if(min > (float)data.get(i)) min = (float)data.get(i);
  }
  return min; 
}

float getMean(ArrayList data){
  float total = (float)data.get(0);
  for(int i=1;i<data.size(); ++i){
    total += (float) data.get(i);
  }
  float mean = (float) total/((float)(data.size()));
  return mean; 
}

// Output the current date in String
String getCurrentTime() {
 
    //add year month day to the file name
    String fname= "";
    fname = fname + year() + "_";
    if (month() < 10) fname=fname+"0";
    fname = fname + month() + "_";
    if (day() < 10) fname = fname + "0";
    fname = fname + day();
    //add hour minute sec to the file name
    fname = fname + "_";
    if (hour() < 10) fname = fname + "0";
    fname = fname + hour() + "_";
    if (minute() < 10) fname = fname + "0";
    fname = fname + minute() + "_";
    if (second() < 10) fname = fname + "0";
    fname = fname + second();

    return fname;
}
