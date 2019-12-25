package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.mdfis;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class TrainningDisintegrationTime {

	public static Logger log = LoggerFactory.getLogger(TrainningDisintegrationTime.class);

	//Random number generator seed, for reproducability
    public static final int seed = 1234567890;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 200;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int trainsetSize = 200;
    public static final int testsetsize = 20;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    //with api properties
    public static final int numInputs = 18;
    public static final int numOutputs = 4;
    public static final int numHiddenNodes = 110;
    	
	public static void main(String[] args) {
		
		
//        CudaEnvironment.getInstance().getConfiguration()
//        // key option enabled
//        .allowMultiGPU(true)
//
//        // we're allowing larger memory caches
//        .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
//
//        // cross-device access is used for faster model averaging over pcie
 //       .allowCrossDeviceAccess(true);
		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        String delimiter = ",";
        RecordReader recordReadertrain = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertrain.initialize(new FileSplit(new ClassPathResource("mdfis/trainingset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortrain = new RecordReaderDataSetIterator(recordReadertrain,trainsetSize,numInputs,numInputs+3,true);
   

        
        RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest.initialize(new FileSplit(new ClassPathResource("mdfis/testingset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,testsetsize,numInputs,numInputs+3,true);

 //       log.info("testData set:" + testData.toString());
        
//        // Normalization
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
   
        DataSet trainningData = iteratortrain.next();
        DataSet testData = iteratortest.next();

        
 //       log.info(trainningData.toString());
 //       log.info(testData.toString());
        
//        normalizer.fitLabel(false);
//        normalizer.fit(trainningData); 
//        normalizer.transform(trainningData); 
//        normalizer.transform(testData); 
//        
        iteratortrain.reset();
        iteratortest.reset();
//        
//       log.info(trainningData.toString());
//        log.info(testData.toString());
//        log.info("training data features:\n" + trainningData.getFeatureMatrix().toString());
//        log.info("training data label:\n" + trainningData.getLabels().toString());
//        normalizer.transform(testData); 
//        log.info("training data features:\n" + testData.getFeatureMatrix().toString());
//        log.info("training data label:\n" + testData.getLabels().toString());
        
        // Network Configuration
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.RELU)
                .regularization(true)
                .l2(1e-3)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
              //  .dropOut(0.5)
                .updater(Updater.NESTEROVS).momentum(0.9)
              //  .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(5, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(6, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(7, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(8, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation("sigmoid")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(1000));
        
        
        List<EpochTerminationCondition> terminationconditions = new LinkedList<EpochTerminationCondition>();
      //  terminationconditions.add(new ScoreImprovementEpochTerminationCondition(100, 1E-10));
        terminationconditions.add(new BestScoreEpochTerminationCondition(0.01));
        terminationconditions.add(new MaxEpochsTerminationCondition(1100));

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        		.epochTerminationConditions(terminationconditions)
        		.scoreCalculator(new DataSetLossCalculator(iteratortest, true))
                .evaluateEveryNEpochs(100)
                .saveLastModel(true)
        		.modelSaver(new LocalFileModelSaver("src/main/resources"))
        		.build();
        
        
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iteratortrain);

        //Conduct early stopping training:
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

       

       //Print out the results:
        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
        
        
        MultiLayerNetwork bestModel = null;
 //   	bestModel = result.getBestModel();
//
        try {
        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/bestModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        MultiLayerNetwork latestModel = null;
    //	bestModel = result.getBestModel();

        try {
        	latestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/latestModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        //Get the best model:
       //  bestModel =  result.getBestModel();
        
//        //Train the network on the full data set, and evaluate in periodically
//        for( int i=0; i<nEpochs; i++ ){
//            net.fit(iteratortrain);
//            iteratortrain.reset();
//        }
        
//        iteratortrain.reset();
 //       DataSet trainningData = iteratortrain.next();
        
 //       iteratortest.reset();
  //      DataSet testData = iteratortest.next();
        
        log.info("========================== testing =========================");
        log.info("========================== latest model =========================");
        //test on latest model
        testOnDiffModel(latestModel, trainningData, testData);
        
        log.info("========================== best model =========================");
        //test on best model
        testOnDiffModel(bestModel, trainningData, testData);
 
	}

	public static void testOnDiffModel(MultiLayerNetwork net, DataSet trainningData,  DataSet testData)
	{
	       // evaluation training set
        RegressionEvaluation evalTrain = new RegressionEvaluation(4);
        
        INDArray featuresTrain = trainningData.getFeatureMatrix();
        INDArray lablesTrain = trainningData.getLabels();
        
        INDArray PredictionTrain = net.output(featuresTrain);
     //   log.info("train label set:\n" + lablesTrain.toString());
     //   log.info("train prediction set:\n" + PredictionTrain.toString());
        evalTrain.eval(lablesTrain, PredictionTrain);	  
        
    //    log.info("training set MSE is:" + String.format("%.10f", (evalTrain.meanSquaredError(0)+evalTrain.meanSquaredError(1)+evalTrain.meanSquaredError(2)+evalTrain.meanSquaredError(3))/4));
        
        double AverTR = 0;
        double AverTMAE = 0;

        for (int i = 0; i < 4; i++) {
        	AverTR += evalTrain.correlationR2(i);
        	AverTMAE += evalTrain.meanAbsoluteError(i);
        }
        
        log.info("training set R is:" + String.format("%.4f", AverTR/4));
        log.info("training set MAE is: " + String.format("%.4f",  AverTMAE/4));
        
//        Evaluation.f2(lablesTrain, PredictionTrain);
//        Evaluation.AccuracyMAE(lablesTrain, PredictionTrain, 0.10);
//        Evaluation.AccuracyMAE(lablesTrain, PredictionTrain, 0.12);
       


        
        // evluation test set
        RegressionEvaluation evalTest = new RegressionEvaluation(4);
        
        INDArray featuresTest = testData.getFeatureMatrix();
    //    log.info("featuresTest" + featuresTest.shapeInfoToString());
    //    log.info("\n" + featuresTest.toString());

        INDArray lablesTest = testData.getLabels();
        
        
//        log.info(evalTest.stats());
        INDArray PredictionTest = net.output(featuresTest);
        
        log.info("test label value: \n" + lablesTest.toString());
        log.info("test prediction value: \n" + PredictionTest.toString());

        evalTest.eval(lablesTest, PredictionTest);	  
        
        double AverTestR = 0;
        double AverMAE = 0;
        for (int i = 0; i < 4; i++) {
        	AverTestR += evalTest.correlationR2(i);
        	AverMAE += evalTest.meanAbsoluteError(i);
        }
        
    //    log.info("testing set MSE is: " + String.format("%.10f", (evalTest.meanSquaredError(0)+evalTest.meanSquaredError(1)+evalTest.meanSquaredError(2)+evalTest.meanSquaredError(3))/4)); 
        log.info("testing set R is: " + String.format("%.4f",  AverTestR/4));
        log.info("testing set MAE is: " + String.format("%.4f",  AverMAE/4));
        
//        Evaluation.f2(lablesTest, PredictionTest);
//        Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.10);
//        Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.12);



	}
}
