package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.md;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PredictDisintegrationTimeSimilarity {
	
	public static Logger log = LoggerFactory.getLogger(TrainningDisintegrationTime.class);
	
    public static final int numInputs = 18;
    public static final int testsetsize = 151;
	
	public static void main(String[] args) {
		
		  RecordReader recordReadertest = new CSVRecordReader(1,",");
	        try {
	        	recordReadertest.initialize(new FileSplit(new ClassPathResource("selecteddata.csv").getFile()));
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

	        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,testsetsize,numInputs,numInputs,true);
	        
	        DataSet alldata = iteratortest.next();
	  
//	        int[] index = {1,2,3};
//	        alldata.get(index);
//	        
	        RegressionEvaluation evalTest = new RegressionEvaluation(1);    
//	        
//	        INDArray featuresTest = alldata.getFeatureMatrix().get(new SpecifiedIndex(80,126,17,1,11,45,25,131,61,72,13,12,65,82,110), NDArrayIndex.all());
	        INDArray featuresTest = alldata.getFeatureMatrix();

	        
	        log.info("featuresTest" + featuresTest.shapeInfoToString());
	        log.info("\n" + featuresTest.toString());

	        INDArray lablesTest = alldata.getLabels();
	       
//	        INDArray ls = lablesTest.get(new SpecifiedIndex(80,126,17,1,11,45,25,131,61,72,13,12,65,82,110), NDArrayIndex.all());

     
	        
	        
//	        log.info("\n" + Arrays.toString(ls.shape()));
	        MultiLayerNetwork bestModel = null;
	        //   	bestModel = result.getBestModel();
	       //
	               try {
	               	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/best.bin"));
	       		} catch (IOException e) {
	       			// TODO Auto-generated catch block
	       			e.printStackTrace();
	       		}
	        
	        
//	        log.info(evalTest.stats());
	        INDArray PredictionTest = bestModel.output(featuresTest);
	        
	        log.info("test label value: \n" + lablesTest.toString());
	        log.info("test prediction value: \n" + PredictionTest.toString());

	        evalTest.eval(lablesTest, PredictionTest);	  
	        
	        log.info("testing set MSE is: " + String.format("%.10f", evalTest.meanSquaredError(0))); 
	        log.info("testing set R is: " + String.format("%.4f", evalTest.correlationR2(0)));
		
	}
}
