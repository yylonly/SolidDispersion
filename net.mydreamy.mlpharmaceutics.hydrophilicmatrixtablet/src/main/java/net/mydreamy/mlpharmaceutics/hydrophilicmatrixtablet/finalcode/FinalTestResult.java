package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.finalcode;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Prediction;

public class FinalTestResult {

	public static void main(String[] args) {
		
		
		//best Model testing
		Prediction.prediction("SRMT", "final-craft/trainset.csv", 200, "src/main/resources/final/bestModel.bin", false); 
		Prediction.prediction("SRMT", "final/testingset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		Prediction.prediction("SRMT", "final/extrascaledtestset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		
	}
}
