package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.finalcode.craft;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Prediction;

public class FinalTestResult {

	public static void main(String[] args) {
		
		
		//best Model testing
		Prediction.prediction("SRMT-craft", "final-craft/trainset.csv", 200, "src/main/resources/final-craft/bestModel.bin", true); 
		Prediction.prediction("SRMT-craft", "final-craft/devset.csv", 20, "src/main/resources/final-craft/bestModel.bin", true); 
		Prediction.prediction("SRMT-craft", "final-craft/testset.csv", 20, "src/main/resources/final-craft/bestModel.bin", true); 
		
//		Prediction.prediction("SRMT-craft", "final-craft/trainset.csv", 200, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/devset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/testset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
		
	}
}
