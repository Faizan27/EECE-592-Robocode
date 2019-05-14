package my.rl.robot;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader; 
import java.io.FileWriter; 
import java.io.IOException; 
import java.io.Writer; 
import java.nio.file.Files; 
import java.nio.file.Paths; 
import java.util.List;

public class NeuralNet implements NeuralNetInterface 
{ 
	public int numInputs; 
	public double[] weights; 
	public static final int INPUTSIZE = 8; 
	public double errorPerPattern; 
	protected int numHiddenNeurons; 
	protected Neuron[] hiddenNeurons; 
	protected Neuron outputNeuron; 
	protected double learningRate; 
	protected double momentum; 
	
	public static Writer trained_weights_wr; 
	public static Writer game_stats_wr; 
	public static Writer error_stats_wr; 
	static String laptop_path = "C:\\Users\\Faizan Mansuri\\My Data\\";  
	
	public NeuralNet (int numInputs, int numHiddenNeurons, double learningRate, double momentum, boolean isOfflineTraining, boolean normalizedSet, String weights_file) 
	{ 
		this.numInputs = numInputs; 
		this.numHiddenNeurons = numHiddenNeurons; 
		this.learningRate = learningRate; 
		this.errorPerPattern = 0; 
		hiddenNeurons = new Neuron[this.numHiddenNeurons];
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			hiddenNeurons[i] = new Neuron(this.numInputs, learningRate, momentum, normalizedSet); 
		} 
		outputNeuron = new Neuron(this.numHiddenNeurons, learningRate, momentum, normalizedSet); 
		if (isOfflineTraining) 
		{ 
			initializeWeights(); 
		} 
		else
		{ 
			// online training 
			try 
			{ 
				load(weights_file); 
			} 
			catch (IOException e) 
			{ 
				e.printStackTrace(); 
			} 
		} 
	}
	
	@Override 
	/** 
	 * * Computes the output of the NN without training (forward pass). 
	 * * @param X - the input vector, an array of doubles. 
	 * * @return Y - Sigmoid threshold output value. 
	 * * @throws IllegalArgumentException if the input vector length is incorrect */ 
	
	public double outputFor(double[] X) throws IllegalArgumentException 
	{
		if (X.length != INPUTSIZE) 
		{ 
			throw new IllegalArgumentException("input vector length is " + X.length + "; expected " + INPUTSIZE); 
		} 
		
		/* Perform forward propagation pass - compute weighted Sum Si and activation Ui for all cells */ 
		double[] hiddenRes = new double[this.numHiddenNeurons]; 
		double result; 
		double sum;
		
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			sum = hiddenNeurons[i].computeWeightedSum(X); 
			hiddenRes[i] = hiddenNeurons[i].applyActivation(sum); 
		} 
		
		if (hiddenRes[0] == 0.0 && hiddenRes[1] == 0.0 && hiddenRes[2] == 0.0 && hiddenRes[3] == 0.0) 
		{ 
			System.exit(1); 
		} 
		
		sum = outputNeuron.computeWeightedSum(hiddenRes); 
		result = outputNeuron.applyActivation(sum); 
		
		return result; 
	}
	
	
	public double trainModified(double[] X, double target) 
	{ 
		// 1. Perform forward propagation step and calculate error for the pattern 
		double nnActualOuput = outputFor(X);
		// 2. Perform backward propagation of error signals and compute weight udpates right away 
		double [] hiddenErrSignals = new double[this.numHiddenNeurons]; 
		double outputErrSignal = this.outputNeuron.applyGradientDescent()*(target - nnActualOuput); 
		
		double [] activatedInputsToOutput = new double[this.numHiddenNeurons]; 
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			activatedInputsToOutput[i] = this.hiddenNeurons[i].lastActivatedValue; 
		} 
		this.outputNeuron.updateWeights(outputErrSignal, activatedInputsToOutput); 
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			hiddenErrSignals[i] = hiddenNeurons[i].applyGradientDescent()*outputErrSignal*this.outputNeuron.weights[i]; 
			hiddenNeurons[i].updateWeights(hiddenErrSignals[i], X); 
		}
		/* Calculate total error for this patten and return */ 
		this.errorPerPattern = errorPerPattern(nnActualOuput, target); 
		return this.errorPerPattern; 
	} 
	
	@Override 
	/** 
	 * * This method is used to update the weights of the neural net 
	 * * @param X - input vector 
	 * * @param target - the target value for this input vector 
	 * * @return double - the error used to train (error before the update) */ 
	
	public double train(double[] X, double target) 
	{ 
		// 1. Perform forward propagation step and calculate error for the pattern 
		double nnActualOuput = outputFor(X);
		// 2. Perform backward propagation of error signals 
		double [] hiddenErrSignals = new double[this.numHiddenNeurons]; 
		double outputErrSignal = this.outputNeuron.applyGradientDescent()*(target - nnActualOuput); 
		
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			hiddenErrSignals[i] = hiddenNeurons[i].applyGradientDescent()*outputErrSignal*this.outputNeuron.weights[i]; 
		} 
		// 3. Update the weights for all neurons 
		double [] activatedInputsToOutput = new double[this.numHiddenNeurons]; 
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			activatedInputsToOutput[i] = this.hiddenNeurons[i].lastActivatedValue; 
		} 
		
		this.outputNeuron.updateWeights(outputErrSignal, activatedInputsToOutput); 
		for (int i = 0; i < this.numHiddenNeurons; i++) 
		{ 
			hiddenNeurons[i].updateWeights(hiddenErrSignals[i], X); 
		} 
		
		/* Calculate total error for this patten and return */ 
		this.errorPerPattern = errorPerPattern(nnActualOuput, target); 
		return this.errorPerPattern; 
	} 
	
	public void save(String path) 
	{ 
		double [] weights; 
		try { 
			path.replaceAll("\\\\", "/"); 
			System.out.println(path);
			trained_weights_wr = new FileWriter(path); 
			System.out.println("File created"); 
			
			// print hidden neuron weights 
			for (int i = 0; i<this.numHiddenNeurons; i++) 
			{ 
				weights = this.hiddenNeurons[i].getWeight(); 
				for (int j = 0; j < weights.length; j++) 
				{ 
					trained_weights_wr.append(String.valueOf(weights[j]+" ")); 
				} 
				trained_weights_wr.append(System.getProperty("line.separator")); 
			} 
			
			// print output neuron weights 
			weights = this.outputNeuron.getWeight(); 
			for (int j = 0; j < weights.length; j++) 
			{ 
				trained_weights_wr.append(String.valueOf(weights[j]+" ")); 
			} 
			trained_weights_wr.close(); 
		} 
		catch (IOException e) {
			// TODO Auto-generated catch block 
			e.printStackTrace(); 
		} 
	}
	
	@Override 
	public void load(String path) throws IOException 
	{ 
		System.out.print("Neural Net loading weights from file "+path); 
		BufferedReader br = new BufferedReader(new FileReader(path)); 
		int count = 0; 
		String [] tokens; 
		double [] weights = null; 
		List<String> lines = Files.readAllLines(Paths.get(path)); 
		for (String line : lines) 
		{ 
			tokens = line.split(" "); 
			weights = new double[tokens.length]; 
			for (int i=0; i<tokens.length; i++) 
			{ 
				weights[i] = Double.valueOf(tokens[i]); 
			} 
			if (count != this.numHiddenNeurons) 
			{ 
				this.hiddenNeurons[count].setWeight(weights); 
				count ++; 
			} 
			else 
			{ 
				this.outputNeuron.setWeight(weights); 
			} 
		} 
		br.close(); 
		checkWeights(); 
	}
	
	public void checkWeights() 
	{ 
		for (int i = 0; i<this.numHiddenNeurons; i++)
		{ 
			this.hiddenNeurons[i].printWeights(); 
		} 
		this.outputNeuron.printWeights(); 
	}
	
	 
	/** 
	 * * @return f(x) = 2 / (1+e(-x)) - 1 */ 
	public double bipolarSigmoid(double x) 
	{ 
		return ((2.0 / (1.0 + Math.exp(-x))) - 1.0); 
	} 
	
	public double binarySigmoid(double x) 
	{ 
		return (1.0 / (1.0 + Math.exp(-x))); 
	} 
	
	public double errorPerPattern(double actual, double target) 
	{ 
		double diff = actual - target; 
		return Math.pow(diff, 2) * 0.5; 
	}
	
	@Override 
	/** 
	 * * Initialize the weights to random values. 
	 * * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias. 
	 * * Like wise for hidden units. For say 2 hidden units which are stored in an array. 
	 * * [0] & [1] are the hidden & [2] the bias. 
	 * * We also initialize the last weight change arrays. This is to implement the alpha term. */ 
	
	public void initializeWeights() 
	{ 
		for (int i = 0; i < this.hiddenNeurons.length; i++) 
		{ 
			hiddenNeurons[i].initializeWeights(); 
		} 
		outputNeuron.initializeWeights(); 
	} 
	
	/** * 
	 * */ 
	public void saveGameStats(int numWinGamesPerHundred, double rewardsPerHundred) 
	{ 
		try 
		{
			String path = laptop_path+"robocodeNN_game_stats.txt"; 
			path.replaceAll("\\\\", "/"); 
			System.out.println(path); 
			game_stats_wr = new FileWriter(path, true); 
			System.out.println("File created"); 
			game_stats_wr.append(String.valueOf(numWinGamesPerHundred+" "+rewardsPerHundred) + System.getProperty("line.separator")); 
			game_stats_wr.close(); 
		} 
		catch (IOException e) { 
			// TODO Auto-generated catch block 
			e.printStackTrace(); 
		} 
	} 
	
	
	public void saveErrorSA(double error, int gameNumber) 
	{ 
		try { 
			String path = laptop_path+"robocodeNN_errorSA_stats.txt"; 
			path.replaceAll("\\\\", "/"); 
			System.out.println(path); 
			error_stats_wr = new FileWriter(path, true); 
			System.out.println("File created"); 
			error_stats_wr.append(String.valueOf(error) +" "+String.valueOf(gameNumber)+ System.getProperty("line.separator")); 
			error_stats_wr.close(); 
		} 
		catch (IOException e) 
		{ 
			// TODO Auto-generated catch block 
			e.printStackTrace(); 
		} 
	}

	@Override
	public void save(File argFile) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double sigmoid(double x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double customSigmoid(double x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void zeroWeights() {
		// TODO Auto-generated method stub
		
	} 
}
