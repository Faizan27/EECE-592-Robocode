package my.rl.robot;

import java.util.Random;
public class Neuron 
{ 
	double [] weights; /* neurons only concerned with input weights */ 
	double [] previousWeights; 
	int numWeights; 
	boolean isBinary; /* if false, use bipolar sigmoid */ 
	double lastActivatedValue = 0.0; 
	double lastWeightedSum = 0.0; 
	double learningRate; 
	double momentum; 
	boolean normalized; 
	final double bias = 1.0;
	
	public Neuron (int numWeights, double learningRate, double momentum, boolean normalized) 
	{ 
		this.numWeights = numWeights + 1; // each neuron has a bias input weight 
		this.weights = new double[this.numWeights]; 
		this.previousWeights = new double[this.numWeights]; 
		this.learningRate = learningRate; 
		this.momentum = momentum; 
		this.normalized = normalized; 
	} 
	
	public void setWeight(double[] weights) 
	{ 
		assert(weights.length == this.numWeights); 
		System.arraycopy(weights, 0, this.weights, 0, this.numWeights); 
		zeroWeights(); 
	} 
	
	public double[] getWeight() 
	{ 
		return this.weights; 
	} 
	
	/** 
	 * * Initializes weights to a random value between [-0.5, 0.5] */ 
	public void initializeWeights() 
	{ 
		Random number = new Random(); 
		double min = -0.5; 
		double max = 0.5; 
		for (int i = 0; i < this.numWeights; i++) 
		{ 
			this.weights[i] = min + (max - min) * number.nextDouble(); 
			assert(this.weights[i] <= 0.5 && this.weights[i] >= -0.5); 
		} 
		zeroWeights(); 	/* Init prev weights to zero */ 
	} 
	
	public void zeroWeights() 
	{ 
		for (int i = 0; i < this.numWeights; i++) 
		{ 
			this.previousWeights[i] = 0.0; 
		} 
	} 
	
	/** 
	 * * @param double[] X - input vector to neuron 
	 * * @return double S = W*X for the one neuron */ 
	
	public double computeWeightedSum(double[] inputs) 
	{ 
		double result = 0.0; 
		assert(inputs.length == this.numWeights - 1); // minus 1 for bias 
		for (int i = 0; i < inputs.length; i++) 
		{ 
			result += this.weights[i] * inputs[i]; 
		} 
		result += this.weights[this.numWeights - 1] * bias; 
		this.lastWeightedSum = result;
		return result; 
	} 
	
	public void printWeights()
	{ 
		System.out.println("printing weights!"); 
		for (int i=0; i < this.weights.length; i++) 
		{ 
			System.out.print(this.weights[i] + " "); 
			assert(!Double.isNaN(this.weights[i])); 
		} 
	} 
	
	/** 
	 * * @param errorSignal - for the neuron 
	 * * @param inputs[] - the inputs to the neuron from the layer below */ 
	public void updateWeights(double errorSignal, double[] inputs) 
	{ 
		double [] tempWeights = new double[this.numWeights]; 
		System.arraycopy(this.weights, 0, tempWeights, 0, this.numWeights); 
		for (int i = 0; i < this.numWeights - 1; i++) 
		{ 
			this.weights[i] = this.weights[i] + this.learningRate * errorSignal * inputs[i] + this.momentum*(this.weights[i] - this.previousWeights[i]); 
		} 
		this.weights[this.numWeights - 1] = (this.weights[this.numWeights - 1] + this.learningRate * errorSignal * bias + this.momentum*(this.weights[this.numWeights - 1] - this.previousWeights[this.numWeights - 1])); 
		this.previousWeights = tempWeights; 
	} 
	
	public double applyActivation(double weightedSum) 
	{
		if (this.normalized) 
		{ 
			this.lastActivatedValue = bipolarSigmoid(weightedSum); 
		} 
		else 
		{ 
			this.lastActivatedValue = customSigmoid(weightedSum); 
		} 
		return this.lastActivatedValue; 
	} 
	
	public double applyGradientDescent() 
	{
		double result = 0.0; 
		if (normalized) 
		{ 
			result = bipolarSigmoidPrime(this.lastActivatedValue); 
		} 
		else 
		{ 
			result = customSigmoidPrime(this.lastWeightedSum); 
		} 
		return result; 
	} 
	
	/** * Return a binary sigmoid 
	 * * @param x The input weighted sum of neuron 
	 * * @return f(x) = 1 / (1 + e(-x)) */ 
	public double binarySigmoid(double x) 
	{ 
		return (1.0 / (1.0 + Math.exp(-x))); 
	} 
	
	/** 
	 * * Return a derivative of binary sigmoid 
	 * * @param x - Ui (output activated value of cell) 
	 * * @return f'(Si) = Ui(1-Ui) */ 
	public double binarySigmoidPrime(double x) 
	{ 
		return (x * (1.0 - x)); 
	} 
	
	//(6/(1+e^-x)) - 4.45 
	public double customSigmoid (double x)
	{ 
		return ((21.0 / (1.0 + Math.exp(-x))) - 19.0); 
	} 
	public double customSigmoidPrime(double x) 
	{ 
		return (12.0*Math.exp(-x) / Math.pow(1.0 + Math.exp(-x), 2)); 
	} 
	
	/** * Return a bipolar sigmoid 
	 * * @param x The input weighted sum of neuron 
	 * * @return f(x) = (2 / (1 + e(-x))) - 1 */ 
	public double bipolarSigmoid(double x) 
	{ 
		return ((2.0 / (1.0 + Math.exp(-x))) - 1.0); 
	} 
	
	public double bipolarSigmoidPrime(double x) 
	{ 
		return (0.5)*(1.0 + x)*(1.0 - x); 
	} 
}
		