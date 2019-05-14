package my.rl.robot;

import java.io.BufferedReader; 
import java.io.FileReader; 
import java.io.FileWriter; 
import java.io.IOException; 
import java.io.Writer; 
import java.nio.file.Files; 
import java.nio.file.Paths; 
import java.util.List;

public class NeuralNetTest 
{ 
	double[][] input; 
	double [] output; 
	int state_action_len; 
	int num_patterns; 
	static String laptop_path = "C:\\Users\\Faizan Mansuri\\My Data\\";  
	public static Writer nn_stats; 
	
	public NeuralNetTest(int featureVectorSize, String file) throws IOException 
	{ 
		this.state_action_len = featureVectorSize; 
		this.num_patterns = getNumPatterns(file); 
		this.input = new double[this.num_patterns][this.state_action_len]; 
		this.output = new double[this.num_patterns]; 
		try 
		{ 
			loadTrainingSet(file); 
		} 
		catch (IOException e) 
		{ 
			e.printStackTrace(); 
		} 
	} 
	
	public int getNumPatterns(String file) throws IOException 
	{ 
		BufferedReader br = new BufferedReader(new FileReader(file)); 
		List<String> lines = Files.readAllLines(Paths.get(file)); 
		int num_patterns = lines.size(); 
		br.close(); 
		return num_patterns; 
	} 
	
	public void trainNet(int numInputs, int numHiddenNeurons, double learningRate, double momentum, boolean isOfflineTraining, String stats_file, String weights_file, boolean normalized) throws IOException 
	{ 
		NeuralNet nnBipolar = new NeuralNet(numInputs, numHiddenNeurons, learningRate, momentum, isOfflineTraining, normalized, weights_file);
		double rms = 10.0; 
		double totalError = 0; 
		int epochs = 0; 
		int same_rms = 0; 
		double rms_prev; 
		try
		{ 
			stats_file.replaceAll("\\\\", "/"); 
			System.out.println(stats_file); 
			nn_stats = new FileWriter(stats_file, true); 
			System.out.println("File created"); 
		} 
		catch (IOException e) 
		{ 
			e.printStackTrace(); 
		} 
		
		while (epochs < 4000) 
		{ 
			rms_prev = rms; 
			totalError = 0.0; 
			for (int i = 0; i < this.num_patterns; i++) 
			{ 
				totalError += nnBipolar.trainModified(this.input[i], this.output[i]); 
			} 
			rms = rootMeanSquare(totalError); 
			epochs++; 
			
			//System.out.println("totalError: "+totalError+" RMS: "+rms); 
			nn_stats.append(String.valueOf(totalError+" "+rms) + System.getProperty("line.separator")); 
			if (rms == rms_prev) 
			{ 
				same_rms ++; 
				if (same_rms == 500) break; 
			} 
		} 
		
		System.out.println("totalError: "+totalError+" RMS: "+rms +" Epochs: "+epochs); 
		nnBipolar.save(weights_file); 
		nn_stats.close(); 
	} 
	
	/** 
	 * * RMS is the prediction error/accuracy of the NN; i.e. output 
	 * * will be +/- RMS Value. In XOR case of RMS(0.05)=+/- 0.158 */ 
	public double rootMeanSquare(double totalErrorPerTrainingSet) 
	{ 
		double rms = Math.sqrt(2*totalErrorPerTrainingSet/this.num_patterns); 
		return rms; 
	} 
		
	public void loadTrainingSet(String file) throws IOException 
	{ 
		BufferedReader br = new BufferedReader(new FileReader(file));
		List<String> lines = Files.readAllLines(Paths.get(file)); 
		int pattern_ind = 0; 
		this.num_patterns = lines.size(); 
		for (String line : lines) 
		{ 
			String[] tokens = line.split(" "); 
			// set input data 
			for (int i=0; i<this.state_action_len; i++) 
			{ 
				this.input[pattern_ind][i] = Double.valueOf(tokens[i]); 
			} 
			// set output data 
			this.output[pattern_ind] = Double.valueOf(tokens[this.state_action_len]); 
			pattern_ind++; 
		} 
		br.close(); 
	} 
	
	public void printTrainingSet() 
	{ 
		int pattern_ind = 0; 
		String str = ""; 
		while (pattern_ind < this.num_patterns) 
		{ 
			for (int i = 0; i < this.state_action_len; i++) 
			{ 
				str += this.input[pattern_ind][i]+" "; 
			} 
			str += this.output[pattern_ind]; 
			System.out.println(str); 
			str = ""; 
			pattern_ind++; 
		} 
	} 
	
	public void printMaxAndMinQvalue() 
	{ 
		double max = this.output[0]; 
		int maxd = 0; 
		for (int i = 0; i < this.output.length; i++) 
		{ 
			if (this.output[i] > max) 
			{ 
				max = this.output[i]; 
				maxd = i; 
			} 
		} 
		System.out.println("max: "+max +" at "+maxd); 
		double min = this.output[0]; 
		int mind = 0; 
		for (int i = 0; i < this.output.length; i++) 
		{ 
			if (this.output[i] < min) 
			{ 
				min = this.output[i]; 
				mind = i; 
			} 
		}
		System.out.println("min: "+min+" at "+mind);
	}
	
	public static void main(String[] args) 
	{ 
		int numHiddenNeurons = 17; 
		double learningRate = 0.02; 
		double momentum = 0.3; 
		boolean isOfflineTraining = true; 
		boolean normalized = true; 
		NeuralNetTest nnt = null; 
		String lut_file = null; 
		int featureVectorSize; 
		
		if (normalized) 
		{ 
			lut_file = laptop_path +"normalized_lut_table.txt"; 
		} 
		
		else 
		{ 
			lut_file = laptop_path +"processed_lut_table.txt"; 
		} 
		
		String offlinestats_file = laptop_path+"trainingNN_stats.txt"; 
		String weights_file = laptop_path+"trained_weights.txt"; 
		featureVectorSize = 8;  // not including qvalue 
		
		try 
		{ 
			nnt = new NeuralNetTest(featureVectorSize, lut_file); 
		} 
		catch (IOException e) 
		{ 
			e.printStackTrace(); 
		} 
		
		//nnt.printTrainingSet(); 
		nnt.printMaxAndMinQvalue(); 
		try 
		{ 
			nnt.trainNet(featureVectorSize, numHiddenNeurons, learningRate, momentum, isOfflineTraining, offlinestats_file, weights_file, normalized); 
		} 
		catch (IOException e) 
		{ 
			e.printStackTrace(); 
		} 
	} 
}
