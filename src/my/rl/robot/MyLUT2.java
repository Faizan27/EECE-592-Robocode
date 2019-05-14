package my.rl.robot;


import java.util.Random;
import java.io.BufferedReader; 
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
//import java.io.FileWriter; 
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
//import java.io.Writer; 
import java.nio.file.Files; 
import java.nio.file.Paths; 
import java.util.List; 

public class MyLUT2{ 
	// static declarations 
	public double[][][][][][] qTable; 
	public static File game_stats_wr; 
	public static File trained_lut_wr; 
	int lutSize = 1; 
	int featureVectorSize; 
	int [] featureNumValues; 
	public boolean trainingMode; 
	String path1 = "C:/Users/Faizan Mansuri/My Data/robocode_game_stats_onpolicy.txt";
	static String path = "C:/Users/Faizan Mansuri/My Data/trained_lut.txt";
	
	public MyLUT2(boolean initializeflag, boolean trainingMode,int featureVectorSize,int [] featureNumValues) 
	{ 
		// determine size of lut 
		this.featureVectorSize = featureVectorSize; 
		this.featureNumValues = featureNumValues; 
		
		for (int i = 0; i < featureVectorSize; i++) { 
			lutSize = lutSize * featureNumValues[i]; 
			System.out.println(featureNumValues[i]); 
		} 
		
		qTable = new double[featureNumValues[5]][featureNumValues[4]][featureNumValues[3]][featureNumValues[2]][featureNumValues[1]][featureNumValues[0]]; 
		this.trainingMode = trainingMode; 
		//System.out.println("I was called");
		//if(initializeflag)
		initialiseLUT(this.trainingMode,initializeflag);
	} 
	
	/** * Initialize the look up table to all zeros. */ 
	public void initialiseLUT(boolean trainingMode, boolean initialflag) 
	{ 
		Random number = new Random(); 
		double min = 0.5; 
		double max = 1; 
		if (trainingMode && initialflag) 
		{ 
			for (int i = 0; i < featureNumValues[5]; i++) 
			{ // 5
				for (int j = 0; j < featureNumValues[4]; j++) 
				{ // 4 
					for (int k = 0; k < featureNumValues[3]; k++) 
					{// 3 
						for (int l = 0; l < featureNumValues[2]; l++) 
						{ // 2 
							for (int m = 0; m < featureNumValues[1]; m++) 
							{ // 1 
								for (int n = 0; n < featureNumValues[0]; n++) 
								{ //0 
									qTable[i][j][k][l][m][n] = min + (max - min) * number.nextDouble(); 
									//System.out.println(qTable[i][j][k][l][m][n]);
								} 
							} 
						} 
					} 
				} 
			} 
		} 
		else 
		{ 
			try { 
				load(path); 
			} 
			catch (IOException e) 
			{ // TODO Auto-generated catch block 
				e.printStackTrace(); 
			} 
		} 
	} 
	
	/** * A helper method that translates a vector being used to index the look up table 
	 * * into an ordinal that can then be used to access the associated look up table element. 
	 * * @param X The state action vector used to index the LUT 
	 * * @return The index where this vector maps to - the first row actions for this feature vector 
	 * */ 
	public int indexFor(double [] X) 
	{ // unused in my implementation of qTable 
		return 0; 
	} 
	
	/** * @param X The input vector. An array of doubles. 
	 * * @return The value returned by the LUT or NN for this input vector 
	 * */ 
	public double outputFor(int [] state) 
	{ 
		assert(state.length == this.featureVectorSize); 
		return qTable[state[5]][state[4]][state[3]][state[2]][state[1]][state[0]]; 
	} 
	
	/** 
	 * * Given a state vector, return the actionSet which is the corresponding Q values for that feature vector 
	 * * @param state
	 * * @return 
	 * */ 
	public double [] outputQvaluesForCurrentState( int [] state) 
	{ 
		assert(state.length == this.featureVectorSize - 1); 
		System.out.println(state[4]+" "+state[3]+" "+state[2]+" "+state[1]+" "+state[0]); 
		assert(state[4] < featureNumValues[5]); 
		assert(state[3] < featureNumValues[4]); 
		assert(state[2] < featureNumValues[3]); 
		assert(state[1] < featureNumValues[2]); 
		assert(state[0] < featureNumValues[1]); 
		double [] actionSet = qTable[state[4]][state[3]][state[2]][state[1]][state[0]]; 
		return actionSet; 
	} 
	
	/** 
	 * * This method will tell the NN or the LUT the output 
	 * * value that should be mapped to the given input vector. I.e. 
	 * * the desired correct output value for an input. 
	 * * @param X The input vector 
	 * * @param argValue The new value to learn 
	 * * @return The error in the output for that input vector */ 
	
	public double train(int [] state, double argValue) 
	{ 
		assert(state.length == this.featureVectorSize); 
		qTable[state[5]][state[4]][state[3]][state[2]][state[1]][state[0]] = argValue; 
		return 0; 
	} 
	
	/** * */ 
	public void saveGameStats(int numWinGamesPerHundred, double rewardsPerHundred) 
	{ 	
		PrintStream saveFile = null;
		
		try {
			saveFile = new PrintStream( new FileOutputStream( path1,true ));
		}
		catch (IOException e) {
			System.out.println( "*** Could not create output stream for Reward save file.");
		}
	
		//saveFile.append(String.valueOf(numWinGamesPerHundred+" "+rewardsPerHundred) + System.getProperty("line.separator"));  
		saveFile.append(String.valueOf(numWinGamesPerHundred) + System.getProperty("line.separator"));
		saveFile.close();
	} 
	
	/** 
	 * * A method to write either a LUT or weights of an neural net to a file. 
	 * * @param argFile of type File. */ 
	
	public void save()
	{ 	
		PrintStream saveFile = null;
		
		try {
			trained_lut_wr = new File(path); 
			trained_lut_wr.createNewFile();
			saveFile = new PrintStream( new FileOutputStream( trained_lut_wr ));
		}
		catch (IOException e) {
			System.out.println( "*** Could not create output stream for LUT save file.");
		}
		
			for (int i = 0; i < featureNumValues[5]; i++) 
			{ // 5 
				for (int j = 0; j < featureNumValues[4]; j++) 
				{ // 4 
					for (int k = 0; k < featureNumValues[3]; k++) 
					{ // 3 
						for (int l = 0; l < featureNumValues[2]; l++)
						{ // 2 
							for (int m = 0; m < featureNumValues[1]; m++)
							{ // 1 
								for (int n = 0; n < featureNumValues[0]; n++) 
								{ //0 
									saveFile.print(String.valueOf(qTable[i][j][k][l][m][n]) + System.getProperty("line.separator")); 
								}
							}
						}
					}
				}
			}
			
		 
		saveFile.close();
	} 
	

	/** 
	 * * Loads the LUT or neural net weights from file. The load must of course 
	 * * have knowledge of how the data was written out by the save method. 
	 * * You should raise an error in the case that an attempt is being 
	 * * made to load data into an LUT or neural net whose structure does not match 
	 * * the data in the file. (e.g. wrong number of hidden neurons). 
	 * * @throws IOException */ 
	public void load(String argFileName) throws IOException 
	{ 
		FileInputStream inputFile = new FileInputStream( argFileName );
		BufferedReader br = new BufferedReader( new InputStreamReader( inputFile ));
		
		int count = 0; 
		List<String> lines = Files.readAllLines(Paths.get(argFileName)); 
		if (lines.size() != lutSize) 
		{ 
			System.out.println("Error loading file, incorrect size"); 
		} 
		for (int i = 0; i < featureNumValues[5]; i++) 
		{ // 5 
			for (int j = 0; j < featureNumValues[4]; j++) 
			{ // 4 
				for (int k = 0; k < featureNumValues[3]; k++) 
				{ // 3 
					for (int l = 0; l < featureNumValues[2]; l++) 
					{ // 2 
						for (int m = 0; m < featureNumValues[1]; m++) 
						{ // 1 
							for (int n = 0; n < featureNumValues[0]; n++) 
							{ //0 
								qTable[i][j][k][l][m][n] = Double.parseDouble(lines.get(count));
								count +=1; 
							}
						}
					}
				}
			}
		} 
		br.close();
	} 
	
	public static void main(String[] args) { // TODO Auto-generated method 
		int [] featureNumValues = {6, 3, 6, 8, 3, 3}; 
		// initialized backwards! 
		int featureVectorSize = 6; 
		boolean trainingMode = true; 
		boolean initializeflag = false;
		MyLUT2 lut = new MyLUT2(initializeflag,trainingMode, featureVectorSize, featureNumValues); 
		lut.save(); 
		//String path = "C:\\Users\\Faizan Mansuri\\My Data\\trained_lut.txt"; 
		
		try 
		{ 
			lut.load(path); 
		} 
		catch (IOException e) 
		{ // TODO Auto-generated catch block 
			e.printStackTrace(); 
		} 
		
		for (int i = 0; i < featureNumValues[5]; i++) 
		{ // 5 
			for (int j = 0; j < featureNumValues[4]; j++) 
			{ // 4 
				for (int k = 0; k < featureNumValues[3]; k++) 
				{ // 3 
					for (int l = 0; l < featureNumValues[2]; l++) 
					{ // 2 
						for (int m = 0; m < featureNumValues[1]; m++) 
						{ // 1
							for (int n = 0; n < featureNumValues[0]; n++) 
							{ //0 
								System.out.println(i+" "+j+" "+k+" "+l+" "+m+" "+n+": "+lut.qTable[i][j][k][l][m][n]); 
							}
						} 
					} 
				} 
			} 
		} 
	} 
}
