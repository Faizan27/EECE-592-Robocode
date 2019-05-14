package my.rl.robot;

import robocode.AdvancedRobot;
import java.util.Arrays; 
import java.util.Random;
import my.rl.robot.NeuralNet; 
import robocode.*;

public class FizzBotFinal extends AdvancedRobot {
	static String laptop_path = "C:\\Users\\Faizan Mansuri\\My Data\\";  
	static String online_weights_file = laptop_path+"online_trained_weights.txt"; 
	static String offline_weights_file = laptop_path+"trained_weights_15_0.02_0.3.txt"; 
	
	// learning versus trained 
	private static boolean Q_LEARNING = true; 
	private static double EPSILON_INIT = 0.0; 
	private static double EPSILON = EPSILON_INIT; 
	private static boolean EXPLORATION = false; 
	private static boolean ON_POLICY = false; 
	private static int MAX_GAMES = 500; 
	
	// actions 
	enum Action {AIMANDFIRE, STRAFE, CIRCLE, DIAGRIGHT, DIAGLEFT} 
	
	// rewards // rewards 
	private static final double WALL_COLLISION_REWARD = -0.9; 
	private static final double AWAY_FROM_WALL = 0.1; 
	private static final double ROBOT_COLLISION_REWARD = -0.8; 
	private static final double HIT_TARGET_REWARD = 0.8; 
	private static final double I_AM_HIT_REWARD = -1.2; 
	private static final double WIN_GAME_REWARD = 1; 
	private static final double LOSE_GAME_REWARD = -1; 
	private static final double LOW_DIST_TO_ENEMY_REWARD = -0.6; 
	private static final double MID_DIST_TO_ENEMY_REWARD = 0.5; 
	
	// end rewards 
	private static double ALPHA = 0.2; 
	private static double GAMMA = 0.5;
	
	
	// stats to save on the enemy when I get to scan him 
	double enemyDistance = 0.0; 
	double enemyEnergy = 0.0; 
	double enemyHeading = 0.0; 
	double enemyBearing = 0.0; 
	
	// for auto aim 
	double enemyBearingRadians = 0.0; 
	double enemyVelocity = 0.0; 
	double enemyHeadingRadians = 0.0; 
	long lastScanTime = 0; 
	Action act; 
	int actionToTake; 
	
	//set corresponding index in feature vector 
	public static final int MY_ENERGY = 3; 
	public static final int DIST = 2; 
	public static final int ENEMY_BEARING = 1; 
	public static final int ACTION = 0; 
	
	public static boolean normalized = true; 
	public static boolean isOfflineTraining = false; 
	public static int numActions = 5; 
	public static int featureVectorSize = 3 + numActions; 
	public static int [] featureNumValues = {5, 37, 11, 11}; 
	
	// initialized backwards! 
	//args to NN: int numInputs, int numHiddenNeurons, double learningRate, double momentum, boolean isOfflineTraining, weights_file 
	
	//the NN that showed learning: public static NeuralNet nn = new NeuralNet(featureVectorSize, 15, 0.00001, 0.0, true, false, offline_weights_file); 
	public static NeuralNet nn = new NeuralNet(featureVectorSize, 15, 0.02, 0.3, isOfflineTraining, normalized, offline_weights_file); 
	int direction = 1; 
	private double [] prevState = new double[featureVectorSize - numActions]; 
	private double [] currState = new double[featureVectorSize - numActions]; 
	private double [] prevStateAction = new double[featureVectorSize]; 
	private static double [] sa = new double[featureVectorSize]; 
	private static boolean qsaSaved = false; 
	private double qsa = -1; 
	private double qPrimeSa = -1; 
	private double [] currStateAction = new double[featureVectorSize]; 
	private double [] currStateActionGreedy = new double[featureVectorSize]; 
	public double rewardPerTurn; 
	
	// learning stats 
	public static double rewardPerHundred = 0.0; 
	public static int numTotalGames = 0; 
	public static int numWinGamesPerHundred = 0; 
	public static int wallMargin = 60;
	
	
	public void run() 
	{ 
		setAdjustRadarForRobotTurn(true);     // keep the radar still while we turn 
		setAdjustGunForRobotTurn(true);     	// keep the gun still while we turn 
		turnRadarRight(180); 
		prevState = getState(); 
		
		/*** Q-Learning with NN ***/ 
		if (Q_LEARNING) 
		{ 
			while (true) 
			{ 
				turnRadarRight(90); 
				
				// choose a from s using policy 
				System.out.println("Choose Action"); 
				prevStateAction = chooseAction(prevState); 
				if (!qsaSaved) 
				{ 
					System.arraycopy(prevStateAction, 0, sa, 0, prevStateAction.length); 
					qsaSaved = true; 
				} 
				if (Arrays.equals(prevStateAction, sa)) 
				{ 
					computeErrorSA(); 
				} 
				
				// take action a 
				performAction(actionToTake); 
				
				// observe some rewards r 
				if (enemyDistance <= 250) 
				{ 
					rewardPerTurn += LOW_DIST_TO_ENEMY_REWARD; 
				} 
				else if (enemyDistance > 250 && enemyDistance <= 333) 
				{ 
					rewardPerTurn += MID_DIST_TO_ENEMY_REWARD; 
				} 
				
				double x, y; 
				x = getX(); 
				y = getY(); 
				
				if ( (x <= wallMargin) || (x >= (getBattleFieldWidth() - wallMargin)) || (y <= wallMargin) || (y >= (getBattleFieldHeight() - wallMargin)) ) 
				{ 
					rewardPerTurn += WALL_COLLISION_REWARD; 
				} 
				else 
				{ 
					rewardPerTurn += AWAY_FROM_WALL; 
				} 
				
				// observe s'
				
				currState = getState(); 
				currStateAction = chooseAction(currState); 
				updateNN(prevStateAction, currStateAction); 
				System.out.println("rewardperTurn: "+rewardPerTurn); 
				rewardPerHundred +=rewardPerTurn; 
				rewardPerTurn = 0; // reset reward for next state transition 
				
				// prevState = currState 
				System.arraycopy(currState, 0, prevState, 0, prevState.length); 
			} 
		} 
		
		else 
		{ 
			// playing from trained table 
			prevState = getState(); 
			while(true) 
			{ 
				if (getTime() - lastScanTime > 5) 
				{ 
					turnRadarRight(90); 
				} 
				
				prevStateAction = chooseAction(prevState); 
				
				// take action a 
				performAction(actionToTake); 
				rewardPerTurn = 0; 
				prevState = getState(); 
			} 
		} 
	} 
	
	
	public void computeErrorSA() 
	{ 
		qPrimeSa = nn.outputFor(sa); 
		double errorSa = qPrimeSa - qsa; 
		nn.saveErrorSA(errorSa, numTotalGames); 
		qsa = qPrimeSa; 
	} 
	
	// all mapping for feature vector state occurs here (i.e. int indexes) 
	public double [] getState() 
	{ 
		/* In order from MSB to LSB */ 
		// energy: 0.0, 0.1...100.0 
		// distance to enemy: 0.0, 0.1...1000.0 
		// bearing: -1.80, -1.79, ... 1.80 
		// action: one hot encoded 
		double [] state = new double[featureVectorSize - numActions];
		
		// energy (low, medium high) 
		double energy = getEnergy(); 
		if (energy > 100) 
			energy = 100.0; 
		
		state[MY_ENERGY - 1] = energy/10.0; 
		
		// distance to enemy (low, med, far) 
		double distance = enemyDistance;
		
		state[DIST - 1] = distance/100.0; 
		// enemy bearing 
		state[ENEMY_BEARING - 1] = enemyBearing/100.0; 
		System.out.println("Energy: "+energy+" Dist: "+distance+" Bearing: "+enemyBearing); 
		System.out.println("Quantized Energy: "+state[MY_ENERGY - 1]+" Dist: "+state[DIST - 1]+" Bearing: "+state[ENEMY_BEARING - 1]); 
		return state; 
	} 
	
	public double[] getCategoricalActionRepresentation(int index) 
	{ 
		double [] representation = null; 
		if (index == 0) representation = new double[]{-1.0, -1.0, -1.0, -1.0, 1.0}; 
		else if(index == 1) representation = new double[]{-1.0, -1.0, -1.0, 1.0, -1.0}; 
		else if (index == 2) representation = new double[]{-1.0, -1.0, 1.0, -1.0, -1.0}; 
		else if (index == 3) representation = new double[]{-1.0, 1.0, -1.0, -1.0, -1.0}; 
		else if (index == 4) representation = new double[]{1.0, -1.0, -1.0, -1.0, -1.0};
		return representation; 
	} 
	
	/** 
	 * * returns state-action in expected NN representation */ 
	
	public double [] chooseAction(double[] prevState2) 
	{ 
		double qValue = 0.0; 
		double maxQValue = Double.NEGATIVE_INFINITY; 
		int actionToTake = 0; 
		Random number = new Random(); 
		double rand = number.nextDouble(); 
		double [] fullFeatureVector = null; // 3 + actions 
		
		double [] qValueSet = outputQValuesForCurrentState(prevState2); 
		assert(qValueSet.length == featureNumValues[0]); 
		
		/* determine random or greedily chosen action */ 
		if (rand < EPSILON && EXPLORATION) 
		{ 
			// choose random 
			actionToTake = (int) (0 + (qValueSet.length - 0) * number.nextDouble()); 
			this.actionToTake = actionToTake; 
			
			if (!ON_POLICY) 
			{ 
				// choose greedy for off policy update later 
				for (int i = 0; i < qValueSet.length; i++) 
				{ 
					qValue = qValueSet[i];
					if (qValue > maxQValue) 
					{ 
						maxQValue = qValue; 
						actionToTake = i; 
					} 
				} 
				this.actionToTake = actionToTake; 
				fullFeatureVector = concat(reverseArray(prevState2), getCategoricalActionRepresentation(this.actionToTake)); 
				System.arraycopy(fullFeatureVector, 0, currStateActionGreedy, 0, fullFeatureVector.length); 
			} 
		} 
		
		else 
		{ 
			// choose greedy 
			for (int i = 0; i < qValueSet.length; i++) 
			{ 
				qValue = qValueSet[i]; 
				if (qValue > maxQValue) 
				{ 
					maxQValue = qValue; 
					actionToTake = i; 
				} 
			} 
			
			this.actionToTake = actionToTake; 
			fullFeatureVector = concat(reverseArray(prevState2), getCategoricalActionRepresentation(this.actionToTake)); 
			System.arraycopy(fullFeatureVector, 0, currStateActionGreedy, 0, fullFeatureVector.length); 
		} 
		return fullFeatureVector; 
	} 
	
	
	public double[] outputQValuesForCurrentState(double [] state)
	{ 
		double [] qValues = new double[numActions]; 
		double [] revState = reverseArray(state); 
		double [] stateAction = null; 
		for (int i = 0; i<numActions; i++) 
		{ 
			stateAction = concat(revState, getCategoricalActionRepresentation(i));
			qValues[i] = nn.outputFor(stateAction);
		} 
		return qValues; 
	} 
	
	public double[] reverseArray(double [] arr) 
	{ 
		double[] toReverse = new double[arr.length]; 
		System.arraycopy(arr, 0, toReverse, 0, arr.length); 
		for(int i = 0; i < toReverse.length / 2; i++) 
		{ 
			double temp = toReverse[i]; 
			toReverse[i] = toReverse[toReverse.length - i - 1]; 
			toReverse[toReverse.length - i - 1] = temp; 
		} 
		return toReverse;
	} 
	
	public double[] concat(double[] a, double[] b) 
	{ 
		int aLen = a.length; 
		int bLen = b.length; 
		double[] c= new double[aLen+bLen]; 
		System.arraycopy(a, 0, c, 0, aLen); 
		System.arraycopy(b, 0, c, aLen, bLen); 
		return c; 
	} 
	
	public void onScannedRobot(ScannedRobotEvent e) 
	{ 
		System.out.println("Scanned enemy robot!"); 
		lastScanTime = getTime(); 
		
		// update state info on enemy 
		enemyDistance = e.getDistance(); 
		enemyEnergy = e.getEnergy(); 
		enemyHeading = e.getHeading(); 
		enemyBearing = e.getBearing(); 
		enemyBearingRadians = e.getBearingRadians(); 
		enemyVelocity = e.getVelocity(); 
		enemyHeadingRadians = e.getHeadingRadians(); 
		
		// auto aim 
		double absBearing=e.getBearingRadians()+getHeadingRadians();  //enemies absolute bearing 
		double latVel=e.getVelocity() * Math.sin(e.getHeadingRadians() - absBearing); //enemies later velocity 
		
		setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar 
		// turn gun to face enemy 
		double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(absBearing- getGunHeadingRadians()+latVel/22);
		//amount to turn our gun, lead just a little bit 
		setTurnGunRightRadians(gunTurnAmt); //turn our gun 
		execute(); 
	} 
	
	public void updateNN(double[] prevStateAction, double[] currStateAction) 
	{ 
		double prevQValue = nn.outputFor(prevStateAction); 
		double currQValueTaken = nn.outputFor(currStateAction); 
		if (ON_POLICY) 
		{ 
			// make update based on the action that was performed 
			nn.trainModified(prevStateAction, ((1-ALPHA)*prevQValue + ALPHA*(rewardPerTurn + GAMMA*currQValueTaken)));
		} 
		else 
		{ 
			// make update based on the greedy action regardless of whether you took it
			double currQValueGreedy = nn.outputFor(currStateActionGreedy); 
			nn.trainModified(prevStateAction, ((1-ALPHA)*prevQValue + ALPHA*(rewardPerTurn + GAMMA*currQValueGreedy))); 
		} 
	} 
	
	public double normalizeQValue(double input) 
	{ 
		double result; 
		double max = 1.3552442660534507; 
		double min = -17.86177741523035; 
		double bipolar_min = -1.0; 
		double bipolar_max = 1.0; 
		result = (input-min)/(max-min); 
		result = result*(bipolar_max - bipolar_min) + bipolar_min; 
		return result; 
	} 
	
	public void performAction(int action) 
	{ 
		act = Action.values()[action]; 
		
		switch (act) 
		{ 
			case AIMANDFIRE: 
				System.out.println("Firing"); 
				//Automatically aim and fire on scanning an enemy robot 
				double absBearing = enemyBearingRadians+getHeadingRadians();//enemies absolute bearing 
				double latVel = enemyVelocity * Math.sin(enemyHeadingRadians - absBearing);//enemies later velocity 
				setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar 
				// turn gun to face enemy 
				double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(absBearing- getGunHeadingRadians()+latVel/22);
				//amount to turn our gun, lead just a little bit 
				setTurnGunRightRadians(gunTurnAmt); //turn our gun 
				setFire(Math.min(400 / enemyDistance, 3)); 
				// if e farther away, use less fire power. Increase fire power if e is closer. 
				execute(); 
				break; 
				
			case STRAFE: 
				System.out.println("Strafe"); 
				setTurnRight(enemyBearing + 90); 
				setAhead(direction * 100); 
				execute(); 
				direction *= -1; 
				setAhead(direction * 100);
				execute(); 
				break; 
				
			case CIRCLE: 
				System.out.println("Circle"); 
				setTurnRight(enemyBearing + 90); 
				setAhead(direction * 100); 
				execute(); 
				break; 
				
			case DIAGRIGHT: 
				System.out.println("DiagRight"); 
				setTurnRight(45); 
				setAhead(direction * 100); 
				execute(); 
				break; 
				
			case DIAGLEFT: 
				System.out.println("DiagLeft"); 
				setTurnLeft(45); 
				setAhead(direction * 100); 
				execute(); 
				break; 
				
			default: 
				break; 
		} 
		
		while(getDistanceRemaining() != 0 || getTurnRemaining() != 0) 
		{ 
			execute(); 
		} 
	}
	
	public void onHitWall(HitWallEvent e) 
	{ 
		System.out.println("Robot hit a wall"); 
		rewardPerTurn += WALL_COLLISION_REWARD; // change direction auto on wall hit 
		direction *= -1; 
		setAhead(100 * direction); 
		execute(); 
	} 
	
	/** 
	 * * Occurs when my robot collides with another robot */ 
	
	public void onHitRobot(HitRobotEvent e) 
	{ 
		System.out.println("Robot hit another robot"); 
		rewardPerTurn += ROBOT_COLLISION_REWARD; 
	} 
	
	/** 
	 * * One of my bullets hit the enemy robot */ 
	public void onBulletHit(BulletHitEvent e) 
	{ 
		System.out.println("Robot shot the enemy"); 
		rewardPerTurn += HIT_TARGET_REWARD; 
	}
	
	/** 
	 * * I am hit by a bullet */ 
	public void onHitByBullet(HitByBulletEvent e) 
	{ 
		System.out.println("I got shot"); 
		rewardPerTurn += I_AM_HIT_REWARD; 
	} 
	
	public void onWin(WinEvent e) 
	{ 
		System.out.println("I won"); 
		rewardPerTurn += WIN_GAME_REWARD; 
		updateNN(prevStateAction, currStateAction); 
		numTotalGames++; 
		System.out.println("Num total games "+numTotalGames); 
		numWinGamesPerHundred++; 
		writeToFile(); 
	} 
	
	public void onDeath(DeathEvent e) 
	{ 
		System.out.println("I died"); 
		rewardPerTurn += LOSE_GAME_REWARD; 
		updateNN(prevStateAction, currStateAction); 
		numTotalGames++; 
		System.out.println("Num total games "+numTotalGames); 
		writeToFile(); 
	} 
	
	public void writeToFile() 
	{ 
		if ((numTotalGames % 100) == 0) 
		{ // output to file numWinGamesPerHundred 
			nn.saveGameStats(numWinGamesPerHundred, rewardPerHundred); 
			numWinGamesPerHundred = 0; 
			rewardPerHundred = 0; 
		} 
		if (getRoundNum() + 1 == MAX_GAMES) 
		{ 
			nn.save(online_weights_file); 
		} 
	} 
	
	/** 
	 * * normalizes a bearing to between +180 and -180 * 
	 * @param angle * @return
	 */
	double normalizeBearing(double angle) 
	{ 
		while (angle > 180) angle -= 360; 
		while (angle < -180) angle += 360; 
		return angle; 
	}
}