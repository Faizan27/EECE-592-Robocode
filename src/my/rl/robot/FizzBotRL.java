package my.rl.robot;
import java.awt.Color;
import static robocode.util.Utils.normalRelativeAngleDegrees;

import java.awt.geom.Point2D;
import java.util.Random;

import robocode.*;
public class FizzBotRL extends AdvancedRobot {
	//learning versus trained 
	private static boolean Q_LEARNING = true; //Set this to true for learning
	private static boolean initializeflag = false; //Set this true if training for first time
	private static double EPSILON_INIT = 0.2; 
	private static double EPSILON = EPSILON_INIT; 
	private static boolean EXPLORATION = true; 
	private static boolean ON_POLICY = false; 
	private static int MAX_GAMES = 100000; 
	
	// actions 
	enum Action {AIMANDFIRE, FORWARD, BACKWARD, DIAGRIGHT, DIAGLEFT, CHANGEDIRECTION} 
	
	// rewards 
	private static final double WALL_COLLISION_REWARD = -0.9; 
	private static final double AWAY_FROM_WALL = 0.4; 
	private static final double ROBOT_COLLISION_REWARD = -0.4; 
	private static final double HIT_TARGET_REWARD = 0.3; 
	private static final double I_AM_HIT_REWARD = -1.2; 
	private static final double WIN_GAME_REWARD = 1; 
	private static final double LOSE_GAME_REWARD = -1; 
	private static final double LOW_DIST_TO_ENEMY_REWARD = -0.6; 
	private static final double MID_DIST_TO_ENEMY_REWARD = 0.7; 
	private static final double CLOSE_BEARING_REWARD = -0.5; 
	private static final double MID_BEARING_REWARD = 0.3; 
	// end rewards

	private static double ALPHA = 0.1; 
	private static double GAMMA = 0.9; 
	int direction = 1; 
	private int [] prevState = new int[featureVectorSize - 1]; 
	private int [] currState = new int[featureVectorSize - 1]; 
	private int [] prevStateAction = new int[featureVectorSize]; 
	private int [] currStateAction = new int[featureVectorSize]; 
	private int [] currStateActionGreedy = new int[featureVectorSize]; 
	//Terminal Reward
	public int totalReward;
	//Intermediate Reward
	public double rewardPerTurn; 
	
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
	//private double gunTurnAmt; 
	
	// static declarations /* In order from MSB to LSB */ 
	// energy: 0, 1, 2 
	// distance to enemy: 0, 1, 2 
	// x_pos: 0, 1, 2, 3, 4, 5, 6, 7 
	// y_pos: 0, 1, 2, 3, 4, 5 
	// bearing: 0, 1, 2 
	// action: 0 - 5 
	//set corresponding index in feature vector 
	public static final int MY_ENERGY = 5; 
	public static final int DIST = 4; 
	public static final int X_DIST = 3; 
	public static final int Y_DIST = 2; 
	public static final int ENEMY_BEARING = 1; 
	public static final int ACTION = 0; 
	public static int featureVectorSize = 6; 
	public static int [] featureNumValues = {6, 3, 6, 8, 3, 3}; 
	// initialized backwards! 
	public static MyLUT2 lut = new MyLUT2(initializeflag,Q_LEARNING, featureVectorSize, featureNumValues); 
	public int changecolor = 0;
	// learning stats 
	public static double rewardPerHundred = 0.0; 
	public static int numTotalGames = 0; 
	public static int numWinGamesPerHundred = 0; 
	public static int wallMargin = 60;
	
	public void run() 
	{ 
		
		setColors(null, new Color(0,255,0), new Color(255,0,0), Color.black, new Color(0, 0, 255));
	    setBodyColor(new java.awt.Color(255-changecolor,192 - changecolor,150 - changecolor,100));
		//lut = new MyLUT2(Q_LEARNING, featureVectorSize, featureNumValues); 
		setAdjustRadarForRobotTurn(true); 
		// keep the radar still while we turn 
		setAdjustGunForRobotTurn(true); 
		setAdjustRadarForGunTurn(true);
		//Fancy Implementation - Cosmetics of Robot
		if (changecolor >= 255)
			changecolor = 0;
		else
			changecolor += 5;
		// keep the gun still while we turn 
		turnGunRight(10); 
		prevState = getState(); 
		// Turn the radar if we have no more turn, starts it if it stops and at the start of round
        if ( getRadarTurnRemaining() == 0.0 )
        	setTurnRadarRightRadians( Double.POSITIVE_INFINITY );
		scan();
		/*** Q-Learning ***/ 
		if (Q_LEARNING) 
		{ 
			while (true) 
			{ 
				//scan();
				turnRadarRight(10); 
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
					turnRadarRight(10); 
				} 
				prevStateAction = chooseAction(prevState); 
				
				// take action a 
				performAction((int)prevStateAction[0]); 
				rewardPerTurn = 0; 
				prevState = getState(); 
			} 
		} 
	} 
	
	// all mapping for feature vector state occurs here (i.e. int indexes) 
	public int [] getState() 
	{ 
		/* In order from MSB to LSB */ 
		// wall distance: 0, 1 
		// energy: 0, 1, 2 
		// distance to enemy: 0, 1, 2 
		// x_dist: 0, 1, 2, 3, 4, 5, 6, 7 
		// y_dist: 0, 1, 2, 3, 4, 5, 6, 
		// bearing: 0, 1, 2 
		// action: 0 - 5 
		
		int [] state = new int[featureVectorSize - 1]; 
		
		// energy (low, medium high) 
		double energy = getEnergy(); 
		if (energy <= 33) 
		{ 
			state[MY_ENERGY - 1] = 0; 
		} 
		else if (energy > 33 && energy <= 66) 
		{ 
			state[MY_ENERGY - 1] = 1; 
		} 
		else 
		{ 
			state[MY_ENERGY - 1] = 2;
		}
		
		// distance to enemy (low, med, far) 
		double distance = enemyDistance; 
		assert(distance >=0 && distance <=1000); 
		if (distance <= 150) 
		{ 
			state[DIST - 1] = 0; 
		} 
		else if (distance > 100 && distance <= 333) 
		{ 
			state[DIST - 1] = 1; 
		} 
		else 
		{ 
			state[DIST - 1] = 2; 
		} 
		
		// x pos 
		state[X_DIST - 1] = (int) (getX() / 100); 
		// y pos 
		state[Y_DIST - 1] = (int) (getY() / 100); 
		
		// enemy bearing //double bearing = Math.abs(enemyBearing); 
		double bearing = Math.abs(normalizeBearing(getHeading() - enemyHeading)); 
		if (bearing <= 60) 
		{ 
			state[ENEMY_BEARING - 1] = 0; 
		} 
		else if (bearing > 60 && bearing <= 120) 
		{ 
			state[ENEMY_BEARING - 1] = 1; 
		} 
		else 
		{ 
			state[ENEMY_BEARING - 1] = 2; 
		} 
		return state; 
	} 
	
	public int [] chooseAction(int[] prevState2) 
	{ 
		double qValue = 0.0; 
		double maxQValue = Double.NEGATIVE_INFINITY; 
		int actionToTake = 0; 
		Random number = new Random(); 
		double rand = number.nextDouble(); 
		int [] fullFeatureVector = new int[featureVectorSize]; 
		assert(prevState2.length == featureVectorSize - 1); 
		System.arraycopy(prevState2, 0, fullFeatureVector, 1, prevState2.length); 
		double [] actionSet = lut.outputQvaluesForCurrentState(prevState2); 
		assert(actionSet.length == featureNumValues[0]); 
		
		/* determine random or greedily chosen action */ 
		if (rand < EPSILON && EXPLORATION) 
		{ 
			// choose random 
			actionToTake = (int) (0 + (actionSet.length - 0) * number.nextDouble()); 
			fullFeatureVector[ACTION] = actionToTake;
			if (!ON_POLICY) 
			{ 
				// choose greedy for off policy update later 
				for (int i = 0; i < actionSet.length; i++) 
				{ 
					qValue = actionSet[i]; 
					if (qValue > maxQValue) 
					{ 
						maxQValue = qValue; 
						actionToTake = i; 
					} 
				} 
				System.arraycopy(fullFeatureVector, 0, currStateActionGreedy, 0, fullFeatureVector.length); 
				currStateActionGreedy[ACTION] = actionToTake; 
			} 
		} 
		else 
		{ 
			// choose greedy 
			for (int i = 0; i < actionSet.length; i++) 
			{ 
				qValue = actionSet[i]; 
				if (qValue > maxQValue) 
				{ 
					maxQValue = qValue; 
					actionToTake = i; 
				} 
			} 
			
			fullFeatureVector[ACTION] = actionToTake; 
			System.arraycopy(fullFeatureVector, 0, currStateActionGreedy, 0, fullFeatureVector.length); 
		} 
		return fullFeatureVector;
	} 
	
	public void onScannedRobot(ScannedRobotEvent e) 
	{ 
		//System.out.println("Scanned enemy robot!"); 
		lastScanTime = getTime(); 
		// update state info on enemy 
		enemyDistance = e.getDistance(); 
		enemyEnergy = e.getEnergy(); 
		enemyHeading = e.getHeading(); 
		enemyBearing = e.getBearing(); 
		enemyBearingRadians = e.getBearingRadians(); 
		enemyVelocity = e.getVelocity(); 
		enemyHeadingRadians = e.getHeadingRadians(); 
		
		// Calculate exact location of the robot
		double absBearing=e.getBearingRadians()+getHeadingRadians();
				double absoluteBearing = getHeading() + e.getBearing();
				double bearingFromGun = normalRelativeAngleDegrees(absoluteBearing - getGunHeading());

				// If it's close enough, fire!
				if (Math.abs(bearingFromGun) <= 3) {
					turnGunRight(bearingFromGun);
					// We check gun heat here, because calling fire()
					// uses a turn, which could cause us to lose track
					// of the other robot.
					if (getGunHeat() == 0) {
						fire(Math.min(3 - Math.abs(bearingFromGun), getEnergy() - .1));
					}
				} // otherwise just set the gun to turn.
				// Note:  This will have no effect until we call scan()
				else {
					turnGunRight(bearingFromGun);
				}
		/*
		double absBearing=e.getBearingRadians()+getHeadingRadians();		//enemies absolute bearing 
		// Absolute angle towards target
	    double angleToEnemy = getHeadingRadians() + e.getBearingRadians();
	     // Calculate exact location of the robot
		double absoluteBearing = getHeading() + e.getBearing();
		double bearingFromGun = Utils.normalRelativeAngleDegrees(absoluteBearing - getGunHeading());

		if (Math.abs(bearingFromGun) <= 3) {
			turnGunRight(bearingFromGun);
		}else {
			turnGunRight(bearingFromGun);
		}
		
	    /******************************/
	    // Subtract current radar heading to get the turn required to face the enemy, be sure it is normalized
	    //double radarTurn = Utils.normalRelativeAngle( angleToEnemy - getRadarHeadingRadians() );
	    //double extraTurn = Math.min( Math.atan( 36.0 / enemyDistance ), Rules.RADAR_TURN_RATE_RADIANS );
	    
	    /*radarTurn =
	            // Absolute bearing to target
	            getHeadingRadians() + e.getBearingRadians()
	            // Subtract current radar heading to get turn required
	            - getRadarHeadingRadians();
	     
	        setTurnRadarRightRadians(2.0*Utils.normalRelativeAngle(radarTurn));
	    
	    //Turn the radar
        if (radarTurn < 0)
	        radarTurn -= extraTurn;
	    else
	        radarTurn += extraTurn;
	    setTurnRadarRightRadians(radarTurn);
	    */
	    /**********************************************
	     * double angleToEnemy = getHeadingRadians() + enemyBearingRadians;
				double radarTurn = Utils.normalRelativeAngle( angleToEnemy - getRadarHeadingRadians() );
			    // The 36.0 is how many units from the center of the enemy robot it scans.
			    double extraTurn = Math.min( Math.atan( 36.0 / enemyDistance ), Rules.RADAR_TURN_RATE_RADIANS );
			    if (radarTurn < 0)
			        radarTurn -= extraTurn;
			    else
			        radarTurn += extraTurn;
			    radarTurn =
			            // Absolute bearing to target
			            getHeadingRadians() + enemyBearingRadians
			            // Subtract current radar heading to get turn required
			            - getRadarHeadingRadians();
			     
			     //   setTurnRadarRightRadians(2.0*Utils.normalRelativeAngle(radarTurn));
			    //Turn the radar
			    setTurnRadarRightRadians(radarTurn);
	     */
	    execute();
		/*
	    //fire
		double qdistancetoenemy =0;
		if((enemyDistance > 0) && (enemyDistance<=250)){
			qdistancetoenemy=1;
			}
		else if((enemyDistance > 250) && (enemyDistance<=500)){
			qdistancetoenemy=2;
			}
		else if((enemyDistance > 500) && (enemyDistance<=750)){
			qdistancetoenemy=3;
			}
		else if((enemyDistance > 750) && (enemyDistance<=1000)){
			qdistancetoenemy=4;
			}
		
		if(qdistancetoenemy==1) {
			if (getGunHeat() == 0 && Math.abs(getGunTurnRemaining()) < 10)
				fire(3);
			}
		if(qdistancetoenemy==2){
			if (getGunHeat() == 0 && Math.abs(getGunTurnRemaining()) < 10)
			//	setFire(firePower);
			fire(2);}
		if(qdistancetoenemy==3){
			if (getGunHeat() == 0 && Math.abs(getGunTurnRemaining()) < 10)
			//	setFire(firePower);
			fire(1);}
		//fire
		*/
	    if (Q_LEARNING) 
		{ 
			// choose a from s using policy 
			System.out.println("Choose Action"); 
			prevStateAction = chooseAction(prevState); 
			// take action a 
			performAction((int)prevStateAction[0]); 
			// observe some rewards r 
			if (enemyDistance <= 250) 
			{ 
				rewardPerTurn += LOW_DIST_TO_ENEMY_REWARD; 
			} 
			else if (enemyDistance > 250 && enemyDistance <= 333) 
			{ 
				rewardPerTurn += MID_DIST_TO_ENEMY_REWARD; 
			} 
			System.out.println("Enemy bearing: "+enemyBearing); 
			absBearing = Math.abs(normalizeBearing(getHeading() - enemyHeading)); 
			System.out.println("Angle between robot headings: "+absBearing); 
			if (absBearing <= 60) 
			{ 
				rewardPerTurn += CLOSE_BEARING_REWARD; 
			} 
			else if (absBearing > 60 && absBearing <= 120) 
			{ 
				rewardPerTurn += MID_BEARING_REWARD; 
			} 
			else 
			{ 
				rewardPerTurn += CLOSE_BEARING_REWARD; 
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
			updateLUT(prevStateAction, currStateAction); 
			System.out.println("rewardperTurn: "+rewardPerTurn); 
			rewardPerHundred +=rewardPerTurn; 
			rewardPerTurn = 0; // reset reward for next state transition
			
			//prevState = currState 
			System.arraycopy(currState, 0, prevState, 0, prevState.length); 
		}
	    if (bearingFromGun == 0) {
			scan();
		}
	
	} 
	

	public void updateLUT(int[] prevStateAction, int[] currStateAction) 
	{ 
		double prevQValue = lut.outputFor(prevStateAction); 
		double currQValueTaken = lut.outputFor(currStateAction); 
		if (ON_POLICY) 
		{ 
			// make update based on the action that was performed 
			lut.train(prevStateAction, ((1-ALPHA)*prevQValue + ALPHA*(rewardPerTurn + GAMMA*currQValueTaken))); 
		} 
		else 
		{ 
			// make update based on the greedy action regardless of whether you took it 
			double currQValueGreedy = lut.outputFor(currStateActionGreedy); 
			lut.train(prevStateAction, ((1-ALPHA)*prevQValue + ALPHA*(rewardPerTurn + GAMMA*currQValueGreedy))); 
		}
	} 
	
	public void performAction(int action) 
	{ 
		act = Action.values()[action]; 
		switch (act) 
		{ 
			case AIMANDFIRE: 
				System.out.println("Firing"); 
				//Automatically aim and fire on scanning an enemy robot 
				double absBearing = enemyBearingRadians+getHeadingRadians(); //enemies absolute bearing 
				double latVel = enemyVelocity * Math.sin(enemyHeadingRadians - absBearing);//enemies later velocity 
				//setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar 
				// turn gun to face enemy 
				double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(absBearing- getGunHeadingRadians()+latVel/22);
				//amount to turn our gun, lead just a little bit 
				setTurnGunRightRadians(gunTurnAmt); //turn our gun 
				setFire(Math.min(400 / enemyDistance, 3)); 
				// if e farther away, use less fire power. Increase fire power if e is closer.
				/*
				double angleToEnemy = getHeadingRadians() + enemyBearingRadians;
				double radarTurn = Utils.normalRelativeAngle( angleToEnemy - getRadarHeadingRadians() );
			    // The 36.0 is how many units from the center of the enemy robot it scans.
			    double extraTurn = Math.min( Math.atan( 36.0 / enemyDistance ), Rules.RADAR_TURN_RATE_RADIANS );
			    if (radarTurn < 0)
			        radarTurn -= extraTurn;
			    else
			        radarTurn += extraTurn;
			    /*
			    radarTurn =
			            // Absolute bearing to target
			            getHeadingRadians() + enemyBearingRadians
			            // Subtract current radar heading to get turn required
			            - getRadarHeadingRadians();
			     
			     //   setTurnRadarRightRadians(2.0*Utils.normalRelativeAngle(radarTurn));
			    //Turn the radar
			    setTurnRadarRightRadians(radarTurn);
			    double absoluteBearing = getHeading() + enemyBearing;
				double bearingFromGun = Utils.normalRelativeAngleDegrees(absoluteBearing - getGunHeading());
				turnGunRight(bearingFromGun);
				//setTurnGunRight(getHeading() - getGunHeading() + enemyBearing);
			    if (getGunHeat() == 0 && Math.abs(getGunTurnRemaining()) < 10)
			    	setFire(Math.min(400 / enemyDistance, 3)); 
				// if e farther away, use less fire power. Increase fire power if e is closer. 
				*/
				execute(); 
				break; 
			
			case FORWARD: 
				System.out.println("Going Forward"); 
				setAhead(direction * 100); 
				execute(); 
				break; 
			
			case BACKWARD: 
				System.out.println("Going Backward");
				setBack(direction * 100); 
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
				
			case CHANGEDIRECTION: 
				System.out.println("ChangingDirection"); 
				direction *= -1; 
				setAhead(200 * direction); 
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
		//rewardPerTurn += WALL_COLLISION_REWARD;
		double xPos=this.getX();
		double yPos=this.getY();
		double width=this.getBattleFieldWidth();
		double height=this.getBattleFieldHeight();
		if(yPos<80)//too close to the bottom
		{
			
			turnLeft(getHeading() % 90);
			//System.out.println("Get heading");
			//System.out.println(getHeading());
			if(getHeading()==0){turnLeft(0);}
			if(getHeading()==90){turnLeft(90);}
			if(getHeading()==180){turnLeft(180);}
			if(getHeading()==270){turnRight(90);}
			ahead(150);
			//System.out.println("Too close to the bottom");
			if ((this.getHeading()<180)&&(this.getHeading()>90))
			{
				this.setTurnLeft(90);
			}
			else if((this.getHeading()<270)&&(this.getHeading()>180))
			{
				this.setTurnRight(90);
			}
			
			
		}
		else if(yPos>height-80){ //to close to the top
			//System.out.println("Too close to the Top");
			if((this.getHeading()<90)&&(this.getHeading()>0)){this.setTurnRight(90);}
			else if((this.getHeading()<360)&&(this.getHeading()>270)){this.setTurnLeft(90);}
			turnLeft(getHeading() % 90);
			//System.out.println("Get heading");
			//System.out.println(getHeading());
			if(getHeading()==0){turnRight(180);}
			if(getHeading()==90){turnRight(90);}
			if(getHeading()==180){turnLeft(0);}
			if(getHeading()==270){turnLeft(90);}
			ahead(150);
			
		}
		else if(xPos<80){
			turnLeft(getHeading() % 90);
			//System.out.println("Get heading");
			//System.out.println(getHeading());
			if(getHeading()==0){turnRight(90);}
			if(getHeading()==90){turnLeft(0);}
			if(getHeading()==180){turnLeft(90);}
			if(getHeading()==270){turnRight(180);}
			ahead(150);
		}
		else if(xPos>width-80){
			turnLeft(getHeading() % 90);
			//System.out.println("Get heading");
			//System.out.println(getHeading());
			if(getHeading()==0){turnLeft(90);}
			if(getHeading()==90){turnLeft(180);}
			if(getHeading()==180){turnRight(90);}
			if(getHeading()==270){turnRight(0);}
			ahead(150);
		}
		
	} 
	
	/** * Occurs when my robot collides with another robot */ 
	public void onHitRobot(HitRobotEvent e) 
	{ 
		System.out.println("Robot hit another robot"); 
		rewardPerTurn += ROBOT_COLLISION_REWARD; 
	} 
	
	/** * One of my bullets hit the enemy robot */ 
	public void onBulletHit(BulletHitEvent e) 
	{ 
		System.out.println("Robot shot the enemy"); 
		rewardPerTurn += HIT_TARGET_REWARD;
		
	} 
	
	/** * I am hit by a bullet */ 
	public void onHitByBullet(HitByBulletEvent e) 
	{ 
		System.out.println("I got shot"); 
		rewardPerTurn += I_AM_HIT_REWARD;
		/****** Dodge Code ***************/
		setTurnRight(e.getBearing()+90-
		         30*direction);
		
		if(enemyDistance <= 250)
			direction = -1;
		else
			direction = 1;
		
		setAhead((enemyDistance/4+25)*direction); 
		execute();
		
	} 
	
	public void onWin(WinEvent e) 
	{ 
		System.out.println("I won"); 
		rewardPerTurn += WIN_GAME_REWARD; 
		updateLUT(prevStateAction, currStateAction); 
		numTotalGames++; 
		System.out.println("Num total games "+numTotalGames); 
		numWinGamesPerHundred++; 
		writeToFile(); 
	} 
	
	public void onDeath(DeathEvent e) 
	{ 
		System.out.println("I died"); 
		rewardPerTurn += LOSE_GAME_REWARD; 
		updateLUT(prevStateAction, currStateAction); 
		numTotalGames++; 
		System.out.println("Num total games "+numTotalGames);
		writeToFile(); 
	} 
	
	public void writeToFile() 
	{ 
		if ((numTotalGames % 100) == 0) 
		{ 
			// output to file numWinGamesPerHundred 
			lut.saveGameStats(numWinGamesPerHundred, rewardPerHundred); 
			numWinGamesPerHundred = 0; 
			rewardPerHundred = 0; 
			
			// exponentially decay EPSILON // 
			//EPSILON = EPSILON_INIT * Math.exp(-numTotalGames/MAX_GAMES); 
		} 
		
		if (numTotalGames == MAX_GAMES) 
		{ 
			lut.save(); 
		} 
	} 
	
	/** * normalizes a bearing to between +180 and -180 
	 * * @param angle
	    *@return */ 
	double normalizeBearing(double angle) 
	{ 
		while (angle > 180) 
			angle -= 360; 
		while (angle < -180) 
			angle += 360; 
		return angle; 
	}
	
	//absolute bearing
	double absoluteBearing(float x1, float y1, float x2, float y2) {
		double xo = x2-x1;
		double yo = y2-y1;
		double hyp = Point2D.distance(x1, y1, x2, y2);
		double arcSin = Math.toDegrees(Math.asin(xo / hyp));
		double bearing = 0;

		if (xo > 0 && yo > 0) { // both pos: lower-Left
			bearing = arcSin;
		} else if (xo < 0 && yo > 0) { // x neg, y pos: lower-right
			bearing = 360 + arcSin; // arcsin is negative here, actuall 360 - ang
		} else if (xo > 0 && yo < 0) { // x pos, y neg: upper-left
			bearing = 180 - arcSin;
		} else if (xo < 0 && yo < 0) { // both neg: upper-right
			bearing = 180 - arcSin; // arcsin is negative here, actually 180 + ang
		}

		return bearing;
	}

}