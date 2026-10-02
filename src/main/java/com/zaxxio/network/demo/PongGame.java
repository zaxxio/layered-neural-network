package com.zaxxio.network.demo;

import com.zaxxio.network.MultiLayerNetwork;
import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.MultiLayerConfiguration;
import com.zaxxio.network.config.step.OptimizationAlgo;
import com.zaxxio.network.model.DenseLayer;
import com.zaxxio.network.model.InputLayer;
import com.zaxxio.network.model.OutputLayer;
import com.zaxxio.network.weight.WeightInit;
import lombok.Getter;
import lombok.Setter;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.Random;

@Getter
@Setter
public class PongGame extends JPanel {

    // --- Micro Virtual Resolution (100x100 Grid) ---
    private static final int VIRTUAL_WIDTH = 100;
    private static final int VIRTUAL_HEIGHT = 100;

    private static final int PADDLE_WIDTH = 2;
    private static final int PADDLE_HEIGHT = 15; // Fast convergence height (15% of grid)
    private static final int BALL_SIZE = 2;

    private static final double PADDLE_SPEED = 2.0;
    private static final double BALL_SPEED = 1.5;

    // Paddle positions on virtual grid
    private static final int LEFT_PADDLE_X = 6;
    private static final int RIGHT_PADDLE_X = VIRTUAL_WIDTH - 6 - PADDLE_WIDTH; // 92

    // RL Hyperparameters
    private static final double GAMMA = 0.95;
    private double epsilon = 1.0;
    private static final double EPSILON_MIN = 0.05;
    private static final double EPSILON_DECAY = 0.9995;

    private final MultiLayerNetwork leftAI;
    private final MultiLayerNetwork rightAI;

    private double leftY = VIRTUAL_HEIGHT / 2.0 - PADDLE_HEIGHT / 2.0;
    private double rightY = VIRTUAL_HEIGHT / 2.0 - PADDLE_HEIGHT / 2.0;

    private double ballX, ballY, ballVX, ballVY;
    private int leftScore, rightScore;

    private final Random random = new Random();
    private final Timer timer;
    private boolean isTrainingHeadless = false;

    public PongGame() {
        leftAI = createRLNetwork();
        rightAI = createRLNetwork();

        setPreferredSize(new Dimension(500, 500));
        setBackground(new Color(15, 15, 25));
        setFocusable(true);

        resetBall(random.nextBoolean());

        // Press SPACEBAR to trigger 50,000 headless training frames instantly
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                    new Thread(() -> trainHeadless(50_000)).start();
                }
            }
        });

        timer = new Timer(16, e -> updateGame());
        timer.start();
    }

    private MultiLayerNetwork createRLNetwork() {
        MultiLayerConfiguration config = new MultiLayerConfiguration.Builder()
                .activation(ActivationFunction.LEAKY_RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgo.STOCHASTIC_GRADIENT_DESCENT)
                .momentum(0.4)
                .updater(0.005)
                .layer(0, new InputLayer.Builder().nIn(5).build())
                .layer(1, new DenseLayer.Builder().nIn(12).activation(ActivationFunction.LEAKY_RELU).build())
                .layer(2, new DenseLayer.Builder().nIn(12).activation(ActivationFunction.LEAKY_RELU).build())
                .layer(3, new OutputLayer.Builder().nOut(3).activation(ActivationFunction.IDENTITY).build())
                .list()
                .minError(0.001)
                .realtime(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        return network;
    }

    /**
     * Fast-forwards RL training without rendering frames or waiting for Swing timers.
     *
     * @param steps Number of game iterations to execute at maximum CPU speed.
     */
    public void trainHeadless(int steps) {
        if (isTrainingHeadless) return;
        isTrainingHeadless = true;
        timer.stop();

        System.out.println("Starting headless training for " + steps + " steps...");
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < steps; i++) {
            updateGame();
        }

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("Finished " + steps + " steps in " + elapsed + " ms (" + 
                (steps * 1000L / Math.max(1, elapsed)) + " FPS)");

        isTrainingHeadless = false;
        timer.start();
        repaint();
    }

    private void updateGame() {
        // 1. Capture State s
        double[] leftState = createInput(leftY);
        double[] rightState = createInput(rightY);

        // 2. Select Actions
        int leftAction = selectAction(leftAI, leftState);
        int rightAction = selectAction(rightAI, rightState);

        // 3. Apply Actions
        applyAction(leftAction, true);
        applyAction(rightAction, false);

        // 4. Ball Movement
        ballX += ballVX;
        ballY += ballVY;
        if (ballY <= 0) { ballY = 0; ballVY = Math.abs(ballVY); }
        if (ballY + BALL_SIZE >= VIRTUAL_HEIGHT) { ballY = VIRTUAL_HEIGHT - BALL_SIZE; ballVY = -Math.abs(ballVY); }

        // 5. Calculate Rewards
        double leftReward = calculateProximityReward(leftY, ballY, ballVX, true);
        double rightReward = calculateProximityReward(rightY, ballY, ballVX, false);

        Rectangle ball = new Rectangle((int) ballX, (int) ballY, BALL_SIZE, BALL_SIZE);
        Rectangle leftPaddle = new Rectangle(LEFT_PADDLE_X, (int) leftY, PADDLE_WIDTH, PADDLE_HEIGHT);
        Rectangle rightPaddle = new Rectangle(RIGHT_PADDLE_X, (int) rightY, PADDLE_WIDTH, PADDLE_HEIGHT);

        // Left Paddle Collision
        if (ballVX < 0 && ball.intersects(leftPaddle)) {
            ballX = LEFT_PADDLE_X + PADDLE_WIDTH;
            ballVX = Math.abs(ballVX) + 0.02;
            ballVY += ((ballY + BALL_SIZE / 2.0) - (leftY + PADDLE_HEIGHT / 2.0)) * 0.1;
            leftReward = 1.0;
            limitBallSpeed();
        }

        // Right Paddle Collision
        if (ballVX > 0 && ball.intersects(rightPaddle)) {
            ballX = RIGHT_PADDLE_X - BALL_SIZE;
            ballVX = -Math.abs(ballVX) - 0.02;
            ballVY += ((ballY + BALL_SIZE / 2.0) - (rightY + PADDLE_HEIGHT / 2.0)) * 0.1;
            rightReward = 1.0;
            limitBallSpeed();
        }

        // Scoring & Reset
        boolean reset = false;
        if (ballX < -BALL_SIZE) {
            rightScore++;
            leftReward = -1.0;
            reset = true;
        } else if (ballX > VIRTUAL_WIDTH) {
            leftScore++;
            rightReward = -1.0;
            reset = true;
        }

        // 6. Q-Learning Step
        double[] nextLeftState = createInput(leftY);
        double[] nextRightState = createInput(rightY);

        trainQNetwork(leftAI, leftState, leftAction, leftReward, nextLeftState, reset);
        trainQNetwork(rightAI, rightState, rightAction, rightReward, nextRightState, reset);

        if (reset) {
            resetBall(random.nextBoolean());
        }

        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

        if (!isTrainingHeadless) {
            repaint();
        }
    }

    private double calculateProximityReward(double paddleY, double ballY, double ballVX, boolean isLeft) {
        boolean ballApproaching = (isLeft && ballVX < 0) || (!isLeft && ballVX > 0);
        if (!ballApproaching) return 0.001;

        double paddleCenterY = paddleY + (PADDLE_HEIGHT / 2.0);
        double ballCenterY = ballY + (BALL_SIZE / 2.0);
        double distance = Math.abs(paddleCenterY - ballCenterY);
        double alignment = Math.max(0.0, 1.0 - (distance / (VIRTUAL_HEIGHT / 2.0)));
        return 0.05 * alignment;
    }

    private int selectAction(MultiLayerNetwork network, double[] state) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(3);
        }
        return argmax(network.predict(state));
    }

    public static double max(double[] array) {
        double maxVal = array[0];
        for (double v : array) {
            if (v > maxVal) maxVal = v;
        }
        return maxVal;
    }

    public static int argmax(double[] array) {
        int bestIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[bestIdx]) {
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private void applyAction(int action, boolean isLeft) {
        double currentY = isLeft ? leftY : rightY;
        if (action == 0) currentY -= PADDLE_SPEED;
        else if (action == 2) currentY += PADDLE_SPEED;

        currentY = clamp(currentY, 0, VIRTUAL_HEIGHT - PADDLE_HEIGHT);
        if (isLeft) leftY = currentY;
        else rightY = currentY;
    }

    private void trainQNetwork(
            MultiLayerNetwork network,
            double[] state,
            int action,
            double reward,
            double[] nextState,
            boolean isTerminal
    ) {
        double[] currentQ = network.predict(state);
        double targetQ = reward;

        if (!isTerminal) {
            double[] nextQ = network.predict(nextState);
            targetQ += GAMMA * max(nextQ);
        }

        double[] targetVector = currentQ.clone();
        targetVector[action] = targetQ;

        network.forward(state);
        network.backward(targetVector);
    }

    private double[] createInput(double paddleY) {
        return new double[]{
                normalizeX(ballX),
                normalizeY(ballY),
                normalizeVelocity(ballVX),
                normalizeVelocity(ballVY),
                normalizeY(paddleY)
        };
    }

    private double normalizeX(double value) { return clamp(value / VIRTUAL_WIDTH, 0, 1); }
    private double normalizeY(double value) { return clamp(value / (VIRTUAL_HEIGHT - BALL_SIZE), 0, 1); }
    private double normalizeVelocity(double value) { return clamp(value / 3.0, -1, 1); }
    private double clamp(double v, double min, double max) { return Math.max(min, Math.min(max, v)); }
    private void limitBallSpeed() { ballVY = clamp(ballVY, -2.5, 2.5); }

    private void resetBall(boolean leftServe) {
        ballX = VIRTUAL_WIDTH / 2.0;
        ballY = VIRTUAL_HEIGHT / 2.0;
        double direction = leftServe ? -1 : 1;
        ballVX = direction * BALL_SPEED;
        ballVY = (random.nextDouble() * 1.0) - 0.5;
        leftY = VIRTUAL_HEIGHT / 2.0 - PADDLE_HEIGHT / 2.0;
        rightY = VIRTUAL_HEIGHT / 2.0 - PADDLE_HEIGHT / 2.0;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Calculate aspect-ratio-preserving 1:1 square scale inside window
        double scaleX = (double) getWidth() / VIRTUAL_WIDTH;
        double scaleY = (double) getHeight() / VIRTUAL_HEIGHT;
        double scale = Math.min(scaleX, scaleY);

        double offsetX = (getWidth() - (VIRTUAL_WIDTH * scale)) / 2.0;
        double offsetY = (getHeight() - (VIRTUAL_HEIGHT * scale)) / 2.0;

        g2.translate(offsetX, offsetY);
        g2.scale(scale, scale);
        g2.setClip(0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT);

        // 1. Midfield Line
        g2.setColor(new Color(255, 255, 255, 30));
        for (int i = 0; i < VIRTUAL_HEIGHT; i += 5) {
            g2.fillRect(VIRTUAL_WIDTH / 2, i, 1, 3);
        }

        // 2. Left Paddle
        g2.setColor(new Color(0, 240, 255));
        g2.fillRect(LEFT_PADDLE_X, (int) leftY, PADDLE_WIDTH, PADDLE_HEIGHT);

        // 3. Right Paddle
        g2.setColor(new Color(255, 0, 150));
        g2.fillRect(RIGHT_PADDLE_X, (int) rightY, PADDLE_WIDTH, PADDLE_HEIGHT);

        // 4. Ball
        g2.setColor(Color.WHITE);
        g2.fillRect((int) ballX, (int) ballY, BALL_SIZE, BALL_SIZE);

        // 5. Dashboard
        g2.setFont(new Font("SansSerif", Font.BOLD, 10));
        String score = leftScore + " : " + rightScore;
        g2.drawString(score, (VIRTUAL_WIDTH - g2.getFontMetrics().stringWidth(score)) / 2, 14);

        g2.setFont(new Font("SansSerif", Font.PLAIN, 4));
        if (isTrainingHeadless) {
            g2.setColor(Color.YELLOW);
            g2.drawString("TRAINING...", 4, VIRTUAL_HEIGHT - 4);
        } else {
            g2.drawString("ε: " + String.format("%.2f", epsilon) + " (SPACE: Fast Train)", 4, VIRTUAL_HEIGHT - 4);
        }

        g2.dispose();
    }
}