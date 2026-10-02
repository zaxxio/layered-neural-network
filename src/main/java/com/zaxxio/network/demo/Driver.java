package com.zaxxio.network.demo;

import com.zaxxio.network.MultiLayerNetwork;
import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.MultiLayerConfiguration;
import com.zaxxio.network.config.step.OptimizationAlgo;
import com.zaxxio.network.model.DenseLayer;
import com.zaxxio.network.model.InputLayer;
import com.zaxxio.network.model.OutputLayer;
import com.zaxxio.network.weight.WeightInit;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import java.util.Random;

public class Driver {

    private static final double GAMMA = 0.95; // Q-learning discount factor
    private static double epsilon = 0.5;      // Exploration rate

    public static void main(String[] args) {

        // Network configured for 3 Inputs and 3 Outputs (Linear Q-value outputs)
        MultiLayerConfiguration config =
                new MultiLayerConfiguration.Builder()
                        .activation(ActivationFunction.LEAKY_RELU)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(
                                OptimizationAlgo.STOCHASTIC_GRADIENT_DESCENT
                        )
                        .momentum(0.4)
                        .updater(0.005)

                        // Input Layer: 3 Features (e.g., ballX, ballY, paddleY)
                        .layer(
                                0,
                                new InputLayer.Builder()
                                        .nIn(3)
                                        .build()
                        )

                        // Hidden Layer 1
                        .layer(
                                1,
                                new DenseLayer.Builder()
                                        .nIn(16)
                                        .activation(
                                                ActivationFunction.LEAKY_RELU
                                        )
                                        .build()
                        )

                        // Hidden Layer 2
                        .layer(
                                2,
                                new DenseLayer.Builder()
                                        .nIn(16)
                                        .activation(
                                                ActivationFunction.LEAKY_RELU
                                        )
                                        .build()
                        )

                        // Output Layer: 3 Actions [0: UP, 1: STAY, 2: DOWN]
                        .layer(
                                3,
                                new OutputLayer.Builder()
                                        .nOut(3)
                                        .activation(
                                                ActivationFunction.IDENTITY // Mandatory for Q-values
                                        )
                                        .build()
                        )

                        .list()
                        .minError(0.001)
                        .realtime(true)
                        .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        SwingUtilities.invokeLater(() -> {
            NeuralNetworkPanel panel = new NeuralNetworkPanel(model, "DQN Test Simulation (3 Inputs -> 3 Outputs)");

            JFrame frame = new JFrame("Zaxxio Neural Network - DQN Driver Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(panel);
            frame.setSize(1400, 900);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            model.setRealtimeListener((network, event) -> {
                panel.setRealtimeEvent(event);
                panel.repaint();
            });

            // Online Reinforcement Learning simulation thread
            Thread dqnThread = new Thread(() -> {
                Random random = new Random();
                int totalSteps = 10_000;

                System.out.println("==========================================");
                System.out.println("STARTING ONLINE DQN SIMULATION (3 in -> 3 out)");
                System.out.println("==========================================");

                for (int step = 1; step <= totalSteps; step++) {

                    // 1. Generate a synthetic state vector s (3 inputs)
                    double[] state = new double[]{
                            random.nextDouble(), // e.g., Ball X
                            random.nextDouble(), // e.g., Ball Y
                            random.nextDouble()  // e.g., Paddle Y
                    };

                    // 2. Forward pass to predict Q-values for 3 actions
                    double[] currentQ = model.predict(state);

                    // 3. Pick action via epsilon-greedy
                    int action;
                    if (random.nextDouble() < epsilon) {
                        action = random.nextInt(3); // Explore
                    } else {
                        action = argmax(currentQ);  // Exploit
                    }

                    // 4. Simulate environment transition & dummy reward shaping
                    // Reward rule: Give +1.0 if paddle (input[2]) is close to ball (input[1]), else -0.5
                    double distance = Math.abs(state[1] - state[2]);
                    double reward = (distance < 0.2) ? 1.0 : -0.5;

                    // 5. Generate next synthetic state s'
                    double[] nextState = new double[]{
                            random.nextDouble(),
                            random.nextDouble(),
                            random.nextDouble()
                    };

                    // 6. Compute Bellman Q-Target: targetQ = r + gamma * max(Q(s'))
                    double[] nextQ = model.predict(nextState);
                    double targetQ = reward + GAMMA * max(nextQ);

                    // 7. Update target vector for the chosen action index
                    double[] targetVector = currentQ.clone();
                    targetVector[action] = targetQ;

                    // 8. Single online SGD update step
                    model.forward(state);
                    model.backward(targetVector);

                    // Decay epsilon gradually
                    epsilon = Math.max(0.05, epsilon * 0.999);

                    // Throttle execution speed slightly for GUI rendering smooth visualization
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }

                    if (step % 500 == 0) {
                        System.out.printf("Step %d/%d | Epsilon: %.3f | Q-values: [%.3f, %.3f, %.3f]%n",
                                step, totalSteps, epsilon, currentQ[0], currentQ[1], currentQ[2]);
                    }
                }

                System.out.println("\nSimulation completed successfully!");

            }, "dqn-simulation-thread");

            dqnThread.setDaemon(true);
            dqnThread.start();
        });
    }

    private static int argmax(double[] array) {
        int bestIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[bestIdx]) bestIdx = i;
        }
        return bestIdx;
    }

    private static double max(double[] array) {
        double maxVal = array[0];
        for (double v : array) {
            if (v > maxVal) maxVal = v;
        }
        return maxVal;
    }
}