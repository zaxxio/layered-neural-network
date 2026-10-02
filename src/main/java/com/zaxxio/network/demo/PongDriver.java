package com.zaxxio.network.demo;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

public class PongDriver {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // Moved inside EDT for Swing thread safety
            PongGame pong = new PongGame();

            JFrame pongFrame = new JFrame("Zaxxio Neural Network - Pong AI");
            pongFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            pongFrame.add(pong);
            pongFrame.pack();
            pongFrame.setLocationRelativeTo(null);

            // --- FIX 1: Allow resizing & full screen ---
            pongFrame.setResizable(true);

            // OPTIONAL: Uncomment to open directly in full window mode
            // pongFrame.setExtendedState(JFrame.MAXIMIZED_BOTH);

            pongFrame.setVisible(true);

            // --- Left AI Visualizer ---
            NeuralNetworkPanel leftPanel = new NeuralNetworkPanel(
                    pong.getLeftAI(),
                    "Left Pong AI"
            );

            JFrame leftFrame = new JFrame("Left AI");
            leftFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            leftFrame.add(leftPanel);
            leftFrame.setSize(700, 500);

            // FIX 2: Math.max prevents placing the left frame off-screen
            leftFrame.setLocation(
                    Math.max(0, pongFrame.getX() - 710),
                    pongFrame.getY()
            );
            leftFrame.setVisible(true);

            // --- Right AI Visualizer ---
            NeuralNetworkPanel rightPanel = new NeuralNetworkPanel(
                    pong.getRightAI(),
                    "Right Pong AI"
            );

            JFrame rightFrame = new JFrame("Right AI");
            rightFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            rightFrame.add(rightPanel);
            rightFrame.setSize(700, 500);
            rightFrame.setLocation(
                    pongFrame.getX() + pongFrame.getWidth() + 10,
                    pongFrame.getY()
            );
            rightFrame.setVisible(true);

            // Realtime listeners
            pong.getLeftAI().setRealtimeListener(
                    (network, event) -> leftPanel.setRealtimeEvent(event)
            );

            pong.getRightAI().setRealtimeListener(
                    (network, event) -> rightPanel.setRealtimeEvent(event)
            );

            // Refresh timer
            new Timer(250, e -> {
                leftPanel.repaint();
                rightPanel.repaint();
            }).start();
        });
    }
}