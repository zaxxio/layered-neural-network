package com.zaxxio.network.demo;

import com.zaxxio.network.MultiLayerNetwork;
import com.zaxxio.network.listener.RealtimeEvent;
import com.zaxxio.network.model.Neuron;
import com.zaxxio.network.model.Synapse;

import javax.swing.JPanel;
import javax.swing.ToolTipManager;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeuralNetworkPanel extends JPanel {

    // --- Light Mode Theme Palette ---
    private static final Color CANVAS_BG = Color.WHITE;                     // #FFFFFF
    private static final Color PANEL_SURFACE = new Color(248, 250, 252);     // #F8FAFC
    private static final Color PANEL_BORDER = new Color(226, 232, 240);      // #E2E8F0
    private static final Color TEXT_PRIMARY = new Color(15, 23, 42);         // #0F172A
    private static final Color TEXT_MUTED = new Color(100, 116, 139);        // #64748B

    // Light Theme Vibrant Accents
    private static final Color ACCENT_POSITIVE = new Color(37, 99, 235);     // #2563EB (Indigo / Positive)
    private static final Color ACCENT_NEGATIVE = new Color(225, 29, 72);     // #E11D48 (Rose / Negative)
    private static final Color ACCENT_HOVER = new Color(59, 130, 246);       // #3B82F6 (Hover Highlight)
    private static final Color BADGE_BG = new Color(241, 245, 249);         // #F1F5F9 Badge Background

    // --- Typography Cache ---
    private static final Font FONT_TITLE = new Font("SansSerif", Font.BOLD, 17);
    private static final Font FONT_SUBTITLE = new Font("SansSerif", Font.PLAIN, 12);
    private static final Font FONT_SECTION = new Font("SansSerif", Font.BOLD, 12);
    private static final Font FONT_BADGE = new Font("SansSerif", Font.BOLD, 10);
    private static final Font FONT_HEADER = new Font("SansSerif", Font.BOLD, 10);
    private static final Font FONT_SUBHEADER = new Font("SansSerif", Font.PLAIN, 9);
    private static final Font FONT_FOOTER = new Font("SansSerif", Font.PLAIN, 10);
    private static final Font FONT_NODE_LARGE = new Font("Monospaced", Font.BOLD, 11);
    private static final Font FONT_NODE_SMALL = new Font("Monospaced", Font.BOLD, 9);
    private static final Font FONT_EMPTY = new Font("SansSerif", Font.PLAIN, 14);

    // --- Animation Constants ---
    private static final long PULSE_DURATION_MS = 600;

    private final MultiLayerNetwork model;
    private final String title;

    private volatile RealtimeEvent realtimeEvent;
    private volatile long eventTime;

    // --- Interactive State ---
    private Neuron hoveredNeuron = null;
    private final Map<Neuron, Shape> neuronHitboxes = new HashMap<>();

    // --- Layout Caching ---
    private List<List<Neuron>> cachedNetwork = new ArrayList<>();
    private List<List<Point>> cachedPositions = new ArrayList<>();
    private Map<Neuron, Point> neuronPositionMap = new HashMap<>();
    private int[] cachedLayerX = new int[0];
    private boolean layoutDirty = true;

    public NeuralNetworkPanel(MultiLayerNetwork model, String title) {
        this.model = model;
        this.title = (title == null || title.isBlank()) ? "Neural Network" : title;

        setBackground(CANVAS_BG);
        setOpaque(true);

        ToolTipManager.sharedInstance().registerComponent(this);
        ToolTipManager.sharedInstance().setInitialDelay(100);

        initMouseListeners();

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                layoutDirty = true;
            }
        });

        // 60 FPS refresh timer for fluid real-time pulse animation
        new javax.swing.Timer(16, e -> repaint()).start();
    }

    private void initMouseListeners() {
        MouseAdapter adapter = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                Neuron previousHover = hoveredNeuron;
                hoveredNeuron = null;

                for (Map.Entry<Neuron, Shape> entry : neuronHitboxes.entrySet()) {
                    if (entry.getValue().contains(e.getPoint())) {
                        hoveredNeuron = entry.getKey();
                        break;
                    }
                }

                if (hoveredNeuron != previousHover) {
                    repaint();
                }
            }

            @Override
            public void mouseExited(MouseEvent e) {
                if (hoveredNeuron != null) {
                    hoveredNeuron = null;
                    repaint();
                }
            }
        };

        addMouseListener(adapter);
        addMouseMotionListener(adapter);
    }

    @Override
    public String getToolTipText(MouseEvent event) {
        if (hoveredNeuron == null) return null;

        StringBuilder sb = new StringBuilder("<html><body style='background-color:#FFFFFF; color:#0F172A; padding:6px; font-family:sans-serif; border: 1px solid #CBD5E1;'>");
        sb.append("<b style='color:#2563EB;'>Neuron Inspector</b><br/>");
        sb.append("Activation: <code style='color:#059669;'>").append(String.format("%.4f", hoveredNeuron.getOutput())).append("</code><br/>");
        sb.append("Incoming Synapses: ").append(hoveredNeuron.getIncomingSynapses().size()).append("<br/>");

        if (!hoveredNeuron.getIncomingSynapses().isEmpty()) {
            sb.append("<hr style='border: 0.5px solid #E2E8F0;'/><b>Synaptic Weights:</b><br/>");
            int count = 0;
            for (Synapse s : hoveredNeuron.getIncomingSynapses()) {
                if (count++ >= 5) {
                    sb.append("<i style='color:#64748B;'>... +").append(hoveredNeuron.getIncomingSynapses().size() - 5).append(" more</i>");
                    break;
                }
                String wColor = s.getSynapticWeight() >= 0 ? "#2563EB" : "#E11D48";
                sb.append("&bull; Weight: <code style='color:").append(wColor).append(";'>").append(String.format("%.3f", s.getSynapticWeight())).append("</code><br/>");
            }
        }
        sb.append("</body></html>");
        return sb.toString();
    }

    public void setRealtimeEvent(RealtimeEvent realtimeEvent) {
        this.realtimeEvent = realtimeEvent;
        this.eventTime = System.currentTimeMillis();
    }

    public void markLayoutDirty() {
        this.layoutDirty = true;
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g = (Graphics2D) graphics.create();

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        int width = getWidth();
        int height = getHeight();

        if (layoutDirty) {
            rebuildLayoutCache(width, height);
            layoutDirty = false;
        }

        drawHeader(g, width);
        drawNetwork(g, width, height);
        drawFooter(g, width, height);

        g.dispose();
    }

    private void rebuildLayoutCache(int width, int height) {
        cachedNetwork = fetchNetworkLayers();
        cachedPositions.clear();
        neuronPositionMap.clear();
        neuronHitboxes.clear();

        if (cachedNetwork.isEmpty()) {
            cachedLayerX = new int[0];
            return;
        }

        int layerCount = cachedNetwork.size();
        cachedLayerX = new int[layerCount];

        int top = 115;
        int bottom = Math.max(top + 100, height - 100);
        int left = 100;
        int right = Math.max(left + 100, width - 100);

        for (int i = 0; i < layerCount; i++) {
            cachedLayerX[i] = (layerCount == 1) ? width / 2 : left + (right - left) * i / (layerCount - 1);
        }

        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++) {
            List<Neuron> neurons = cachedNetwork.get(layerIndex);
            int count = neurons.size();
            List<Point> points = new ArrayList<>(count);
            boolean isInput = (layerIndex == 0);
            boolean isOutput = (layerIndex == layerCount - 1);

            for (int i = 0; i < count; i++) {
                int y = (count == 1) ? (top + bottom) / 2 : top + (bottom - top) * i / (count - 1);
                Point pt = new Point(cachedLayerX[layerIndex], y);
                points.add(pt);

                Neuron neuron = neurons.get(i);
                neuronPositionMap.put(neuron, pt);

                double radius = isOutput ? 26.0 : (isInput ? 22.0 : 19.0);
                neuronHitboxes.put(neuron, new Ellipse2D.Double(pt.x - radius, pt.y - radius, radius * 2, radius * 2));
            }
            cachedPositions.add(points);
        }
    }

    private List<List<Neuron>> fetchNetworkLayers() {
        List<List<Neuron>> network = new ArrayList<>();
        if (model.getInputLayer() != null && !model.getInputLayer().isEmpty()) {
            network.add(model.getInputLayer());
        }
        if (model.getHiddenLayers() != null) {
            network.addAll(model.getHiddenLayers());
        }
        if (model.getOutputLayer() != null && !model.getOutputLayer().isEmpty()) {
            network.add(model.getOutputLayer());
        }
        return network;
    }

    private void drawHeader(Graphics2D g, int width) {
        g.setColor(PANEL_SURFACE);
        g.fillRect(0, 0, width, 72);
        g.setColor(PANEL_BORDER);
        g.drawLine(0, 71, width, 71);

        g.setFont(FONT_TITLE);
        g.setColor(TEXT_PRIMARY);
        g.drawString("ZAXXIO", 24, 30);

        g.setFont(FONT_SUBTITLE);
        g.setColor(TEXT_MUTED);
        g.drawString(" / Neural Network Inspector", 91, 30);

        g.setFont(FONT_SECTION);
        g.setColor(TEXT_PRIMARY);
        g.drawString(title, 24, 52);

        drawStatus(g, width);
    }

    private void drawStatus(Graphics2D g, int width) {
        RealtimeEvent event = realtimeEvent;
        String status = (event == RealtimeEvent.FORWARD) ? "FORWARD" : (event == RealtimeEvent.BACKWARD) ? "BACKWARD" : "READY";
        Color statusColor = (event == RealtimeEvent.FORWARD) ? ACCENT_POSITIVE : (event == RealtimeEvent.BACKWARD) ? ACCENT_NEGATIVE : TEXT_MUTED;

        int boxWidth = 96;
        int boxHeight = 28;
        int x = width - boxWidth - 24;
        int y = 21;

        g.setColor(BADGE_BG);
        g.fillRoundRect(x, y, boxWidth, boxHeight, 8, 8);
        g.setColor(PANEL_BORDER);
        g.drawRoundRect(x, y, boxWidth, boxHeight, 8, 8);

        g.setColor(statusColor);
        g.fillOval(x + 10, y + 10, 7, 7);

        g.setFont(FONT_BADGE);
        g.setColor(TEXT_PRIMARY);
        g.drawString(status, x + 24, y + 18);
    }

    private void drawNetwork(Graphics2D g, int width, int height) {
        if (cachedNetwork.isEmpty()) {
            drawEmptyState(g, width, height);
            return;
        }

        drawConnections(g);
        drawAnimatedPulses(g);
        drawNeurons(g);
        drawLayerHeaders(g);
    }

    private void drawConnections(Graphics2D g) {
        for (int layerIndex = 1; layerIndex < cachedNetwork.size(); layerIndex++) {
            List<Neuron> currentLayer = cachedNetwork.get(layerIndex);

            for (int neuronIndex = 0; neuronIndex < currentLayer.size(); neuronIndex++) {
                Neuron targetNeuron = currentLayer.get(neuronIndex);
                Point to = cachedPositions.get(layerIndex).get(neuronIndex);

                for (Synapse synapse : targetNeuron.getIncomingSynapses()) {
                    Neuron sourceNeuron = synapse.getFromNeuron();
                    Point from = neuronPositionMap.get(sourceNeuron);

                    if (from != null) {
                        boolean isRelated = isSynapseConnectedToHover(sourceNeuron, targetNeuron);
                        drawLightConnection(g, from, to, synapse.getSynapticWeight(), isRelated);
                    }
                }
            }
        }
    }

    private boolean isSynapseConnectedToHover(Neuron source, Neuron target) {
        if (hoveredNeuron == null) return false;
        if (hoveredNeuron.equals(source) || hoveredNeuron.equals(target)) return true;

        for (Synapse s : hoveredNeuron.getIncomingSynapses()) {
            if (s.getFromNeuron().equals(source) && hoveredNeuron.equals(target)) return true;
        }
        return false;
    }

    private void drawLightConnection(Graphics2D g, Point from, Point to, double weight, boolean highlighted) {
        double magnitude = Math.min(1.0, Math.abs(weight));
        Color base = highlighted ? ACCENT_HOVER : (weight >= 0) ? ACCENT_POSITIVE : ACCENT_NEGATIVE;

        int alpha;
        if (hoveredNeuron != null) {
            alpha = highlighted ? 230 : 20;
        } else {
            alpha = 35 + (int) (110 * magnitude);
        }

        Line2D line = new Line2D.Double(from.x, from.y, to.x, to.y);

        // Highlight Glow Pass
        if (highlighted) {
            g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), 40));
            g.setStroke(new BasicStroke(5.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            g.draw(line);
        }

        // Main Stroke Pass
        g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha));
        g.setStroke(new BasicStroke((float) (highlighted ? 2.5 : (0.8 + magnitude * 1.8))));
        g.draw(line);
    }

    private void drawAnimatedPulses(Graphics2D g) {
        long elapsed = System.currentTimeMillis() - eventTime;
        if (realtimeEvent == null || elapsed > PULSE_DURATION_MS) return;

        double progress = (double) elapsed / PULSE_DURATION_MS;
        if (realtimeEvent == RealtimeEvent.BACKWARD) {
            progress = 1.0 - progress;
        }

        Color pulseColor = (realtimeEvent == RealtimeEvent.FORWARD) ? ACCENT_POSITIVE : ACCENT_NEGATIVE;

        for (int layerIndex = 1; layerIndex < cachedNetwork.size(); layerIndex++) {
            List<Neuron> currentLayer = cachedNetwork.get(layerIndex);

            for (int neuronIndex = 0; neuronIndex < currentLayer.size(); neuronIndex++) {
                Neuron neuron = currentLayer.get(neuronIndex);
                Point to = cachedPositions.get(layerIndex).get(neuronIndex);

                for (Synapse synapse : neuron.getIncomingSynapses()) {
                    Point from = neuronPositionMap.get(synapse.getFromNeuron());
                    if (from != null) {
                        double px = from.x + (to.x - from.x) * progress;
                        double py = from.y + (to.y - from.y) * progress;

                        // Soft Pulse Ring
                        g.setColor(new Color(pulseColor.getRed(), pulseColor.getGreen(), pulseColor.getBlue(), 60));
                        g.fill(new Ellipse2D.Double(px - 7, py - 7, 14, 14));

                        // Solid Center
                        g.setColor(pulseColor);
                        g.fill(new Ellipse2D.Double(px - 3.5, py - 3.5, 7, 7));
                    }
                }
            }
        }
    }

    private void drawNeurons(Graphics2D g) {
        int layerCount = cachedNetwork.size();
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++) {
            List<Neuron> neurons = cachedNetwork.get(layerIndex);
            boolean isInput = (layerIndex == 0);
            boolean isOutput = (layerIndex == layerCount - 1);

            for (int i = 0; i < neurons.size(); i++) {
                Neuron neuron = neurons.get(i);
                drawNeuron(g, neuron, cachedPositions.get(layerIndex).get(i), isInput, isOutput);
            }
        }
    }

    private void drawNeuron(Graphics2D g, Neuron neuron, Point point, boolean input, boolean output) {
        double radius = output ? 26.0 : (input ? 22.0 : 19.0);
        double diameter = radius * 2.0;
        double activation = neuron.getOutput();
        boolean isHovered = neuron.equals(hoveredNeuron);

        double x = point.x - radius;
        double y = point.y - radius;

        Color accentColor = isHovered ? ACCENT_HOVER : (activation >= 0 ? ACCENT_POSITIVE : ACCENT_NEGATIVE);

        // 1. Hover Glow Ring
        if (isHovered) {
            g.setColor(new Color(59, 130, 246, 35));
            g.fill(new Ellipse2D.Double(x - 8, y - 8, diameter + 16, diameter + 16));
            g.setColor(new Color(59, 130, 246, 70));
            g.fill(new Ellipse2D.Double(x - 4, y - 4, diameter + 8, diameter + 8));
        }

        // 2. Pure White Surface Base
        g.setColor(CANVAS_BG);
        g.fill(new Ellipse2D.Double(x, y, diameter, diameter));

        // 3. Inner Structural Track Ring
        double trackMargin = 3.5;
        double trackDiameter = diameter - (trackMargin * 2.0);
        g.setColor(PANEL_SURFACE);
        g.setStroke(new BasicStroke(2.0f));
        g.draw(new Ellipse2D.Double(x + trackMargin, y + trackMargin, trackDiameter, trackDiameter));

        // 4. Dynamic Activation Core Fill
        double absValue = Math.min(1.0, Math.abs(activation));
        double minCoreRadius = 4.0;
        double maxCoreRadius = radius - 5.0;
        double coreRadius = minCoreRadius + (maxCoreRadius - minCoreRadius) * absValue;
        double coreDiameter = coreRadius * 2.0;

        g.setColor(new Color(accentColor.getRed(), accentColor.getGreen(), accentColor.getBlue(), 35));
        g.fill(new Ellipse2D.Double(point.x - coreRadius, point.y - coreRadius, coreDiameter, coreDiameter));

        g.setColor(accentColor);
        double centerDotDiameter = Math.max(3.0, coreRadius * 0.5);
        g.fill(new Ellipse2D.Double(point.x - (centerDotDiameter / 2.0), point.y - (centerDotDiameter / 2.0), centerDotDiameter, centerDotDiameter));

        // 5. Outer Border Outline
        g.setColor(isHovered ? ACCENT_HOVER : PANEL_BORDER);
        g.setStroke(new BasicStroke(isHovered ? 2.0f : 1.2f));
        g.draw(new Ellipse2D.Double(x, y, diameter, diameter));

        // 6. Centered Dark Typography
        String value = String.format("%.2f", activation);
        g.setFont(output ? FONT_NODE_LARGE : FONT_NODE_SMALL);
        FontMetrics metrics = g.getFontMetrics();

        int textWidth = metrics.stringWidth(value);
        int textHeight = metrics.getAscent() - metrics.getDescent();
        float textX = (float) (point.x - (textWidth / 2.0));
        float textY = (float) (point.y + (textHeight / 2.0) - 1);

        g.setColor(TEXT_PRIMARY);
        g.drawString(value, textX, textY);
    }

    private void drawLayerHeaders(Graphics2D g) {
        int layerCount = cachedNetwork.size();
        for (int i = 0; i < layerCount; i++) {
            String label = (i == 0) ? "INPUT" : (i == layerCount - 1) ? "OUTPUT" : "HIDDEN " + i;
            String count = cachedNetwork.get(i).size() + " neurons";

            g.setFont(FONT_HEADER);
            FontMetrics metrics = g.getFontMetrics();
            g.setColor(TEXT_PRIMARY);
            g.drawString(label, cachedLayerX[i] - metrics.stringWidth(label) / 2, 92);

            g.setFont(FONT_SUBHEADER);
            metrics = g.getFontMetrics();
            g.setColor(TEXT_MUTED);
            g.drawString(count, cachedLayerX[i] - metrics.stringWidth(count) / 2, 106);
        }
    }

    private void drawFooter(Graphics2D g, int width, int height) {
        int y = height - 45;
        g.setColor(PANEL_BORDER);
        g.drawLine(0, y - 15, width, y - 15);

        g.setFont(FONT_FOOTER);
        int x = 24;

        drawFooterItem(g, x, y, ACCENT_POSITIVE, "Positive weight");
        x += 125;
        drawFooterItem(g, x, y, ACCENT_NEGATIVE, "Negative weight");
        x += 125;

        g.setColor(TEXT_MUTED);
        g.drawString("Node value = activation", x, y);

        RealtimeEvent event = realtimeEvent;
        if (event != null) {
            boolean isForward = (event == RealtimeEvent.FORWARD);
            drawFooterItem(g, x + 170, y, isForward ? ACCENT_POSITIVE : ACCENT_NEGATIVE, isForward ? "Forward propagation" : "Backward propagation");
        }
    }

    private void drawFooterItem(Graphics2D g, int x, int y, Color color, String text) {
        g.setColor(color);
        g.fillOval(x, y - 7, 7, 7);
        g.setColor(TEXT_MUTED);
        g.drawString(text, x + 13, y);
    }

    private void drawEmptyState(Graphics2D g, int width, int height) {
        String text = "Network not initialized";
        g.setFont(FONT_EMPTY);
        g.setColor(TEXT_MUTED);
        FontMetrics metrics = g.getFontMetrics();
        g.drawString(text, (width - metrics.stringWidth(text)) / 2, height / 2);
    }
}