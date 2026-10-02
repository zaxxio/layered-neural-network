/*
 * MIT License
 *
 * Copyright (c) 2020 Partha Sutradhar.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.zaxxio.network;

import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.activation.IActivationFunction;
import com.zaxxio.network.config.MultiLayerConfiguration;
import com.zaxxio.network.config.ScoreListener;
import com.zaxxio.network.config.step.OptimizationAlgo;
import com.zaxxio.network.data.MLData;
import com.zaxxio.network.data.MLDataSet;
import com.zaxxio.network.listener.RealtimeEvent;
import com.zaxxio.network.listener.RealtimeListener;
import com.zaxxio.network.model.Layer;
import com.zaxxio.network.model.Neuron;
import com.zaxxio.network.weight.impl.WeightInitializerImpl;
import lombok.Getter;
import lombok.Setter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;
import java.util.*;

@Getter
@Setter
public class MultiLayerNetwork implements Serializable {

    private static final Logger logger = LogManager.getLogger(MultiLayerNetwork.class);

    private MultiLayerConfiguration config;
    private transient RealtimeListener realtimeListener;
    private double error = 1.0;
    private ScoreListener scoreListener;
    private Random random = new Random();

    private List<Neuron> inputLayer;
    private List<List<Neuron>> hiddenLayers;
    private List<Neuron> outputLayer;
    private List<Layer> layers;

    public MultiLayerNetwork(MultiLayerConfiguration config) {
        this.config = config;
        if (config.isRealTimeEnabled()) {
            logger.info("Server running : http://localhost:8080");
        }
    }

    public void init() {
        this.layers = config.getLayers();
        this.inputLayer = new ArrayList<>();
        this.hiddenLayers = new ArrayList<>();
        this.outputLayer = new ArrayList<>();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);

            if (i == 0) {
                // Input Layer
                for (int x = 0; x < layer.getnIn(); x++) {
                    this.inputLayer.add(new Neuron());
                }
            } else if (i < layers.size() - 1) {
                // Hidden Layers
                int nIn = layers.get(i - 1).getnIn();
                int nOut = layer.getnOut();
                IActivationFunction function = layer.getActivationFunctionInLayer();
                ActivationFunction functionEnum = layer.getActivationFunction();
                WeightInitializerImpl initializer = new WeightInitializerImpl(nIn, nOut, config.getWeightInit());

                List<Neuron> prevLayer = (i == 1) ? this.inputLayer : this.hiddenLayers.get(i - 2);
                List<Neuron> hiddenLayer = new ArrayList<>();

                for (int x = 0; x < layer.getnIn(); x++) {
                    hiddenLayer.add(new Neuron(prevLayer, function, functionEnum, initializer));
                }
                hiddenLayers.add(hiddenLayer);
            } else {
                // Output Layer
                int nIn = layers.get(i - 1).getnIn();
                int nOut = layer.getnOut();
                IActivationFunction function = layer.getActivationFunctionInLayer();
                ActivationFunction functionEnum = layer.getActivationFunction();
                WeightInitializerImpl initializer = new WeightInitializerImpl(nIn, nOut, config.getWeightInit());

                List<Neuron> prevLayer = hiddenLayers.isEmpty()
                        ? this.inputLayer
                        : hiddenLayers.get(hiddenLayers.size() - 1);

                for (int x = 0; x < nOut; x++) {
                    this.outputLayer.add(new Neuron(prevLayer, function, functionEnum, initializer));
                }
            }
        }
        logger.info("MultiLayerNetwork Initialized.");
    }

    private double calcError(double[] targets) {
        double sum = 0.0;
        for (int i = 0; i < outputLayer.size(); i++) {
            sum += Math.abs(outputLayer.get(i).error(targets[i]));
        }
        return sum;
    }

    public void backward(double... targets) {
        for (int i = 0; i < outputLayer.size(); i++) {
            outputLayer.get(i).calculateGradient(targets[i], error);
        }

        List<List<Neuron>> reversed = new ArrayList<>(hiddenLayers);
        Collections.reverse(reversed);

        for (List<Neuron> layer : reversed) {
            for (Neuron neuron : layer) {
                neuron.calculateGradient();
            }
        }

        for (List<Neuron> layer : hiddenLayers) {
            for (Neuron neuron : layer) {
                neuron.updateConnections(config.getUpdater(), config.getMomentum());
            }
        }

        for (Neuron neuron : outputLayer) {
            neuron.updateConnections(config.getUpdater(), config.getMomentum());
        }

        realtimeUpdate(RealtimeEvent.BACKWARD);
    }

    public void forward(double... inputs) {
        if (inputs.length != getInputLayer().size()) {
            throw new IllegalArgumentException(
                String.format("Input dimension mismatch: expected %d, got %d", getInputLayer().size(), inputs.length)
            );
        }

        for (int i = 0; i < getInputLayer().size(); i++) {
            getInputLayer().get(i).setOutput(inputs[i]);
        }

        for (List<Neuron> hiddenLayer : getHiddenLayers()) {
            for (Neuron neuron : hiddenLayer) {
                neuron.calculateOutput();
            }
        }

        for (Neuron neuron : getOutputLayer()) {
            neuron.calculateOutput();
        }

        realtimeUpdate(RealtimeEvent.FORWARD);
    }

    public void fit(MLDataSet dataSets) {
        long startTime = System.currentTimeMillis();
        for (int epoch = 0; error > config.getMinError(); epoch++) {
            if (config.getOptimizationAlgo() == OptimizationAlgo.STOCHASTIC_GRADIENT_DESCENT) {
                Collections.shuffle(dataSets.getDataList());
            }
            List<Double> errors = new ArrayList<>();
            for (MLData dataSet : dataSets.getDataList()) {
                forward(dataSet.getInputs());
                backward(dataSet.getTargets());
                errors.add(calcError(dataSet.getTargets()));
            }
            error = errors.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            if (config.getMaxEpoch() != 0 && epoch + 1 >= config.getMaxEpoch()) break;
        }
        logger.info(String.format("Training Finished : %.3fs", (System.currentTimeMillis() - startTime) / 1000.0));
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[getOutputLayer().size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = getOutputLayer().get(i).getOutput();
        }
        realtimeListener.onUpdate(this, RealtimeEvent.PREDICT);
        return output;
    }

    public double[] predict(MLData inputs) {
        return predict(inputs.getInputs());
    }

    public void addScoreListener(ScoreListener scoreListener) {
        this.scoreListener = scoreListener;
    }

    public void setRealtimeListener(RealtimeListener realtimeListener) {
        this.realtimeListener = realtimeListener;
    }

    private void realtimeUpdate(RealtimeEvent event) {
        if (config.isRealTimeEnabled() && realtimeListener != null) {
            realtimeListener.onUpdate(this, event);
        }
    }
}