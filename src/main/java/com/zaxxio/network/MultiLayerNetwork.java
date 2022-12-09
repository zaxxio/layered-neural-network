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
import com.zaxxio.network.model.Layer;
import com.zaxxio.network.model.Neuron;
import com.zaxxio.network.weight.impl.WeightInitializerImpl;
import lombok.Getter;
import lombok.Setter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

@Getter
@Setter
public class MultiLayerNetwork implements Serializable {

    private static final Logger logger = LogManager.getLogger(MultiLayerNetwork.class);

    private MultiLayerConfiguration config;
    private double error = 1;
    private ScoreListener scoreListener;
    private ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private Random random = new Random();

    private List<Neuron> inputLayer;
    private List<List<Neuron>> hiddenLayers;
    private List<Neuron> outputLayer;
    private List<Layer> layers;

    public MultiLayerNetwork(MultiLayerConfiguration config) {
        this.config = config;
        if (config.isServer()) {
            logger.info("Server running : http://localhost:8080");
        }
    }

    public void init() {
        this.layers = config.getLayers();
        this.inputLayer = new ArrayList<>();
        this.hiddenLayers = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                for (int x = 0; x < layers.get(i).getnIn(); x++) {
                    this.inputLayer.add(new Neuron());
                }
            } else if (i < layers.size() - 1) {
                if (i == 1) {
                    List<Neuron> hiddenLayer = new ArrayList<>();
                    for (int x = 0; x < layers.get(i).getnIn(); x++) {
                        int nIn = layers.get(i - 1).getnIn();
                        int nOut = layers.get(i).getnOut();
                        IActivationFunction function = layers.get(i).getActivationFunctionInLayer();
                        ActivationFunction functionEnum = layers.get(i).getActivationFunction();
                        WeightInitializerImpl initializer = new WeightInitializerImpl(nIn, nOut, config.getWeightInit());
                        hiddenLayer.add(new Neuron(this.inputLayer, function, functionEnum, initializer));
                    }
                    hiddenLayers.add(hiddenLayer);
                } else {
                    List<Neuron> hiddenLayer = new ArrayList<>();
                    for (int x = 0; x < layers.get(i).getnIn(); x++) {
                        int nOut = layers.get(i).getnOut();
                        IActivationFunction function = layers.get(i).getActivationFunctionInLayer();
                        int nIn = layers.get(i - 1).getnIn();
                        ActivationFunction functionEnum = layers.get(i).getActivationFunction();
                        WeightInitializerImpl initializer = new WeightInitializerImpl(nIn, nOut, config.getWeightInit());
                        hiddenLayer.add(new Neuron(this.hiddenLayers.get(i - 2), function, functionEnum, initializer));
                    }
                    hiddenLayers.add(hiddenLayer);
                }
            } else {
                for (int x = 0; x < layers.get(i).getnOut(); x++) {
                    int nOut = layers.get(i).getnOut();
                    int nIn = layers.get(i - 1).getnIn();
                    IActivationFunction function = layers.get(i).getActivationFunctionInLayer();
                    ActivationFunction functionEnum = layers.get(i).getActivationFunction();
                    WeightInitializerImpl initializer = new WeightInitializerImpl(nIn, nOut, config.getWeightInit());
                    this.outputLayer.add(new Neuron(hiddenLayers.get(hiddenLayers.size() - 1), function, functionEnum, initializer));
                }
            }
        }
        logger.info("MultiLayerNetwork Initialized.");
    }

    private double calcError(double[] targets) {
        AtomicInteger i = new AtomicInteger();
        return getOutputLayer().stream().mapToDouble(neuron -> Math.abs(neuron.error(targets[i.getAndIncrement()]))).sum();
    }

    public void backward(double... targets) {
        if (targets.length != getOutputLayer().size()) try {
            throw new NeuralException("Output dimension's does not match.");
        } catch (NeuralException e) {
            e.printStackTrace();
        }
        AtomicInteger i = new AtomicInteger();
        for (Neuron neuron : this.getOutputLayer()) {
            neuron.calculateGradient(targets[i.getAndIncrement()], error);
        }
        Collections.reverse(getHiddenLayers());
        for (List<Neuron> hiddenLayer : getHiddenLayers()) {
            for (Neuron neuron : hiddenLayer) {
                neuron.updateConnections(config.getUpdater(), config.getMomentum());
            }
        }
        Collections.reverse(getHiddenLayers());
        for (Neuron neuron : getOutputLayer()) {
            neuron.updateConnections(config.getUpdater(), config.getMomentum());
        }
    }

    public void forward(double... inputs) {
        if (inputs.length != getInputLayer().size()) try {
            throw new NeuralException("Input dimension's does not match.");
        } catch (NeuralException e) {
            e.printStackTrace();
            System.exit(0);
        }
        int i = 0;
        for (Neuron neuron : getInputLayer()) {
            neuron.setOutput(inputs[i++]);
        }
        for (List<Neuron> hiddenLayer : getHiddenLayers()) {
            for (Neuron neuron : hiddenLayer) {
                neuron.calculateOutput();
            }
        }
        for (Neuron neuron : getOutputLayer()) {
            neuron.calculateOutput();
        }
    }

    public void fit(MLDataSet dataSets) {
        double epochIndex = 0;
        double startTime = System.currentTimeMillis();
        while (config.getMinError() < error) {
            if (config.getOptimizationAlgo() == OptimizationAlgo.STOCHASTIC_GRADIENT_DESCENT) {
                Collections.shuffle(dataSets.getDataList());
            }
            List<Double> errors = new ArrayList<>();
            if (scoreListener != null) {
                if (epochIndex % scoreListener.getStep() == 0) {
                    scoreListener.setEpoch(epochIndex);
                    scoreListener.setError(error);
                    executor.execute(scoreListener);
                }
            }
            if (config.getMaxEpoch() != 0 && epochIndex == config.getMaxEpoch()) break;
            if (error < config.getMinError()) break;
            for (MLData dataSet : dataSets.getDataList()) {
                forward(dataSet.getInputs());
                backward(dataSet.getTargets());
                errors.add(calcError(dataSet.getTargets()));
            }
            OptionalDouble average = errors.stream().mapToDouble(a -> a).average();
            error = average.isPresent() ? average.getAsDouble() : 0;
            epochIndex++;
        }
        long endTime = System.currentTimeMillis();
        logger.info("Training Finished : " + ((endTime - startTime) / 1000) + "s");
        executor.shutdown();
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[getOutputLayer().size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = getOutputLayer().get(i).getOutput();
        }
        return output;
    }

    public double[] predict(MLData inputs) {
        forward(inputs.getInputs());
        double[] output = new double[getOutputLayer().size()];
        for (int i = 0; i < getOutputLayer().size(); i++) {
            output[i] = getOutputLayer().get(i).getOutput();
        }
        return output;
    }

    public void addScoreListener(ScoreListener scoreListener) {
        this.scoreListener = scoreListener;
    }

}
