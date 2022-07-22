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

package com.zaxxio.network.config.step;


import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.MultiLayerConfiguration;
import com.zaxxio.network.model.Layer;
import com.zaxxio.network.model.Neuron;
import com.zaxxio.network.server.MultiLayerNetworkView;
import com.zaxxio.network.weight.WeightInit;
import fi.iki.elonen.NanoHTTPD;
import lombok.Getter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Getter
public class ConfigurationBuilder implements WeightInitConfigStep, MomentumConfigStep, OptimizationAlgoConfigStep, ServerConfigStep, MaxEpochConfigStep, MinErrorConfigStep, UpdaterConfigStep, LayerConfigStep ,ConfigurationBuilderStep {

    public static ActivationFunction activationFunction;
    private final List<Layer> layerList;
    private final List<Neuron> inputLayer;
    private final List<List<Neuron>> hiddenLayers;
    private final List<Neuron> outputLayer;

    private WeightInit weightInit;
    private double momentum;
    private OptimizationAlgo optimizationAlgo;
    private boolean serverEnabled;
    private double maxEpoch;
    private double minError;
    private double updater;
    private MultiLayerNetworkView network;

    public ConfigurationBuilder(ActivationFunction acFunc) {
        activationFunction = acFunc;
        this.layerList = new ArrayList<>();
        this.inputLayer = new ArrayList<>();
        this.hiddenLayers = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
    }

    @Override
    public OptimizationAlgoConfigStep weightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    @Override
    public UpdaterConfigStep momentum(double momentum) {
        this.momentum = momentum;
        return this;
    }

    @Override
    public MomentumConfigStep optimizationAlgo(OptimizationAlgo optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
        return this;
    }

    @Override
    public ConfigurationBuilderStep server(boolean enabled) {
        this.serverEnabled = enabled;
        if (enabled) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < layerList.size() - 1; i++) {
                builder.append(layerList.get(i).getnIn()).append(",");
            }
            builder.append(layerList.get(layerList.size() - 1).getnOut());
            MultiLayerNetworkView.DATA_NETWORK = builder.toString();
            network = new MultiLayerNetworkView(8080);
            try {
                network.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return this;
    }

    @Override
    public MultiLayerConfiguration build() {
        return new MultiLayerConfiguration(this);
    }

    @Override
    public MinErrorConfigStep maxEpoch(double maxEpoch) {
        this.maxEpoch = maxEpoch;
        return this;
    }

    @Override
    public ServerConfigStep minError(double minError) {
        this.minError = minError;
        return this;
    }

    @Override
    public LayerConfigStep updater(double updater) {
        this.updater = updater;
        return this;
    }

    @Override
    public LayerConfigStep layer(int index, Layer layer) {
        this.layerList.add(index, layer);
        return this;
    }

    @Override
    public MaxEpochConfigStep list() {
        buildNetwork();
        return this;
    }

    private void buildNetwork() {

    }

}
