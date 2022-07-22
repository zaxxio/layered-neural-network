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

package com.zaxxio.network.config;


import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.step.ActivationFunctionConfigStep;
import com.zaxxio.network.config.step.ConfigurationBuilder;
import com.zaxxio.network.config.step.OptimizationAlgo;
import com.zaxxio.network.model.Layer;
import com.zaxxio.network.weight.WeightInit;
import lombok.Getter;
import lombok.Setter;

import java.util.List;


@Getter
@Setter
public class MultiLayerConfiguration {

    private ActivationFunction activationFunction;
    private WeightInit weightInit;
    private double momentum;
    private OptimizationAlgo optimizationAlgo;
    private boolean server;
    private double maxEpoch;
    private double minError;
    private double updater;

    private List<Layer> layers;



    public MultiLayerConfiguration(ConfigurationBuilder configurationBuilder) {
        this.activationFunction = ConfigurationBuilder.activationFunction;
        this.weightInit = configurationBuilder.getWeightInit();
        this.momentum = configurationBuilder.getMomentum();
        this.optimizationAlgo = configurationBuilder.getOptimizationAlgo();
        this.server = configurationBuilder.isServerEnabled();
        this.maxEpoch = configurationBuilder.getMaxEpoch();
        this.minError = configurationBuilder.getMinError();
        this.updater = configurationBuilder.getUpdater();
        this.layers = configurationBuilder.getLayerList();
    }

    public static class Builder implements ActivationFunctionConfigStep {
        @Override
        public ConfigurationBuilder activation(ActivationFunction activationFunction) {
            return new ConfigurationBuilder(activationFunction);
        }
    }

}
