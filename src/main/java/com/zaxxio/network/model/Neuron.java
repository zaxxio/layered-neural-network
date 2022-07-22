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

package com.zaxxio.network.model;


import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.activation.IActivationFunction;
import com.zaxxio.network.weight.WeightInitializer;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Getter
@Setter
public class Neuron {

    private UUID neuronId;

    private List<Synapse> incomingSynapses;
    private List<Synapse> outgoingSynapses;

    private double bias;
    private double gradient;
    private double output = 0.0;
    private double outputBeforeActivation = 0.0;
    private WeightInitializer weightInitializer;
    private IActivationFunction activationFunction;
    private ActivationFunction activationFunctionEnum;
    private double epsilon = 0.00001;

    public Neuron() {
        this.neuronId = UUID.randomUUID();
        this.incomingSynapses = new ArrayList<>();
        this.outgoingSynapses = new ArrayList<>();
        this.bias = 1;
    }

    public Neuron(List<Neuron> neurons, IActivationFunction activationFunction, ActivationFunction activationFunctionEnum, WeightInitializer weightInitializer) {
        this();
        this.weightInitializer = weightInitializer;
        this.activationFunction = activationFunction;
        this.activationFunctionEnum = activationFunctionEnum;
        for (Neuron neuron : neurons) {
            Synapse synapse = new Synapse(neuron, this, weightInitializer);
            neuron.getOutgoingSynapses().add(synapse);
            incomingSynapses.add(synapse);
        }
    }


    public void calculateOutput() {
        this.outputBeforeActivation = 0.0;
        for (Synapse synapse : incomingSynapses) {
            outputBeforeActivation += synapse.getSynapticWeight() * synapse.getFromNeuron().getOutput();
        }
        this.output = activationFunction.output(this.outputBeforeActivation + bias);
    }


    public double error(Double target) {
        return (target - output);
    }

    public void calculateGradient() {
        gradient = outgoingSynapses.stream().mapToDouble(synapse -> synapse.getToNeuron().getGradient() * synapse.getSynapticWeight()).sum()
                * activationFunction.outputDerivative(output);
    }

    public void calculateGradient(double target, double error) {
        /*gradient = error(target) * error;*/
        gradient = error(target) * activationFunction.outputDerivative(output);
    }

    public void updateConnections(double lr, double mu) {
        for (Synapse synapse : incomingSynapses) {
            double prevDelta = synapse.getSynapticDeltaWeight();
            synapse.setSynapticDeltaWeight(lr * getGradient() * synapse.getFromNeuron().getOutput());
            synapse.updateSynapticWeight(synapse.getSynapticDeltaWeight() + mu * prevDelta);
        }
    }

}
