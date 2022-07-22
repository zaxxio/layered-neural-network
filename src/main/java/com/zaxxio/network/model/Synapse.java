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


import com.zaxxio.network.weight.WeightInit;
import com.zaxxio.network.weight.WeightInitializer;
import lombok.Getter;
import lombok.Setter;

import java.util.UUID;

@Getter
@Setter
public class Synapse {

    private UUID synapseId;
    private UUID fromNeuronId;
    private UUID toNeuronId;
    private Neuron fromNeuron;
    private Neuron toNeuron;
    private double synapticWeight;
    private double synapticDeltaWeight = 0.0;
    private double epsilon = 0.000001;

    public Synapse(Neuron fromNeuron, Neuron toNeuron, WeightInitializer weightInitializer) {
        this.synapseId = UUID.randomUUID();
        this.fromNeuron = fromNeuron;
        this.fromNeuronId = fromNeuron.getNeuronId();
        this.toNeuron = toNeuron;
        this.toNeuronId = toNeuron.getNeuronId();
        if(weightInitializer.getWeightInit() == WeightInit.XAVIER){
            this.synapticWeight = weightInitializer.generateSynapticWeight(toNeuron.getActivationFunctionEnum());
        }else {
            this.synapticWeight = weightInitializer.generateSynapticWeight();
        }
    }

    public void updateSynapticWeight(Double synapticWeight) {
        this.synapticWeight += synapticWeight;
    }

}

