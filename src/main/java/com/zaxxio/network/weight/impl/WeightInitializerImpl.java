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

package com.zaxxio.network.weight.impl;

import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.utils.RandomGenerator;
import com.zaxxio.network.weight.WeightInit;
import com.zaxxio.network.weight.WeightInitializer;

public class WeightInitializerImpl extends WeightInitializer {

    private final double epsilon = 0.000001;

    public WeightInitializerImpl(int fanIn, int fanOut, WeightInit weightInit) {
        super(fanIn, fanOut, weightInit);
    }

    @Override
    public double generateSynapticWeight(ActivationFunction activationFunction) {
        switch (activationFunction) {
            case RELU:
            case LEAKY_RELU:
            case SWISH:
                double a = Math.sqrt(12.0 / (nIn + nOut));
                return RandomGenerator.randomValue(-a, a) + epsilon;
            case TANH:
                double b = Math.sqrt(6.0 / (nIn + nOut));
                return RandomGenerator.randomValue(-b, b) + epsilon;
            case SIGMOID:
                double c = Math.sqrt(4.0 / (nIn + nOut));
                return RandomGenerator.randomValue(-c, c) + epsilon;
        }
        return Math.random();
    }

    @Override
    public double generateSynapticWeight() {
        return RandomGenerator.randomValue(-2, 2);
    }
}
