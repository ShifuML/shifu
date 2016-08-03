/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Bytable;

/**
 * Predict wrappers classification and regression result in {@link #predict}, for classification {@link #classValue} means the
 * probability, for regression no meaning for {@link #classValue}
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Predict implements Bytable {

    /**
     * Regression predict value no matter classification or regression tree.
     */
    private double predict;

    /**
     * Classification result, only for classification.
     */
    private double classValue;

    public Predict() {
    }

    public Predict(double predict) {
        this.predict = predict;
    }

    public Predict(double predict, double classValue) {
        this.predict = predict;
        this.classValue = classValue;
    }

    /**
     * @return the predict
     */
    public double getPredict() {
        return predict;
    }

    /**
     * @return the prob
     */
    public double getClassValue() {
        return classValue;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeDouble(predict);
        out.writeDouble(classValue);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        predict = in.readDouble();
        classValue = in.readDouble();
    }

    @Override
    public String toString() {
        return "Predict [predict=" + predict + ", classValue=" + classValue + "]";
    }

}
