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
import ml.shifu.shifu.udf.norm.PrecisionType;

/**
 * {@link Predict} wrappers classification and regression result in {@link #predict}, for classification
 * {@link #classValue} denotes the classified category, for regression no meaning for {@link #classValue},
 * {@link #predict} is the probability used.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Predict implements Bytable {

    /**
     * Regression predict value no matter classification or regression tree.
     */
    private double predict;

    /**
     * Classification result, only for classification. 'byte' is ok for index of classes. No more than 127 classes.
     */
    private byte classValue;

    public Predict() {
    }

    public Predict(double predict) {
        this.predict = predict;
    }

    public Predict(double predict, byte classValue) {
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
    public byte getClassValue() {
        return classValue;
    }

    public void write(DataOutput out, PrecisionType pt) throws IOException {
        switch(pt) {
            case FLOAT32:
                out.writeFloat((float) predict);
                break;
            case FLOAT16:
                out.writeFloat((float) PrecisionType.FLOAT16.to(predict));
                break;
            case DOUBLE64:
                out.writeDouble(predict);
                break;
            default:
                throw new UnsupportedOperationException("Not supported precision");
        }
        out.writeByte(classValue);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        this.write(out, PrecisionType.DOUBLE64);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        predict = in.readDouble();
        classValue = in.readByte();
    }

    public void readFields(DataInput in, int version) throws IOException {
        predict = in.readDouble();
        classValue = in.readByte();
    }

    public void readFields(DataInput in, int version, PrecisionType pt) throws IOException {
        switch(pt) {
            case FLOAT32:
                predict = in.readFloat();
                break;
            case FLOAT16:
                predict = in.readFloat();
                break;
            case DOUBLE64:
                predict = in.readDouble();
                break;
            default:
                throw new UnsupportedOperationException("Not supported precision");
        }
        classValue = in.readByte();
    }

    @Override
    public String toString() {
        return "Predict [predict=" + predict + ", classValue=" + classValue + "]";
    }

}
