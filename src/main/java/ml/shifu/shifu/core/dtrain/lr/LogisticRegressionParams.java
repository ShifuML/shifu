/*
 * Copyright [2013-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.dtrain.lr;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;

/**
 * A model class to store logistic regression weight on first iteration by using {@link #parameters}, while in other
 * iterations {@link #parameters} is used to store gradients.
 * 
 * <p>
 * To make all workers started at the same model, master will compute a consistent model weights at the first iteration
 * and then send to all the workers. Workers will start computing from the second iteration.
 * 
 * <p>
 * Workers are responsible to compute local accumulated gradients and send to master while master accumulates all
 * gradients together to build a global model.
 */
public class LogisticRegressionParams extends HaltBytable implements Combinable<LogisticRegressionParams> {

    /**
     * Model weights in the first iteration, gradients in other iterations.
     */
    private double[] parameters;

    /**
     * Current test error which can be sent to master
     */
    private double validationError = 0;

    /**
     * Current train error which can be sent to master
     */
    private double trainError = 0;

    /**
     * Weighted wgt training record count in one worker
     */
    private double trainSize;

    /**
     * Weighted wgt testing record count in one worker
     */
    private double validationSize;

    /**
     * Training record count in one worker
     */
    private double trainCount;

    /**
     * Testing record count in one worker
     */
    private double validationCount;

    public LogisticRegressionParams() {
    }

    public LogisticRegressionParams(double[] parameters) {
        this.parameters = parameters;
    }

    public LogisticRegressionParams(double[] parameters, double trainError, double validationError, double trainSize,
            double validationSize, double trainCount, double validationCount) {
        this.parameters = parameters;
        this.trainError = trainError;
        this.validationError = validationError;
        this.trainSize = trainSize;
        this.validationSize = validationSize;
        this.trainCount = trainCount;
        this.validationCount = validationCount;
    }

    public double[] getParameters() {
        return parameters;
    }

    public void setParameters(double[] parameters) {
        this.parameters = parameters;
    }

    /**
     * @return the trainSize
     */
    public double getTrainSize() {
        return trainSize;
    }

    /**
     * @param trainSize
     *            the trainSize to set
     */
    public void setTrainSize(double trainSize) {
        this.trainSize = trainSize;
    }

    @Override
    public LogisticRegressionParams combine(LogisticRegressionParams from) {
        assert from != null;
        this.trainError += from.trainError;
        this.validationError += from.validationError;
        this.trainSize += from.trainSize;
        this.validationSize += from.validationSize;
        this.trainCount += from.trainCount;
        this.validationCount += from.validationCount;
        assert this.parameters != null && from.parameters != null;
        for(int i = 0; i < this.parameters.length; i++) {
            this.parameters[i] += from.parameters[i];
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(parameters == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.parameters.length);
            for(int i = 0; i < this.parameters.length; i++) {
                out.writeDouble(this.parameters[i]);
            }
        }
        out.writeDouble(this.trainError);
        out.writeDouble(this.validationError);
        out.writeDouble(this.trainSize);
        out.writeDouble(this.validationSize);
        out.writeDouble(this.trainCount);
        out.writeDouble(this.validationCount);

    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        int length = in.readInt();
        this.parameters = new double[length];
        for(int i = 0; i < length; i++) {
            this.parameters[i] = in.readDouble();
        }
        this.trainError = in.readDouble();
        this.validationError = in.readDouble();
        this.trainSize = in.readDouble();
        this.validationSize = in.readDouble();
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
    }

    /**
     * @return the trainError
     */
    public double getTrainError() {
        return trainError;
    }

    /**
     * @param trainError
     *            the trainError to set
     */
    public void setTrainError(double trainError) {
        this.trainError = trainError;
    }

    /**
     * @return the trainCount
     */
    public double getTrainCount() {
        return trainCount;
    }

    /**
     * @param trainCount
     *            the trainCount to set
     */
    public void setTrainCount(double trainCount) {
        this.trainCount = trainCount;
    }

    /**
     * @return the validationError
     */
    public double getValidationError() {
        return validationError;
    }

    /**
     * @param validationError
     *            the validationError to set
     */
    public void setValidationError(double validationError) {
        this.validationError = validationError;
    }

    /**
     * @return the validationSize
     */
    public double getValidationSize() {
        return validationSize;
    }

    /**
     * @param validationSize
     *            the validationSize to set
     */
    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    /**
     * @return the validationCount
     */
    public double getValidationCount() {
        return validationCount;
    }

    /**
     * @param validationCount
     *            the validationCount to set
     */
    public void setValidationCount(double validationCount) {
        this.validationCount = validationCount;
    }

}
