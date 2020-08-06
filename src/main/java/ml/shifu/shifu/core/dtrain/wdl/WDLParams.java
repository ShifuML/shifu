/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;

/**
 * {@link WDLParams} is message sent between master and workers for wide and deep model training.
 *
 * <p>
 * In worker, it will collect combined gradients and then send to master for merging. While in master when model weights
 * are updated, master will send model new weights to all works for next epoch iterations.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WDLParams extends HaltBytable implements Combinable<WDLParams> {

    private static final boolean WDL_IS_NULL = true;

    /**
     * # of weighted training records per such worker.
     */
    private double trainSize;

    /**
     * # of weighted validation records per such worker.
     */
    private double validationSize;

    /**
     * # of training records per such worker.
     */
    private double trainCount;

    /**
     * # of weighted validation records per such worker.
     */
    private double validationCount;

    /**
     * Train error for such worker and such iteration.
     */
    private double trainError;

    /**
     * Validation error for such worker and such iteration.
     */
    private double validationError;

    /**
     * Serialization type. Default to MODEL_SPEC.
     */
    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    private WideAndDeep wnd;

    public void update(WideAndDeep wnd) {
        this.getWnd().updateWeights(wnd);
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
     * @return the serializationType
     */
    public SerializationType getSerializationType() {
        return serializationType;
    }

    /**
     * @param serializationType
     *            the serializationType to set
     */
    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    @Override
    public WDLParams combine(WDLParams from) {
        this.trainCount += from.trainCount;
        this.trainError += from.trainError;
        this.validationCount += from.validationCount;
        this.validationError += from.validationError;
        this.trainSize += from.trainSize;
        this.validationSize += from.validationSize;
        // In the first iteration, the worker may send a empty WDLParams without WideAndDeep Init
        if(from.getWnd() != null) {
            this.wnd = this.wnd.combine(from.getWnd());
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(this.wnd == null) {
            // for the first iteration, the wnd will be null
            out.writeBoolean(WDL_IS_NULL);
        } else {
            out.writeBoolean(!WDL_IS_NULL);
            this.wnd.setSerializationType(serializationType);
            this.wnd.write(out);
        }
        out.writeDouble(this.trainCount);
        out.writeDouble(this.validationCount);
        out.writeDouble(this.trainError);
        out.writeDouble(this.validationError);
        out.writeDouble(this.trainSize);
        out.writeDouble(this.validationSize);
        out.writeInt(this.serializationType.getValue());
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        boolean wdlIsNull = in.readBoolean();
        if(!wdlIsNull) {
            if(this.wnd == null) {
                this.wnd = new WideAndDeep();
            }
            this.wnd.readFields(in);
        }
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
        this.trainError = in.readDouble();
        this.validationError = in.readDouble();
        this.trainSize = in.readDouble();
        this.validationSize = in.readDouble();
        this.serializationType = SerializationType.getSerializationType(in.readInt());
    }

    /**
     * @return the wnd
     */
    public WideAndDeep getWnd() {
        return wnd;
    }

    /**
     * @param wnd
     *            the wnd to set
     */
    public void setWnd(WideAndDeep wnd) {
        this.wnd = wnd;
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

}
