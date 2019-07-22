package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;
import ml.shifu.shifu.core.dtrain.SerializationType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author haillu
 * @date 7/17/2019 3:58 PM
 */
public class MTNNParams extends HaltBytable implements Combinable<MTNNParams> {
    private static final boolean MTNN_IS_NULL = true;

    private double trainSize;

    private double validationSize;

    /**
     * # of training records per such worker.(Now, we don't use it in MTNN.)
     */
    private double trainCount;

    private double validationCount;

    private double trainError;

    private double validationError;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    private MultiTaskNN mtnn;


    @Override
    public MTNNParams combine(MTNNParams from) {
        this.trainCount += from.trainCount;
        this.trainError += from.trainError;
        this.validationCount += from.validationCount;
        this.validationError += from.validationError;
        if (from.getMtnn() != null) {
            this.mtnn = this.mtnn.combine(from.getMtnn());
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if (this.mtnn == null) {
            out.writeBoolean(MTNN_IS_NULL);
        } else {
            out.writeBoolean(!MTNN_IS_NULL);
            this.mtnn.setSerializationType(serializationType);
            this.mtnn.write(out);
        }
        out.writeDouble(this.trainCount);
        out.writeDouble(this.validationCount);
        out.writeDouble(this.trainError);
        out.writeDouble(this.validationError);
        out.writeInt(this.serializationType.getValue());

    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        boolean mtnnIsNull = in.readBoolean();
        if (!mtnnIsNull) {
            if (this.mtnn == null) {
                this.mtnn = new MultiTaskNN();
            }
            this.mtnn.readFields(in);
        }
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
        this.trainError = in.readDouble();
        this.validationError = in.readDouble();
        this.serializationType = SerializationType.getSerializationType(in.readInt());
    }


    public double getTrainCount() {
        return trainCount;
    }

    public void setTrainCount(double trainCount) {
        this.trainCount = trainCount;
    }

    public double getValidationCount() {
        return validationCount;
    }

    public void setValidationCount(double validationCount) {
        this.validationCount = validationCount;
    }

    public double getTrainSize() {
        return trainSize;
    }

    public void setTrainSize(double trainSize) {
        this.trainSize = trainSize;
    }

    public double getValidationSize() {
        return validationSize;
    }

    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    public double getTrainError() {
        return trainError;
    }

    public void setTrainError(double trainError) {
        this.trainError = trainError;
    }

    public double getValidationError() {
        return validationError;
    }

    public void setValidationError(double validationError) {
        this.validationError = validationError;
    }

    public SerializationType getSerializationType() {
        return serializationType;
    }

    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    public MultiTaskNN getMtnn() {
        return mtnn;
    }

    public void setMtnn(MultiTaskNN mtnn) {
        this.mtnn = mtnn;
    }
}
