package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;
import ml.shifu.shifu.core.dtrain.SerializationType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author haillu
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

    private double[] trainErrors;

    private double[] validationErrors;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    private MultiTaskNN mtnn;


    @Override
    public MTNNParams combine(MTNNParams from) {
        this.trainCount += from.trainCount;
        for (int i = 0; i < trainErrors.length; i++) {
            this.trainErrors[i] += from.trainErrors[i];
        }
        this.validationCount += from.validationCount;
        for (int i = 0; i < validationErrors.length; i++) {
            this.validationErrors[i] += from.validationErrors[i];
        }
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
        for (int i = 0; i < trainErrors.length; i++) {
            out.writeDouble(trainErrors[i]);
        }
        for (int i = 0; i < validationErrors.length; i++) {
            out.writeDouble(validationErrors[i]);
        }
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
        for (int i = 0; i < trainErrors.length; i++) {
            this.trainErrors[i] = in.readDouble();
        }
        for (int i = 0; i < validationErrors.length; i++) {
            this.validationErrors[i] = in.readDouble();
        }
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

    public double[] getTrainErrors() {
        return trainErrors;
    }

    public void setTrainErrors(double[] trainError) {
        this.trainErrors = trainError;
    }

    public double[] getValidationErrors() {
        return validationErrors;
    }

    public void setValidationErrors(double[] validationErrors) {
        this.validationErrors = validationErrors;
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
