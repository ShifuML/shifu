package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author haillu
 * @date 7/17/2019 3:58 PM
 */
public class MTNNParams extends HaltBytable implements Combinable<MTNNParams> {
    @Override
    public MTNNParams combine(MTNNParams from) {
        return null;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {

    }

    @Override
    public void doReadFields(DataInput in) throws IOException {

    }
}
