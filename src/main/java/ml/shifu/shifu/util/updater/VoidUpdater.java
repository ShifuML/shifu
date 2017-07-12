package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

import java.io.IOException;

/**
 * Created by zhanhu on 2/23/17.
 */
public class VoidUpdater extends BasicUpdater {

    public VoidUpdater(ModelConfig modelConfig) throws IOException {
        super(modelConfig);
    }

    @Override
    public void updateColumnConfig(ColumnConfig columnConfig) {
        // do nothing
    }
}
