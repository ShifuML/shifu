package ml.shifu.shifu.core.dtrain;

import java.util.Properties;
import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.shifu.core.dtrain.dt.DTMaster;
import ml.shifu.shifu.core.dtrain.dt.DTMasterParams;
import ml.shifu.shifu.core.dtrain.dt.DTWorker;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams;
import org.junit.Test;

public class GBDTTest {

    final private String modelConfigPath = "/data/gbdt-test/ModelConfig.json";
    final private String columnConfigPath = "/data/gbdt-test/ColumnConfig.json";
    final private String normDataPath = "/data/gbdt-test/data_norm";
    final private String iterationCount = "10";

    /**
     * Train NN model without any variable select.
     */
    @Test
    public void testTrain() {
        new GuaguaMRUnitDriver<>(generateProperties(modelConfigPath, columnConfigPath, normDataPath)).run();
    }

    private Properties generateProperties(String modelConfigPath, String columnConfigPath, String dataPath) {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, DTMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, DTWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, iterationCount);
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, DTMasterParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, DTWorkerParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG,
            getClass().getResource(modelConfigPath).toString());
        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
            getClass().getResource(columnConfigPath).toString());
        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR,
            getClass().getResource(dataPath).toString());
        return props;
    }
}
