package ml.shifu.shifu.core.dtrain;

import java.util.Properties;
import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.shifu.core.dtrain.nn.NNMaster;
import ml.shifu.shifu.core.dtrain.nn.NNParams;
import ml.shifu.shifu.core.dtrain.nn.NNWorker;
import org.junit.Test;

public class NNTest {

    final private String modelConfigPath = "/data/nn-test/ModelConfig.json";
    final private String columnConfigPath = "/data/nn-test/ColumnConfig.json";
    final private String compactColumnConfigPath = "/data/nn-test/ColumnConfig_FinalSelect.json";
    final private String normDataPath = "/data/nn-test/data_norm";
    final private String compactNormDataPath = "/data/nn-test/data_norm_compact";
    final private String iterationCount = "10";

    /**
     * Train NN model without any variable select.
     */
    @Test
    public void testTrain() {
        new GuaguaMRUnitDriver<>(generateProperties(modelConfigPath, columnConfigPath, normDataPath)).run();
    }

    /**
     * Train NN model with variable select in column config and full normalized data.
     */
    @Test
    public void testTrainNormalDataWithSelect() {
        new GuaguaMRUnitDriver<>(generateProperties(modelConfigPath, compactColumnConfigPath, normDataPath)).run();
    }

    /**
     * Train NN model with variable select in column config and compact normalized data.
     */
    @Test
    public void testTrainCompactData() {
        new GuaguaMRUnitDriver<>(generateProperties(modelConfigPath, compactColumnConfigPath, compactNormDataPath)).run();
    }

    private Properties generateProperties(String modelConfigPath, String columnConfigPath, String dataPath) {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, NNMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, NNWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, iterationCount);
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, NNParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, NNParams.class.getName());
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
